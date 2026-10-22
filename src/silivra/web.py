"""Bounded, public-HTTPS retrieval and optional Brave search (stdlib only).

Retrieved pages and search snippets are untrusted data, never instructions.
Requests bypass environment proxies, pin a validated DNS address, and verify TLS
against the original hostname. No network request is made merely by importing.
"""

from __future__ import annotations

import html
from html.parser import HTMLParser
import http.client
import ipaddress
import json
import os
import queue
import socket
import ssl
import threading
import time
from urllib.parse import urlencode, urljoin, urlsplit, urlunsplit


class WebError(ValueError):
    """An intentionally credential-free network error."""


_DNS_SLOTS = threading.BoundedSemaphore(4)
_MAX_BYTES = 1_048_576
_TIMEOUT = 15.0
_USER_AGENT = "Silivra/0.2 (public knowledge retrieval)"


def _remaining(deadline: float) -> float:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise WebError("Network request exceeded its time budget.")
    return remaining


def _validate_url(url: str) -> tuple[str, str]:
    if not isinstance(url, str) or not url or len(url) > 4096:
        raise WebError("URL must be a nonempty HTTPS URL of at most 4096 characters.")
    if any(ord(c) < 33 or ord(c) == 127 for c in url) or "\\" in url:
        raise WebError("URL contains unsupported characters.")
    try:
        parts = urlsplit(url)
        if (parts.scheme.lower() != "https" or not parts.hostname
                or parts.username is not None or parts.password is not None
                or parts.port not in (None, 443)):
            raise WebError("Only public HTTPS URLs on port 443 without credentials are supported.")
        host = parts.hostname.encode("idna").decode("ascii").lower().rstrip(".")
        if not host or "%" in host:
            raise WebError("Unsupported hostname.")
        authority = f"[{host}]" if ":" in host else host
        return urlunsplit(("https", authority, parts.path or "/", parts.query, "")), host
    except (UnicodeError, ValueError) as error:
        if isinstance(error, WebError):
            raise
        raise WebError("Malformed HTTPS URL.") from None


def _public_address(address: str) -> bool:
    try:
        ip = ipaddress.ip_address(address)
        if isinstance(ip, ipaddress.IPv6Address):
            if ip.ipv4_mapped:
                ip = ip.ipv4_mapped
            elif ip in ipaddress.ip_network("64:ff9b::/96"):
                # Well-known NAT64 can otherwise turn a global IPv6 address
                # into a private IPv4 destination at the network translator.
                ip = ipaddress.IPv4Address(int(ip) & 0xffffffff)
            elif ip.sixtofour is not None:
                # Python's classification of transition ranges differs by
                # version. Validate the actual IPv4 tunnel endpoint too.
                embedded = ip.sixtofour
                if not embedded.is_global or embedded.is_multicast:
                    return False
            elif ip.teredo is not None:
                if any(not endpoint.is_global or endpoint.is_multicast
                       for endpoint in ip.teredo):
                    return False
        return ip.is_global and not ip.is_multicast
    except ValueError:
        return False


def _resolve(host: str, deadline: float) -> list[tuple]:
    """Bound DNS waits and outstanding resolver threads, including hung DNS."""
    if not _DNS_SLOTS.acquire(timeout=_remaining(deadline)):
        raise WebError("DNS resolution is temporarily busy.")
    result: queue.Queue = queue.Queue(maxsize=1)

    def work():
        try:
            result.put(socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM))
        except OSError:
            result.put(None)
        finally:
            _DNS_SLOTS.release()

    threading.Thread(target=work, daemon=True).start()
    try:
        addresses = result.get(timeout=_remaining(deadline))
    except queue.Empty:
        raise WebError("DNS resolution exceeded its time budget.") from None
    if not addresses:
        raise WebError("Hostname could not be resolved.")
    # Reject mixed public/private DNS answers instead of choosing the public one.
    if any(not _public_address(item[4][0]) for item in addresses):
        raise WebError("URLs resolving to nonpublic addresses are blocked.")
    return addresses


class _PinnedHTTPSConnection(http.client.HTTPSConnection):
    def __init__(self, host: str, address: tuple, timeout: float):
        super().__init__(host, timeout=timeout, context=ssl.create_default_context())
        self.address = address
        self._transport = None
        self._aborted = False

    def connect(self):
        family, kind, protocol, _, sockaddr = self.address
        raw = socket.socket(family, kind, protocol)
        self.sock = raw  # Expose to the deadline watchdog during TLS setup too.
        self._transport = raw
        try:
            if self._aborted:
                raise TimeoutError
            raw.settimeout(self.timeout)
            raw.connect(sockaddr)
            self.sock = self._context.wrap_socket(raw, server_hostname=self.host)
            self._transport = self.sock
            if self._aborted:
                raise TimeoutError
        except BaseException:
            raw.close()
            raise


def _abort(connection):
    connection._aborted = True
    sock = connection._transport
    if sock is not None:
        try:
            sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
    connection.close()


def _fetch_bytes(url: str, *, headers: dict | None = None,
                 max_bytes: int = _MAX_BYTES, timeout: float = _TIMEOUT) -> dict:
    deadline = time.monotonic() + timeout
    current, original_host = _validate_url(url)
    for redirect in range(5):
        current, host = _validate_url(current)
        addresses = _resolve(host, deadline)
        connection = _PinnedHTTPSConnection(host, addresses[0], _remaining(deadline))
        watchdog = threading.Timer(_remaining(deadline), _abort, args=(connection,))
        watchdog.daemon = True
        watchdog.start()
        try:
            parts = urlsplit(current)
            request_headers = {"User-Agent": _USER_AGENT, "Accept-Encoding": "identity"}
            for key, value in (headers or {}).items():
                # Provider credentials never cross a redirect to another host.
                if host == original_host or key.lower() in {"accept", "user-agent"}:
                    request_headers[key] = value
            connection.request("GET", urlunsplit(("", "", parts.path, parts.query, "")),
                               headers=request_headers)
            response = connection.getresponse()
            if response.status in {301, 302, 303, 307, 308}:
                location = response.getheader("Location")
                if not location or redirect == 4:
                    raise WebError("Redirect limit exceeded or redirect URL is missing.")
                current = urljoin(current, location)
                continue
            if response.status != 200:
                raise WebError(f"Remote server returned HTTP {response.status}.")
            if response.getheader("Content-Encoding", "identity").lower() != "identity":
                raise WebError("Compressed responses are not supported.")
            declared = response.getheader("Content-Length")
            if declared is not None:
                try:
                    if int(declared) < 0 or int(declared) > max_bytes:
                        raise WebError("Response exceeds the byte budget.")
                except ValueError as error:
                    if isinstance(error, WebError):
                        raise
                    raise WebError("Invalid response length.") from None
            chunks, size = [], 0
            while True:
                if connection._transport is not None:
                    connection._transport.settimeout(_remaining(deadline))
                else:
                    _remaining(deadline)
                chunk = response.read1(min(8192, max_bytes + 1 - size))
                if not chunk:
                    break
                size += len(chunk)
                if size > max_bytes:
                    raise WebError("Response exceeds the byte budget.")
                chunks.append(chunk)
            _remaining(deadline)
            return {"url": current, "body": b"".join(chunks),
                    "content_type": response.getheader("Content-Type", "").lower()}
        except WebError:
            raise
        except (OSError, http.client.HTTPException, ValueError):
            raise WebError("HTTPS request failed or exceeded its time budget.") from None
        finally:
            watchdog.cancel()
            connection.close()
    raise WebError("Redirect limit exceeded.")


class _PageText(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []
        self.title_parts = []
        self.hidden = 0
        self.in_title = False

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style", "noscript", "template"}:
            self.hidden += 1
        if tag == "title":
            self.in_title = True
        if tag in {"p", "div", "br", "li", "h1", "h2", "h3", "tr", "section"}:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in {"script", "style", "noscript", "template"}:
            self.hidden = max(0, self.hidden - 1)
        if tag == "title":
            self.in_title = False
        if tag in {"p", "div", "li", "h1", "h2", "h3", "tr", "section"}:
            self.parts.append("\n")

    def handle_data(self, data):
        if not self.hidden:
            self.parts.append(data)
            if self.in_title:
                self.title_parts.append(data)


def _clean_text(text: str) -> str:
    return "\n".join(line for part in text.splitlines() if (line := " ".join(part.split())))


def web_fetch(url: str, *, max_chars: int = 20000) -> dict:
    if type(max_chars) is not int or not 100 <= max_chars <= 50000:
        raise WebError("max_chars must be an integer between 100 and 50000.")
    response = _fetch_bytes(url)
    mime = response["content_type"].split(";", 1)[0].strip()
    if mime not in {"text/html", "application/xhtml+xml", "text/plain", "text/markdown",
                    "application/json"}:
        raise WebError("Only HTML, plain text, Markdown, and JSON pages are supported.")
    text = response["body"].decode("utf-8", errors="replace")
    title = ""
    if mime in {"text/html", "application/xhtml+xml"}:
        parser = _PageText()
        parser.feed(text)
        text = _clean_text("".join(parser.parts))
        title = " ".join("".join(parser.title_parts).split())[:300]
    return {"url": response["url"], "title": title, "text": text[:max_chars],
            "truncated": len(text) > max_chars, "untrusted": True,
            "source_type": "external_web_page"}


def _json_response(response):
    try:
        return json.loads(response["body"])
    except (ValueError, UnicodeError):
        raise WebError("Search provider returned invalid JSON.") from None


def web_search(query: str, *, limit: int = 5) -> dict:
    if not isinstance(query, str) or not query.strip() or len(query) > 600 or len(query.split()) > 75:
        raise WebError("Search query must contain 1 to 600 characters and at most 75 words.")
    if type(limit) is not int or not 1 <= limit <= 10:
        raise WebError("Search limit must be an integer between 1 and 10.")
    key = os.environ.get("BRAVE_SEARCH_API_KEY", "").strip()
    if key:
        response = _fetch_bytes("https://api.search.brave.com/res/v1/web/search?" +
                                urlencode({"q": query, "count": limit}),
                                headers={"Accept": "application/json", "X-Subscription-Token": key})
        data = _json_response(response)
        try:
            results = [{"title": str(item.get("title", ""))[:300],
                        "url": str(item.get("url", ""))[:4096],
                        "snippet": html.unescape(str(item.get("description", "")))[:1500]}
                       for item in data.get("web", {}).get("results", [])[:limit]]
        except (AttributeError, TypeError):
            raise WebError("Search provider returned an unexpected result format.") from None
        return {"provider": "brave", "coverage": "web", "query": query,
                "results": results, "untrusted": True}
    response = _fetch_bytes("https://en.wikipedia.org/w/api.php?" + urlencode({
        "action": "opensearch", "search": query, "limit": limit, "namespace": 0, "format": "json"}),
        headers={"Accept": "application/json"})
    data = _json_response(response)
    try:
        if (not isinstance(data, list) or len(data) != 4
                or not all(isinstance(row, list) for row in data[1:])):
            raise ValueError
        results = [{"title": str(title)[:300], "url": str(url)[:4096],
                    "snippet": str(snippet)[:1500]}
                   for title, snippet, url in zip(data[1], data[2], data[3])][:limit]
    except (TypeError, ValueError):
        raise WebError("Knowledge search returned an unexpected result format.") from None
    return {"provider": "wikipedia", "coverage": "encyclopedia_only", "query": query,
            "notice": "Wikipedia knowledge search only. Configure BRAVE_SEARCH_API_KEY for whole-web search.",
            "results": results, "untrusted": True}
