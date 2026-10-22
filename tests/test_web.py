"""Offline web tests: DNS rebinding defense, redirect/byte limits, and providers."""

import json
import socket
import unittest
from unittest.mock import MagicMock, PropertyMock, patch

from silivra import web


PUBLIC = [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))]
PRIVATE = [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 443))]


class FakeResponse:
    def __init__(self, body=b"", *, status=200, headers=None):
        self.status = status
        self.headers = headers or {"Content-Type": "text/plain"}
        self.body = body

    def getheader(self, key, default=None):
        return self.headers.get(key, default)

    def read1(self, size):
        part, self.body = self.body[:size], self.body[size:]
        return part


class FakeConnection:
    responses = []
    requests = []

    def __init__(self, host, address, timeout):
        self.host = host
        self.address = address
        self._transport = None

    def request(self, *args, **kwargs):
        self.requests.append((self.host, args, kwargs))

    def getresponse(self):
        return self.responses.pop(0)

    def close(self):
        pass


class WebTests(unittest.TestCase):
    def setUp(self):
        FakeConnection.responses = []
        FakeConnection.requests = []

    def fetch_fake(self, response, **kwargs):
        FakeConnection.responses = [response]
        with patch.object(web, "_resolve", return_value=PUBLIC), patch.object(web, "_PinnedHTTPSConnection", FakeConnection):
            return web._fetch_bytes("https://example.com/path", **kwargs)

    def test_nonpublic_addresses_and_unusual_urls_are_rejected(self):
        for address in ("127.0.0.1", "10.2.3.4", "172.16.0.1", "192.168.1.1", "169.254.169.254",
                        "100.64.0.1", "0.0.0.0", "::1", "::", "fc00::1", "fe80::1", "::ffff:127.0.0.1", "224.1.1.1", "64:ff9b::a00:1"):
            self.assertFalse(web._public_address(address), address)
        self.assertTrue(web._public_address("93.184.216.34"))
        for url in ("http://example.com", "file:///etc/passwd", "https://user:secret@example.com",
                    "https://example.com:8443", "https://example.com\n/path", "https://example.com\\@evil.com", "https://[fe80::1%25en0]/"):
            with self.subTest(url=url), self.assertRaises(web.WebError):
                web._validate_url(url)

    def test_dns_with_any_private_address_is_blocked(self):
        with patch.object(socket, "getaddrinfo", return_value=PUBLIC + PRIVATE):
            with self.assertRaisesRegex(web.WebError, "nonpublic"):
                web._resolve("example.com", web.time.monotonic() + 1)

    def test_ipv6_tunnel_endpoints_are_checked_independently_of_python_version(self):
        # Older Python releases classified transition IPv6 ranges differently.
        # Force an outer global classification to exercise the embedded checks.
        with patch.object(web.ipaddress.IPv6Address, "is_global", new_callable=PropertyMock, return_value=True):
            for address in ("2002:7f00:1::", "2002:a00:1::", "2002:e000:101::",
                            "2001:0:0808:0808:0:0:80ff:fffe",  # Teredo client 127.0.0.1.
                            "2001:0:7f00:0001:0:0:f7f7:f7f7"):  # Teredo server 127.0.0.1.
                with self.subTest(address=address):
                    self.assertFalse(web._public_address(address))

    def test_validated_address_is_used_directly_and_tls_hostname_is_preserved(self):
        transport = MagicMock()
        context = MagicMock()
        with patch.object(web.ssl, "create_default_context", return_value=context), patch.object(web.socket, "socket", return_value=transport), patch.object(web.socket, "getaddrinfo") as resolver:
            connection = web._PinnedHTTPSConnection("example.com", PUBLIC[0], 3)
            connection.connect()
        resolver.assert_not_called()
        transport.connect.assert_called_once_with(("93.184.216.34", 443))
        context.wrap_socket.assert_called_once_with(transport, server_hostname="example.com")

    def test_redirect_dns_is_revalidated_before_connect(self):
        FakeConnection.responses = [FakeResponse(status=302, headers={"Location": "https://private.example/"})]
        with patch.object(socket, "getaddrinfo", side_effect=[PUBLIC, PRIVATE]), patch.object(web, "_PinnedHTTPSConnection", FakeConnection):
            with self.assertRaisesRegex(web.WebError, "nonpublic"):
                web._fetch_bytes("https://example.com/")
        self.assertEqual(len(FakeConnection.requests), 1)

    def test_provider_authorization_is_not_forwarded_to_redirect_host(self):
        FakeConnection.responses = [FakeResponse(status=302, headers={"Location": "https://other.example/"}), FakeResponse(b"done")]
        with patch.object(web, "_resolve", return_value=PUBLIC), patch.object(web, "_PinnedHTTPSConnection", FakeConnection):
            result = web._fetch_bytes("https://example.com/", headers={"X-Subscription-Token": "secret", "Accept": "application/json"})
        self.assertEqual(result["body"], b"done")
        self.assertIn("X-Subscription-Token", FakeConnection.requests[0][2]["headers"])
        self.assertNotIn("X-Subscription-Token", FakeConnection.requests[1][2]["headers"])

    def test_byte_declared_length_compression_status_and_redirect_budgets(self):
        for response in (FakeResponse(b"x" * 12), FakeResponse(headers={"Content-Length": "12"}),
                         FakeResponse(headers={"Content-Length": "garbage"}),
                         FakeResponse(headers={"Content-Encoding": "gzip"}), FakeResponse(status=403)):
            with self.subTest(headers=response.headers, status=response.status), self.assertRaises(web.WebError):
                self.fetch_fake(response, max_bytes=10)
        FakeConnection.responses = [FakeResponse(status=302, headers={"Location": "/again"}) for _ in range(5)]
        with patch.object(web, "_resolve", return_value=PUBLIC), patch.object(web, "_PinnedHTTPSConnection", FakeConnection):
            with self.assertRaisesRegex(web.WebError, "Redirect"):
                web._fetch_bytes("https://example.com")

    def test_network_errors_do_not_echo_credentials_or_url(self):
        class Broken(FakeConnection):
            def request(self, *args, **kwargs):
                raise OSError("provider-secret query-private-data")
        with patch.object(web, "_resolve", return_value=PUBLIC), patch.object(web, "_PinnedHTTPSConnection", Broken):
            with self.assertRaises(web.WebError) as caught:
                web._fetch_bytes("https://example.com/?private=query", headers={"X-Subscription-Token": "provider-secret"})
        self.assertNotIn("secret", str(caught.exception))
        self.assertNotIn("query", str(caught.exception))

    def test_html_extraction_excludes_script_and_marks_untrusted_content(self):
        response = {"url": "https://example.com/", "content_type": "text/html; charset=utf-8",
                    "body": b"<title>Example &amp; Title</title><script>SECRET</script><style>SECRET</style><h1>Welcome</h1><p>Hello <b>world</b></p>"}
        with patch.object(web, "_fetch_bytes", return_value=response):
            result = web.web_fetch("https://example.com/", max_chars=100)
        self.assertEqual(result["title"], "Example & Title")
        self.assertIn("Hello world", result["text"])
        self.assertNotIn("SECRET", result["text"])
        self.assertTrue(result["untrusted"])
        response["content_type"] = "application/pdf"
        with patch.object(web, "_fetch_bytes", return_value=response), self.assertRaises(web.WebError):
            web.web_fetch("https://example.com")

    def test_brave_configuration_search_query_and_response(self):
        response = {"body": json.dumps({"web": {"results": [{"title": "Title", "url": "https://example.com", "description": "A &amp; B"}]}}).encode()}
        with patch.dict(web.os.environ, {"BRAVE_SEARCH_API_KEY": "private-key"}), patch.object(web, "_fetch_bytes", return_value=response) as fetch:
            result = web.web_search("python docs", limit=2)
        self.assertEqual(result["coverage"], "web")
        self.assertEqual(result["provider"], "brave")
        self.assertEqual(result["results"][0]["snippet"], "A & B")
        self.assertIn("q=python+docs", fetch.call_args.args[0])
        self.assertEqual(fetch.call_args.kwargs["headers"]["X-Subscription-Token"], "private-key")

    def test_no_key_is_explicitly_wikipedia_knowledge_search(self):
        response = {"body": json.dumps(["python", ["Python"], ["Language"], ["https://en.wikipedia.org/wiki/Python"]]).encode()}
        with patch.dict(web.os.environ, {"BRAVE_SEARCH_API_KEY": ""}), patch.object(web, "_fetch_bytes", return_value=response) as fetch:
            result = web.web_search("python")
        self.assertEqual(result["coverage"], "encyclopedia_only")
        self.assertIn("knowledge search only", result["notice"])
        self.assertIn("en.wikipedia.org/w/api.php", fetch.call_args.args[0])
        self.assertTrue(result["untrusted"])

    def test_invalid_search_and_response_arguments_are_explicit(self):
        for query, limit in (("", 5), ("x" * 601, 5), ("a " * 76, 5), ("python", True), ("python", 11)):
            with self.subTest(query=query, limit=limit), self.assertRaises(web.WebError):
                web.web_search(query, limit=limit)
        with patch.dict(web.os.environ, {"BRAVE_SEARCH_API_KEY": ""}), patch.object(web, "_fetch_bytes", return_value={"body": b"<html>error</html>"}), self.assertRaises(web.WebError):
            web.web_search("python")


if __name__ == "__main__":
    unittest.main()
