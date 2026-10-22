# Security boundaries

Silivra's assistant is experimental. File tools enforce workspace and sensitive-path
rules. Network/write/command capabilities require explicit flags. Local command
execution runs with your OS permissions and is not a sandbox. Review the
[architecture](docs/ARCHITECTURE.md) before enabling it.

Private memory and transcripts stay in `.silivra/` by default and must not be
committed. Durable facts require an explicit user action. Search/page content,
workspace files, memories and tool responses are untrusted model inputs.

When reporting a suspected vulnerability, describe the affected version, required
permissions, source-to-effect path and a minimal reproduction with synthetic data.
Do not include real credentials or private conversations in a public GitHub issue.
