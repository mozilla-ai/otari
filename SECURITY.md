# Security Policy

## Reporting a Vulnerability

We take the security of otari seriously. If you believe you have found a
security vulnerability, please report it to us privately.

**Please do not report security vulnerabilities through public GitHub issues,
discussions, or pull requests.**

Instead, please use one of the following channels:

- Open a private [GitHub Security Advisory](https://github.com/mozilla-ai/otari/security/advisories/new)
  on this repository, or
- Email **security@mozilla.ai** with the details.

Please include as much of the following as you can:

- The type of issue (e.g. budget bypass, authentication/authorization flaw,
  injection, SSRF, etc.).
- The affected component and version/commit.
- Step-by-step instructions to reproduce, and a proof-of-concept if available.
- The impact of the issue, including how an attacker might exploit it.

Test only against your own self-hosted instance. Do not run scans or send
exploit traffic against any mozilla.ai-operated infrastructure.

## Disclosure Process

- We will acknowledge receipt of your report within a few business days.
- We will investigate and keep you informed of our progress.
- Once a fix is available, we will coordinate a disclosure timeline with you and
  credit you in the advisory (unless you prefer to remain anonymous).

## Scope Notes for Operators

otari is a self-hosted LLM gateway; several controls are configuration-dependent:

- **Pricing is required by default.** `require_pricing` defaults to `true`:
  requests for models with no configured pricing are rejected (HTTP 402) so an
  unpriced model cannot be served free and unmetered. Operators running
  genuinely free / self-hosted models must add an explicit `$0` pricing entry,
  or set `require_pricing=false` to opt out.
- **Budgets are enforced via atomic pre-debit reservations.** Concurrent
  requests cannot collectively exceed `max_budget`.
- **The client `user` field is not trusted for non-master keys.** Spend is bound
  to the API key's own user; only the master key may bill on behalf of an
  arbitrary user. By default a non-master key naming a different user is rejected
  (403); set `reject_user_mismatch=false` to instead bind to the key's own user
  while still forwarding `user` to the provider as an end-user tag.
- **Audio and moderation endpoints** have no token-based pricing model and are
  exempt from `require_pricing` (treated as $0 when unpriced).
