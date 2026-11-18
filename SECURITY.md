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

otari is a self-hosted AI gateway; several controls are configuration-dependent:

- **Pricing is required by default.** `require_pricing` defaults to `true`:
  requests for models with no configured pricing are rejected (HTTP 402) so an
  unpriced model cannot be served free and unmetered. Operators running
  genuinely free / self-hosted models must add an explicit `$0` pricing entry,
  or set `require_pricing=false` to opt out.
- **Budgeted inference uses atomic reservations.** Concurrent token-priced and
  image requests reserve estimated cost before dispatch. Retrospective imports,
  batches, and currently unreserved flat-price paths are not hard real-time
  caps; see [Access control](docs/access-control.md#budgets).
- **The client `user` field is not trusted for non-master keys.** Spend is bound
  to the API key's own user; only the master key may bill on behalf of an
  arbitrary user. By default a non-master key naming a different user is rejected
  (403); set `reject_user_mismatch=false` on a single key, or deployment-wide, to
  instead bind to the key's own user. The supplied name never reaches a provider
  as the client wrote it: where a route sends an end-user identifier it is one
  input to a digest, and elsewhere it is dropped.
- **Audio and moderation endpoints** have no token-based pricing model and are
  exempt from `require_pricing` (treated as $0 when unpriced).
- **MCP servers and guardrail URLs are tenant destinations.** A workspace, an
  organization or an API key holder chooses them, not the operator, so Otari
  connects only to public addresses. Loopback, private, link-local and other
  internal addresses are refused. The address is checked when a URL is saved or sent and again on every
  connection, and the connection is made to the address that was checked, so a
  name re-pointed afterwards reaches nothing internal. The same holds for the
  sign-in addresses an MCP server publishes. Redirects are not followed, except
  that the managed tool loop follows one that stays on the server's origin,
  checked again. These connections do not use `HTTP_PROXY` or `HTTPS_PROXY`,
  since a proxy would resolve the name again after Otari checked it, so a
  deployment that reaches the internet only through a proxy cannot reach MCP
  servers. They also ignore `SSL_CERT_FILE` and `SSL_CERT_DIR` and trust only
  the default certificate store, so an MCP server, including one listed in
  `mcp_private_hosts`, behind a certificate from a private CA fails its TLS
  handshake. The operator's own `guardrails_url` is not held to this rule.
- **`mcp_private_hosts` is the one exception, and it is the operator's.** It
  lists exact hosts, each with an optional port, that an MCP server URL (and
  the sign-in addresses it leads to) may name on a loopback or private address.
  Such a destination is still checked on every connection and dialed at the
  checked address, and link-local, metadata and unspecified addresses are
  refused even for a listed host. **Every tenant who can name an MCP server can
  reach a listed host**, so it is for a single-tenant or trusted deployment. It
  is empty by default, read only from the environment or the config file, never
  writable through the dashboard or the API, and guardrail URLs never read it.
  Otherwise, an MCP server on a private network is reached through a tunnel
  ([guide](docs/connect-a-private-mcp-server.md)).
- **An address inside a stored guardrail definition** must be public, and a
  definition that names its own address must also carry its own credentials,
  as non-empty text: the operator's environment variables (`LAKERA_API_KEY`,
  `OPENAI_API_KEY`, `WATSONX_APIKEY` and the like) never fill a gap and travel
  to an address an organization wrote. Its host must be plain letters, digits,
  dots and hyphens or an IP address, with no user information or backslash, so
  every HTTP library reads the same host. Both rules run when a definition is saved and again
  each time its guardrail is built, so a row saved before them is refused
  rather than used. For `watsonx_guardian` and `openai_moderation`, every
  connection is also checked and dialed at the checked address. The other
  definitions that take an address (`alinia`, `lakera_guard`, `patronus`,
  `azure_prompt_shields`, `azure_content_safety`) are dialed by the vendor's
  own library, which follows redirects and gives Otari no client to pin, so
  their address must be on the vendor's own hosts: `api.lakera.ai` and its
  documented regional hosts, `api.patronus.ai`, a subdomain of
  `cognitiveservices.azure.com`. Alinia publishes no API host, so an Alinia
  definition cannot name an endpoint and uses the operator's. What remains is
  the vendor's own domain resolving or redirecting somewhere it should not,
  which is the vendor's to prevent, not a tenant's to cause.
