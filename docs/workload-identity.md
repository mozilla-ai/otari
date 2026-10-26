# Workload identity

A backend running on Google Cloud (Cloud Run, GKE, Compute Engine) can
authenticate to Otari with the identity token its platform already gives it,
instead of an API key's secret. The backend then holds no Otari secret at all:
nothing to store in Secret Manager, nothing to rotate, nothing to leak.

It works the way Workload Identity Federation does for other services. The
operator lists an OpenID Connect issuer the deployment trusts, and gives an API
key one or more **federation rules**, each naming an identity. A request that
presents a token from that issuer, matching one of the key's rules,
authenticates as that key. Everything the key carries still applies: its
active flag and expiry, its user being blocked, its model allow-list, its
budgets and ceilings, and whether it is a service key.

Workload identity needs `OTARI_SECRET_KEY`, which signs the access tokens of the
[token exchange](#claude-code-and-the-anthropic-sdks): with `federation_issuers`
set and no `OTARI_SECRET_KEY`, the gateway refuses to start.

## How a token is checked

When `federation_issuers` is empty, the default, none of this runs and every
credential is checked as an API key. Once it lists an issuer, a credential
shaped like a signed JWT is checked as an identity token:

1. Its issuer (`iss`) must be one of `federation_issuers`, compared exactly.
2. Its signature must verify against the issuer's published signing keys
   (JWKS), with an asymmetric algorithm: RS256, RS384, RS512, PS256, PS384,
   PS512, ES256, ES384, ES512 or EdDSA. `none` and the HMAC algorithms are refused,
   a key is used only with the algorithm it is published for, and an RSA key
   shorter than 2048 bits is refused.
3. It must carry `iss`, `sub`, `aud`, `exp` and `iat`. It must not be expired,
   and neither `iat` nor `nbf` may be in the future, each with 60 seconds of
   allowed clock skew. Its `exp` may be at most 24 hours away: platform identity
   tokens last an hour, and one that lasts days is a static secret in effect.
4. It must match a federation rule: the rule's issuer, an `aud` value equal to
   the rule's audience, and each of the rule's `subject` and `email` that is
   set. An email matches only when the token's `email_verified` is `true`.
5. The rules that match must all belong to one key. A token matching rules on
   two keys authenticates as neither.

Any failure answers `401 Invalid API key`, the same as an unknown key, so a
caller cannot tell which check failed. The `gateway_auth_failures` metric
records which one did, under a `federation_*` reason. When the issuer's keys
cannot be fetched and none are cached, the request answers `503` and is never
let through.

An identity token is never the master key: management routes that need the
master key refuse it. Routes for a service key's own end users accept it when
the key it matches is a service key.

### Signing keys

Otari reads the issuer's signing-key location from
`<issuer>/.well-known/openid-configuration`, whose `issuer` must equal the
configured one, unless the entry names a `jwks_uri` itself. Both are fetched
over https only, without following redirects, through the same outbound
address checks as other gateway fetches: a host that resolves to a private,
loopback or link-local address is refused unless the
`federation_allow_private_hosts` setting (`OTARI_FEDERATION_ALLOW_PRIVATE_HOSTS`)
is on, for an issuer on your own network. Like `federation_issuers`, it is read
and validated when the gateway starts.

Keys are cached for an hour per process. A token no cached key verifies
refetches them at most once a minute, which picks up a rotation without
letting made-up key ids turn each request into a fetch. A failed fetch is
retried after a minute; until one succeeds, keys still within their hour
keep working and nothing else does.

## Set it up on Google Cloud

The example runs a backend on Cloud Run as the service account
`hackbot-api@my-project.iam.gserviceaccount.com`, calling an Otari served at
`https://otari.example.com`.

### 1. Trust Google as an issuer

In `config.yml`:

```yaml
federation_issuers:
  - https://accounts.google.com
```

Or in the environment:

```bash
export OTARI_FEDERATION_ISSUERS='["https://accounts.google.com"]'
```

An entry may also be a mapping, `{issuer: ..., jwks_uri: ...}`, to name the
signing-key location rather than discover it. Restart the gateway to apply it.

### 2. Create the key and give it a rule

Create the key as usual, with the user, budgets and allow-list the backend
should have, then give it a rule. The audience is any string you choose; using
the gateway's URL keeps it recognizable. Naming both the service account's
email and its unique ID is the strictest rule: the unique ID changes if the
account is deleted and recreated under the same email.

```bash
gcloud iam service-accounts describe hackbot-api@my-project.iam.gserviceaccount.com \
  --format='value(uniqueId)'
# 113051416744238958061

curl -X POST "https://otari.example.com/api/v1/keys/$KEY_ID/federation-rules" \
  -H "Otari-Key: Bearer $OTARI_MASTER_KEY" \
  -H "Content-Type: application/json" \
  -d '{
        "issuer": "https://accounts.google.com",
        "audience": "https://otari.example.com",
        "subject": "113051416744238958061",
        "email": "hackbot-api@my-project.iam.gserviceaccount.com"
      }'
```

A rule names a `subject`, an `email`, or both; an issuer and audience alone are
refused, because anyone with a Google account can mint a token for any
audience. The issuer must be on `federation_issuers`, and one identity cannot
be given to two keys: a rule that shares a subject or an email with another
key's rule for the same issuer and audience, and contradicts it in neither,
answers `409`. Otari cannot tell whether a rule naming only a subject and
another naming only an email are the same account, so it accepts both, and a
token matching both authenticates as neither.
`GET /api/v1/keys/{key_id}/federation-rules` lists a
key's rules and `DELETE /api/v1/keys/{key_id}/federation-rules/{rule_id}`
removes one, effective with the next request. Deleting the key deletes its
rules. These routes need the deployment operator's standing, like the rest of
`/api/v1/keys`. On a deployment that turns on `federation_member_rules`, a
member can give their own keys rules instead; see
[Rules on a member's own key](#rules-on-a-members-own-key).

Once the rule is in place, the key's secret is not needed by the backend.
Keep it out of the backend's environment entirely.

### 3. Run the backend as that service account

```bash
gcloud run deploy hackbot-api \
  --image=REGION-docker.pkg.dev/my-project/hackbot/api \
  --service-account=hackbot-api@my-project.iam.gserviceaccount.com
```

On GKE, bind the Kubernetes service account to the Google one with Workload
Identity; on Compute Engine, attach the service account to the instance.

### 4. Present the token

The metadata server issues the token. Ask for `format=full`, which is what
puts the `email` and `email_verified` claims in it:

```bash
TOKEN=$(curl -s -H "Metadata-Flavor: Google" \
  "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/identity?audience=https://otari.example.com&format=full")

curl https://otari.example.com/api/v1/chat/completions \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"model": "openai:gpt-4o", "messages": [{"role": "user", "content": "Hello"}]}'
```

The token goes wherever a key would: `Authorization: Bearer`, `Otari-Key`, or
`x-api-key`. It expires after an hour, so a long-running process refreshes it.
In Python, `google-auth` asks for `format=full` and refreshes for you:

```python
import google.auth.transport.requests
from google.oauth2 import id_token
from openai import OpenAI

AUDIENCE = "https://otari.example.com"
request = google.auth.transport.requests.Request()
credentials = id_token.fetch_id_token_credentials(AUDIENCE, request=request)


def otari() -> OpenAI:
    """A client holding a token that is valid now."""
    if not credentials.valid:
        credentials.refresh(request)
    return OpenAI(base_url=f"{AUDIENCE}/api/v1", api_key=credentials.token)


reply = otari().chat.completions.create(model="openai:gpt-4o", messages=[{"role": "user", "content": "Hello"}])
```

## Claude Code and the Anthropic SDKs

Claude Code and the Anthropic SDKs speak workload identity themselves: given an
identity token and a federation rule, they exchange the token for a
short-lived access token at `{ANTHROPIC_BASE_URL}/v1/oauth/token` and send that
token on every request. Otari answers that exchange at
`POST /api/v1/oauth/token`, so a workload already set up this way for Anthropic
moves to Otari by changing two variables:

```bash
unset ANTHROPIC_API_KEY ANTHROPIC_AUTH_TOKEN
export ANTHROPIC_BASE_URL="https://otari.example.com/api"
export ANTHROPIC_FEDERATION_RULE_ID="<the id of the Otari federation rule>"
export ANTHROPIC_ORGANIZATION_ID="<the UUID of the key's Otari organization>"
export ANTHROPIC_IDENTITY_TOKEN_FILE="/var/run/otari/identity-token"
claude
```

Unset `ANTHROPIC_API_KEY` and `ANTHROPIC_AUTH_TOKEN` first. In the Anthropic
SDKs either one, when set, takes precedence over the federation variables,
which are then ignored with only a one-time warning on the `anthropic` logger;
Claude Code checks them ahead of the federation variables too.

- `ANTHROPIC_FEDERATION_RULE_ID` is the `id` that
  `POST /api/v1/keys/{key_id}/federation-rules` returned (and
  `GET /api/v1/keys/{key_id}/federation-rules` lists). The identity token must
  match **that** rule; a token matching only another rule is refused.
- `ANTHROPIC_ORGANIZATION_ID` is the UUID of the Otari organization the key
  belongs to: `organization.id` in `GET /api/v1/organizations/me`. That route
  answers the caller's own organization, so read it with the authority that
  manages the key: the master key, or a dashboard session of the operator.
  When set, it must be that organization.
- `ANTHROPIC_IDENTITY_TOKEN_FILE` (or `ANTHROPIC_IDENTITY_TOKEN`) holds the
  identity token; see [Keeping the token file fresh](#keeping-the-token-file-fresh).

Both are Otari's ids, not Anthropic's, so a deployment moving from Anthropic
replaces both values. `ANTHROPIC_WORKSPACE_ID` and `ANTHROPIC_SERVICE_ACCOUNT_ID`
can stay as they are: Otari ignores them, since the rule alone decides the key.

The access token lasts `federation_access_token_ttl_sec` seconds (600 by
default, between 60 and 3600), and never past the identity token it was
exchanged for; the SDKs refresh it before then. It authenticates exactly as the
rule's key, the same way an identity token presented directly does, and it is
checked again on every request: deleting the rule, deactivating the key, the
key expiring, removing an operator's rule's issuer from `federation_issuers`,
or turning `federation_member_rules` off for a member's rule cuts it off at
once. It is never the master key.

The access token is stateless. Otari signs it with HMAC-SHA256 under a key
HKDF derives from `OTARI_SECRET_KEY`, and stores nothing, so every replica that
shares that secret accepts it, and rotating the secret (prepending a new key)
keeps tokens signed with the old one valid until they expire. This is why
federation needs `OTARI_SECRET_KEY` to start.

A refused exchange answers in the OAuth error shape. A request that is not the
`urn:ietf:params:oauth:grant-type:jwt-bearer` grant is
`400 unsupported_grant_type`; one without an `assertion` or a
`federation_rule_id` is `400 invalid_request`. Every refusal of the identity
token itself, including a rule, organization, workspace or service account
that does not match, is `401 invalid_grant` with one generic description,
which the SDKs report with a hint to check the federation rule. A refusal
writes nothing to the database.

The endpoint takes no credential, since the identity token is one, and is
mounted only when `federation_issuers` lists an issuer or
`federation_member_rules` is on: otherwise it answers 404.

### Keeping the token file fresh

Claude Code and the SDKs reread `ANTHROPIC_IDENTITY_TOKEN_FILE` at every
exchange, but on Cloud Run (and Compute Engine) nothing writes that file: the
metadata server hands out tokens on request. Run a small refresher beside the
workload. It asks the metadata server for a token with `format=full`, through
`google-auth`, writes it to a 0600 file atomically, so a reader never sees half
a token, and rewrites it every 30 minutes, well inside the token's hour.
Hackbot's runtime already does this for its Anthropic credentials.

```python
import os
import tempfile
import threading

import google.auth.transport.requests
from google.oauth2 import id_token

AUDIENCE = "https://otari.example.com"
TOKEN_FILE = "/var/run/otari/identity-token"
REFRESH_SECONDS = 30 * 60


def write_token() -> None:
    # On Google Cloud this reads the metadata server's identity endpoint with format=full.
    token = id_token.fetch_id_token(google.auth.transport.requests.Request(), AUDIENCE)
    directory = os.path.dirname(TOKEN_FILE)
    os.makedirs(directory, mode=0o700, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=directory)  # created 0600
    with os.fdopen(fd, "w") as handle:
        handle.write(token)
    os.replace(temporary, TOKEN_FILE)


def refresh_forever() -> None:
    stop = threading.Event()
    while not stop.wait(REFRESH_SECONDS):
        write_token()


write_token()  # before the workload starts
threading.Thread(target=refresh_forever, daemon=True).start()
```

On GKE, a projected service account token volume can write the file instead,
with its own refresh.

## Rules on a member's own key

On a deployment shared by several organizations, one list of trusted issuers
does not fit every tenant, and the operator should not have to write every
tenant's rules. With `federation_member_rules: true` (or
`OTARI_FEDERATION_MEMBER_RULES=true`), an organization's owners and admins give
their own keys rules, naming the issuer on the rule itself, and the deployment
operator can give one to any key in their active organization. A plain member
cannot add a rule, but can list and delete the rules on their own keys.

In the dashboard, open **API keys**, choose **Workload identity…** from the
key's actions menu, and add a rule with the issuer, the audience your workload
asks its platform for, and the service account's email or subject. The dialog
then shows the environment that points Claude Code or an Anthropic SDK at that
rule, with the rule id and the organization id filled in; set
`ANTHROPIC_IDENTITY_TOKEN_FILE` to wherever your refresher writes the token
(see [Keeping the token file fresh](#keeping-the-token-file-fresh)).

The same is available over the API, authenticated by the member's dashboard
session:

```bash
curl -X POST "https://otari.example.com/api/v1/organizations/me/keys/$KEY_ID/federation-rules" \
  --cookie "otari_dashboard_session=$SESSION" \
  -H "Content-Type: application/json" \
  -d '{
        "issuer": "https://accounts.google.com",
        "audience": "https://api.anthropic.com",
        "email": "hackbot@my-project.iam.gserviceaccount.com"
      }'
```

`GET` on the same path lists the key's rules and `DELETE .../{rule_id}`
removes one. The key must be the caller's own, as on the rest of
`/api/v1/organizations/me/keys`, and anyone else's answers 404, except for the
deployment operator. Adding a rule as a plain member answers 403.

A member's rule differs from the operator's in three ways, and each is what
keeps one tenant's rule from reaching another tenant's workload:

- **It names any https issuer.** The issuer need not be on
  `federation_issuers`. Its discovery document and signing keys are always
  fetched from a public host, whatever `federation_allow_private_hosts` says,
  because a member chose the URL.
- **It matches only in the token exchange that names it.** A workload presents
  the rule's id to `POST /api/v1/oauth/token`, as Claude Code and the Anthropic
  SDKs do, and only that rule is checked. A bare identity token is looked up
  among the operator's rules alone, so a member's rule for another tenant's
  service account never claims that tenant's tokens.
- **It conflicts only with itself.** The same rule twice on one key answers
  409. A rule on another key for the same identity does not, since each is
  named in its own exchange, and a 409 would tell a member that someone else's
  key trusts that identity.

A member's rule lets a workload act as that member's key and nothing more,
which is no more than the member could do by handing the workload the key's
secret. Turning `federation_member_rules` off stops every member's rule from
exchanging, and ends the access tokens already exchanged for one, at once; a
member can still list and delete their rules.

The response's `member_managed` field tells a member's rule from the
operator's.

## Other issuers

Any OpenID Connect issuer that publishes a discovery document and signs with
one of the algorithms above works the same way: list its `iss` value exactly
as its tokens carry it, and write rules against the `sub` (and verified
`email`, where it has one) it issues. A Kubernetes cluster's service account
issuer, for instance, puts `system:serviceaccount:<namespace>:<name>` in `sub`.

## Related documentation

- [Access control](access-control.md): API keys, service keys and budgets.
- [Configuration](configuration.md): where `federation_issuers` and `federation_member_rules` are set.
- [Use with Claude Code](use-with-claude-code.md): routing Claude Code through Otari.
