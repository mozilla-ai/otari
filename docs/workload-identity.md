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

Workload identity is available in standalone and hosted modes. A hybrid
gateway verifies no keys of its own and refuses the setting at startup.

## How a token is checked

When `federation_issuers` is empty, the default, none of this runs and every
credential is checked as an API key. Once it lists an issuer, a credential
shaped like a signed JWT is checked as an identity token:

1. Its issuer (`iss`) must be one of `federation_issuers`, compared exactly.
2. Its signature must verify against the issuer's published signing keys
   (JWKS), with an asymmetric algorithm: RS256, RS384, RS512, PS256, PS384,
   PS512, ES256, ES384 or ES512. `none` and the HMAC algorithms are refused,
   a key is used only with the algorithm it is published for, and an RSA key
   shorter than 2048 bits is refused.
3. It must carry `iss`, `sub`, `aud`, `exp` and `iat`. It must not be expired,
   and neither `iat` nor `nbf` may be in the future, each with 60 seconds of
   allowed clock skew.
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
loopback or link-local address is refused unless
`OTARI_FEDERATION_ALLOW_PRIVATE_HOSTS=true`, for an issuer on your own network.

Keys are cached for an hour per process. A token naming a key id the cache
lacks refetches at most once a minute, which picks up a rotation without
letting made-up key ids turn each request into a fetch. A failed fetch is
retried after ten seconds; until one succeeds, keys still within their hour
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
`/api/v1/keys`.

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


reply = otari().chat.completions.create(
    model="openai:gpt-4o", messages=[{"role": "user", "content": "Hello"}]
)
```

## Other issuers

Any OpenID Connect issuer that publishes a discovery document and signs with
one of the algorithms above works the same way: list its `iss` value exactly
as its tokens carry it, and write rules against the `sub` (and verified
`email`, where it has one) it issues. A Kubernetes cluster's service account
issuer, for instance, puts `system:serviceaccount:<namespace>:<name>` in `sub`.

## Related documentation

- [Access control](access-control.md): API keys, service keys and budgets.
- [Configuration](configuration.md): where `federation_issuers` is set.
