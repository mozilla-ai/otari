---
description: Run your own guardrail service on an open-weight safety model, so a guardrail needs no third-party vendor key.
---

# Host your own guardrail

Every guardrail in the [catalog](https://docs.otari.ai/reference/gateway/guardrails-reference#the-guardrail-catalog) needs a vendor account and key. If you have none, you can run a small guardrail service of your own instead. It receives the text Otari is about to send to a model, asks an open-weight safety model such as Llama Guard 4 or gpt-oss-safeguard whether the text is safe, and answers Otari's [guardrails service contract](https://docs.otari.ai/reference/gateway/guardrails-reference#what-a-guardrails-service-answers).

The service asks the safety model through Otari, with your own API key. The check is therefore an ordinary model call: it appears in your usage, counts against your budgets and is billed at the model's rate, like any other request. Otari itself does not run a model to judge text.

This guide assumes you have read [Configure guardrails](configure-guardrails.md) and have an Otari API key. The model ids below are examples: use the ids your deployment lists in `GET /api/v1/models`.

## Write the service

The service below is a complete example in Python. It maps a safety model's answer onto the `valid`, `explanation` and `score` fields the contract defines.

```python
import asyncio
import hmac
import os

from fastapi import FastAPI, Header, HTTPException
from openai import AsyncOpenAI, OpenAIError
from pydantic import BaseModel

OTARI_BASE_URL = os.environ.get("OTARI_BASE_URL", "https://api.otari.ai/api/v1")
GUARD_MODEL = os.environ["GUARD_MODEL"]  # for example meta/llama-guard-4-12b
GUARD_POLICY = os.environ.get("GUARD_POLICY")  # a system prompt, for models that need one
GUARD_TOKEN = os.environ.get("GUARD_TOKEN")  # the credential the gateway sends
if not GUARD_TOKEN and os.environ.get("GUARD_NO_AUTH") != "1":
    raise SystemExit("set GUARD_TOKEN, or GUARD_NO_AUTH=1 for a service only the gateway can reach")

# The whole model call ends under the gateway's 30 second limit, so it fails
# here, with a message you can read, before the gateway gives up on this service.
DEADLINE_S = 20
client = AsyncOpenAI(base_url=OTARI_BASE_URL, api_key=os.environ["OTARI_API_KEY"], max_retries=0)
app = FastAPI()


class ValidateRequest(BaseModel):
    profile: str
    input_text: str
    validate_kwargs: dict | None = None


def read_verdict(answer: str) -> dict:
    """Map a Llama Guard style answer onto Otari's verdict.

    Assumes the first line is `safe` or `unsafe` and that an unsafe answer
    lists category codes on the next line. Anything else is no verdict.
    """
    lines = [line.strip() for line in answer.strip().splitlines() if line.strip()]
    first = lines[0].lower() if lines else ""
    if first == "safe":
        return {"valid": True, "explanation": None, "score": None}
    if first == "unsafe":
        return {"valid": False, "explanation": ", ".join(lines[1:]) or None, "score": None}
    return {"valid": None, "explanation": "unreadable guard answer", "score": None}


@app.post("/validate")
async def validate(body: ValidateRequest, authorization: str | None = Header(default=None)):
    if GUARD_TOKEN and not hmac.compare_digest(authorization or "", f"Bearer {GUARD_TOKEN}"):
        raise HTTPException(status_code=401)
    messages = [{"role": "user", "content": body.input_text}]
    if GUARD_POLICY:
        messages.insert(0, {"role": "system", "content": GUARD_POLICY})
    try:
        async with asyncio.timeout(DEADLINE_S):
            completion = await client.chat.completions.create(model=GUARD_MODEL, messages=messages, temperature=0)
    except (OpenAIError, TimeoutError) as exc:
        raise HTTPException(status_code=502, detail=type(exc).__name__) from exc
    answer = completion.choices[0].message.content or ""
    return {"profile": body.profile, "result": read_verdict(answer)}
```

What the example assumes, and how it fails:

- **The answer format.** `read_verdict` reads the format Llama Guard models use: `safe`, or `unsafe` followed by category codes. It does not check what a given model returns. For a model that needs a policy, such as gpt-oss-safeguard, set `GUARD_POLICY` to a system prompt that tells the model to answer in that same format, or change `read_verdict` to match the model's own.
- **It fails closed.** An answer `read_verdict` cannot read comes back as `valid: null`, never `valid: true`. A model call that fails or runs past `DEADLINE_S` answers `502`. The service does not start without `GUARD_TOKEN`, unless `GUARD_NO_AUTH=1` says only the gateway can reach it. In both cases a `block` guardrail refuses the request by default, as [the unreachable section](https://docs.otari.ai/reference/gateway/guardrails-reference#when-the-guardrails-service-is-unreachable) describes.
- **`score` is `null`.** A safety model that answers in words gives no score.
- **`profile` and `validate_kwargs` are ignored.** One service can serve several profiles by choosing the model or policy from `profile`.

Install and run it with a key from a workspace that has no guardrail mandated (see the note on loops below):

```bash
pip install fastapi uvicorn openai
export OTARI_API_KEY=<your-otari-key>
export GUARD_MODEL=meta/llama-guard-4-12b
export GUARD_TOKEN=$(openssl rand -hex 32)
uvicorn guard_service:app --port 9101
```

Check it directly:

```bash
curl http://localhost:9101/validate \
  -H "Authorization: Bearer $GUARD_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"profile": "safety", "input_text": "How do I bake bread?"}'
# {"profile":"safety","result":{"valid":true,"explanation":null,"score":null}}
```

A flagged input answers `{"profile":"safety","result":{"valid":false,"explanation":"S9","score":null}}`, where `S9` is the category line the model gave.

## Point a request at it

Name the service in the guardrail entry's `url`. Give the base address only: Otari adds `/validate` itself.

```bash
curl https://api.otari.ai/api/v1/chat/completions \
  -H "Authorization: Bearer $OTARI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "anthropic:claude-sonnet-4-6",
    "messages": [{"role": "user", "content": "Hello"}],
    "guardrails": [
      {
        "profile": "safety",
        "mode": "block",
        "url": "https://guard.example.com",
        "credential": "<the value of GUARD_TOKEN>"
      }
    ]
  }'
```

`profile` is a name you choose; it is sent to your service as given. A `url` in a request must be on a public address, with no redirect followed, so a service on `localhost` or a private network cannot be reached this way. `credential` is sent to the service as `Authorization: Bearer`, which is what `GUARD_TOKEN` checks, and it requires an `https` address. [The `guardrails` request field](https://docs.otari.ai/reference/gateway/guardrails-reference#the-guardrails-request-field) lists every rule.

To check every request of an organization without changing client code, mandate the same `url` and `credential` on the **Guardrails** page (choose your own guardrail service) or over the API ([Mandate fields](https://docs.otari.ai/reference/gateway/guardrails-reference#mandate-fields)). The credential is stored encrypted and never returned.

Otari reads the result as it does any other guardrail's: a flagged input in `block` mode is refused with `403` and `guardrail_violation`, and the verdict of a served request is on the `Otari-Guardrails` header ([Reading the header](https://docs.otari.ai/reference/gateway/guardrails-reference#reading-the-otari-guardrails-header)).

## Point a deployment you run at it

On a deployment you run, the operator can set one service for every profile:

```bash
export OTARI_GUARDRAILS_URL=http://localhost:9101
```

This address may be private and plain `http`, so a service beside the gateway works. Otari sends no credential to it, which means the service has to be reachable only from the gateway (start it with `GUARD_NO_AUTH=1` instead of `GUARD_TOKEN`, and keep the port off the public network). A request or a mandate that sets its own `url` replaces it for that entry.

## Keep each check inside the time limit

Otari waits up to 30 seconds for a service to answer ([How long a service has to answer](https://docs.otari.ai/reference/gateway/guardrails-reference#how-long-a-service-has-to-answer)), and the guardrail runs before the model is called, so every second of it is added to your request. After that, the check counts as unavailable. With the default `on_unavailable: block` the request is refused with `502`; with `on_unavailable: monitor` it is served unchecked.

An open-weight model that has not been called for a while can take most of that: Llama Guard 4 has been reported to answer in 20 to 30 seconds on a cold call and in about a second once warm. To stay inside the limit:

- Keep the service's own deadline below the gateway's, as the example does, so a slow model call ends with a clear error.
- Warm the model before you rely on it, for example with a request on a schedule.
- Decide what a slow check should mean for your traffic. `on_unavailable: block` favors safety and refuses; `monitor` favors availability and serves the request with `valid: null` and `status: unavailable` on the header.
- For a stricter bound, have the service ask two models at once and answer with the first usable verdict, or return its own fallback verdict after a short deadline.

A verdict is kept for about an hour per gateway worker for the same text ([What a guardrail reads](https://docs.otari.ai/reference/gateway/guardrails-reference#what-a-guardrail-reads)), and a check that gave none is asked again, so a request that timed out does not stay refused.

## What a check costs

Each check is one chat completion on the key the service uses, billed at that model's price on that key's workspace and user. Two things follow:

- **Use a key that is not guarded by this service.** A guardrail mandated for a workspace runs on every chat completion from it, including the service's own call. A key from a workspace that mandates the service would send each check back to the service. Give the service a key from a workspace without that mandate, with a [budget](set-a-budget-and-ceiling.md) that limits what checks can spend.
- **The check is metered when it runs, not when it flags.** A safe input costs a call as much as an unsafe one.
