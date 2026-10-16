# pydantic-ai-otari

Run [Pydantic AI](https://ai.pydantic.dev) agents through
[Otari](https://github.com/mozilla-ai/otari), the open-source LLM gateway, and get
its budgets, usage tracking, fallback routing and guardrails on every model call.

```bash
pip install pydantic-ai-otari
export OTARI_API_KEY=<your-otari-key>                # from otari.ai, or from your own gateway
export OTARI_BASE_URL=http://localhost:8000/api/v1   # only for a self-hosted gateway: its URL plus /api/v1
```

```python
from pydantic_ai import Agent
from pydantic_ai_otari import OtariModel

agent = Agent(OtariModel("anthropic:claude-sonnet-4-6"))
print(agent.run_sync("Say hello in one sentence.").output)
```

- `OtariModel(model_name)` takes any model your gateway routes: `provider:model`,
  or the name of a routing policy or alias. The provider part also selects the
  model's capabilities in Pydantic AI, so a Claude model is treated as one.
- `OtariProvider(base_url=..., api_key=...)` configures the gateway in code;
  pass it as `OtariModel(..., provider=...)`. Without `OTARI_BASE_URL`, requests
  go to otari.ai's gateway, `https://api.otari.ai/api/v1`.
- `OPENAI_API_KEY` is never used as a fallback, so an OpenAI key is not sent to
  the gateway by mistake.

Tools, structured output and streaming work unchanged. The full guide, including
how to do this without the package:
[Use with Pydantic AI](https://github.com/mozilla-ai/otari/blob/main/docs/use-with-pydantic-ai.md).

Developed in [mozilla-ai/otari](https://github.com/mozilla-ai/otari/tree/main/integrations/pydantic-ai-otari),
where every change is tested against a running Otari.
