# Use with Pydantic AI

[Pydantic AI](https://ai.pydantic.dev) agents run through Otari with
`pydantic-ai-otari`, a model class that points Pydantic AI at Otari's
OpenAI-compatible API (`POST /api/v1/chat/completions`). Every agent run then gets
Otari's budgets, usage tracking, fallback routing and guardrails, with no other
change to your agent code.

## Quick start

Install the package:

```bash
pip install pydantic-ai-otari
```

Create an API key at [otari.ai](https://otari.ai) and export it:

```bash
export OTARI_API_KEY=<your-otari-key>
```

That is all the configuration otari.ai's gateway needs, once your organization
there has credit or provider keys of its own; without either, it answers 402. For
a gateway you run yourself, use a key it issued and also export its API root:

```bash
export OTARI_BASE_URL=http://localhost:8000/api/v1   # your gateway's URL plus /api/v1
```

Then give your agent an `OtariModel`:

```python
from pydantic_ai import Agent
from pydantic_ai_otari import OtariModel

agent = Agent(OtariModel("openai:gpt-4o"))

result = agent.run_sync("Say hello in one sentence.")
print(result.output)
```

Without `OTARI_BASE_URL`, requests go to otari.ai's gateway,
`https://api.otari.ai/api/v1`. To configure the gateway in code instead, pass a
provider: `OtariModel("openai:gpt-4o", provider=OtariProvider(base_url=..., api_key=...))`.
`OtariProvider` never falls back to `OPENAI_API_KEY`, so an OpenAI key in your
environment is not sent to the gateway by mistake.

`OtariModel` works for every provider Otari routes to: the request is always
Chat Completions, and Otari translates it for Anthropic, Mistral, Gemini and the
rest, so the same agent switches provider by changing only the model name. The
provider part of the name also tells Pydantic AI what the model can do, so
`anthropic:claude-sonnet-4-6` gets Claude's capabilities rather than OpenAI's.

## Tools

Tools run in your process as usual. Otari forwards the tool definitions to the
model and the tool results back, and records every model call of the run:

```python
from pydantic_ai import Agent
from pydantic_ai_otari import OtariModel

agent = Agent(OtariModel("openai:gpt-4o"))


@agent.tool_plain
def get_weather(city: str) -> str:
    """Return the current weather for a city."""
    return f"Sunny and 21 C in {city}"


result = agent.run_sync("What is the weather in Paris?")
print(result.output)
```

## Structured output

`output_type` works unchanged; Pydantic AI validates the model's answer into your
type:

```python
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai_otari import OtariModel


class Weather(BaseModel):
    city: str
    temperature_c: int


agent = Agent(OtariModel("openai:gpt-4o"), output_type=Weather)

result = agent.run_sync("What is the weather in Paris?")
print(result.output)
```

## Streaming

```python
import asyncio

from pydantic_ai import Agent
from pydantic_ai_otari import OtariModel

agent = Agent(OtariModel("openai:gpt-4o"))


async def main() -> None:
    async with agent.run_stream("Say hello in one sentence.") as result:
        async for text in result.stream_text(delta=True):
            print(text, end="", flush=True)
    print()


asyncio.run(main())
```

## Choosing a model

The model name is whatever your gateway routes, as `provider:model`. The same
agent code works across providers: every example on this page runs unchanged on
`openai:gpt-4o`, `anthropic:claude-sonnet-4-6` and `mistral:mistral-large-latest`.

- **On otari.ai:** any model your organization's catalog lists, which
  `GET /api/v1/models` returns for your key.
- **On your own gateway:** `provider:model` for any provider it is configured
  with, for example `openai:gpt-4o` or `mistral:mistral-large-latest`.
- **A routing policy or alias:** its name, for example `OtariModel("support-agent")`.
  Otari picks the candidate and falls over to the next on failure, so the agent
  code does not change when the policy does. See [Routing](routing.md).

## One key per project

Give each project (or each client) its own Otari key. Its usage and spend are
then reported apart, a budget ceiling on the key caps that project alone, and
revoking the key stops only that project. See [Access control](access-control.md)
and [Budgets and API keys in config](configuration.md#budgets-and-api-keys-in-config).

## Without the package

`OtariModel` is a thin layer over Pydantic AI's own OpenAI model, so the same
thing works with `pydantic-ai-slim[openai]` alone:

```python
import os

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

otari = OpenAIProvider(
    base_url=os.environ.get("OTARI_BASE_URL", "https://api.otari.ai/api/v1"),
    api_key=os.environ["OTARI_API_KEY"],
)
agent = Agent(OpenAIChatModel("openai:gpt-4o", provider=otari))

print(agent.run_sync("Say hello in one sentence.").output)
```

What you give up is the model capabilities that follow the model family: every
model is described to Pydantic AI as an OpenAI one.

## See also

- [Use with LangChain](use-with-langchain.md) and [Use with LangGraph](use-with-langgraph.md):
  the same gateway from LangChain's chat models.
- [Agent traces](traces.md): what Otari records about an agent's runs.
