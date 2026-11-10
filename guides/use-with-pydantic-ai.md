# Use with Pydantic AI

`pydantic-ai-otari` gives [Pydantic AI](https://ai.pydantic.dev) a model class
that sends requests through Otari. Your agents then get Otari's budgets, usage
tracking, fallback routing and guardrails, with no other code change.

## Quick start

Install the package:

```bash
pip install pydantic-ai-otari
```

Create a key on the **API keys** page of the [otari.ai](https://otari.ai)
dashboard and export it:

```bash
export OTARI_API_KEY=<your-otari-key>
```

On otari.ai, the key is all you need, as long as your organization has credit or
its own provider keys (otherwise requests answer `402`). For a gateway you run,
use a key it issued and also export its API root:

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
`https://api.otari.ai/api/v1`. EU accounts set `OTARI_BASE_URL` to
`https://eu.api.otari.ai/api/v1` ([Base URL](https://docs.otari.ai/reference#base-url)).
To set the gateway in code, pass a provider:
`OtariModel("openai:gpt-4o", provider=OtariProvider(base_url=..., api_key=...))`.
`OtariProvider` never reads `OPENAI_API_KEY`, so an OpenAI key in your
environment is never sent to the gateway.

`OtariModel` works with every provider Otari routes to, so an agent switches
provider by changing only the model name. The provider part of the name also
tells Pydantic AI what the model can do: `anthropic:claude-sonnet-4-6` gets
Claude's capabilities.

## Tools

Tools run in your process as usual, and Otari records every model call of the
run:

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

Write the model as Otari spells it, `provider:model`, such as
`OtariModel("anthropic:claude-sonnet-4-6")`. Every example on this page also runs
on `anthropic:claude-sonnet-4-6` and `mistral:mistral-large-latest`.

A routing policy or alias name works too, such as
`OtariModel("support-agent")`, so your code does not change when the policy
does. An alias names one model; a policy with fallbacks moves to the next model
on a retryable failure. See
[Model format](https://docs.otari.ai/reference/gateway/models#model-format),
[Listing available models](https://docs.otari.ai/reference/gateway/models#listing-available-models) and
[Routing](https://docs.otari.ai/reference/gateway/routing).

## One key per project

Give each project its own key. Its usage is then reported separately, a
[budget on the key](api-keys.md#cap-a-keys-spend) caps that project alone, and
deleting the key stops only that project.

## Without the package

`OtariModel` is a thin layer over Pydantic AI's OpenAI model, so
`pydantic-ai-slim[openai]` alone also works:

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

Without the package, Pydantic AI treats every model as an OpenAI one, so
models from other families lose their own capabilities.

## See also

- [Use with LangChain](use-with-langchain.md) and [Use with LangGraph](use-with-langgraph.md)
- [Sessions](sessions.md): what Otari records about an agent's runs.
