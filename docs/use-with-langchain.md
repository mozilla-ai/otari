# Use with LangChain

`langchain-otari` gives LangChain a chat model, `ChatOtari`, that sends
requests through Otari. Your chains and agents then get Otari's budgets, usage
tracking, fallback routing and guardrails. For graphs, see
[Use with LangGraph](use-with-langgraph.md).

## Quick start

Install the package:

```bash
pip install langchain-otari
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

Then create the chat model:

```python
from langchain_otari import ChatOtari

llm = ChatOtari(model="openai:gpt-4o")

print(llm.invoke("Say hello in one sentence.").content)
```

Without `OTARI_BASE_URL`, requests go to otari.ai's gateway,
`https://api.otari.ai/api/v1`. EU accounts set `OTARI_BASE_URL` to
`https://eu.api.otari.ai/api/v1` ([Base URL](api-reference.md#base-url)).
The `base_url` and `api_key` arguments override both variables. `ChatOtari`
never reads `OPENAI_API_KEY`, so an OpenAI key in your environment is never sent
to the gateway, and it reports token usage on streams too.

`ChatOtari` takes every `ChatOpenAI` argument and works with every provider Otari
routes to, so a chain switches provider by changing only `model`. Write `model`
as Otari spells it, `provider:model` (such as
`ChatOtari(model="anthropic:claude-sonnet-4-6")`), or as a routing policy or
alias name. See [Model format](models.md#model-format) and
[Listing available models](models.md#listing-available-models).

## Agents and tools

`create_agent` runs the tool loop, and every model call in it goes through
Otari:

```python
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_otari import ChatOtari


@tool
def get_weather(city: str) -> str:
    """Return the current weather for a city."""
    return f"Sunny and 21 C in {city}"


agent = create_agent(ChatOtari(model="openai:gpt-4o"), tools=[get_weather])
result = agent.invoke({"messages": [{"role": "user", "content": "What is the weather in Paris?"}]})
print(result["messages"][-1].content)
```

## Structured output

```python
from langchain_otari import ChatOtari
from pydantic import BaseModel


class Weather(BaseModel):
    city: str
    temperature_c: int


llm = ChatOtari(model="openai:gpt-4o")

print(llm.with_structured_output(Weather).invoke("What is the weather in Paris?"))
```

## Streaming

```python
from langchain_otari import ChatOtari

llm = ChatOtari(model="openai:gpt-4o")

for chunk in llm.stream("Say hello in one sentence."):
    print(chunk.content, end="", flush=True)
print()
```

## One key per project

Give each project its own key. Its usage is then reported separately, a
[budget on the key](api-keys.md#cap-a-keys-spend) caps that project alone, and
deleting the key stops only that project.

## Without the package

`ChatOtari` is a thin layer over `ChatOpenAI`, so `langchain-openai` alone also
works:

```python
import os

from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="openai:gpt-4o",
    base_url=os.environ.get("OTARI_BASE_URL", "https://api.otari.ai/api/v1"),
    api_key=os.environ["OTARI_API_KEY"],
    stream_usage=True,
)

print(llm.invoke("Say hello in one sentence.").content)
```

## See also

- [Use with LangGraph](use-with-langgraph.md): the same model inside your own graph.
- [Use with Pydantic AI](use-with-pydantic-ai.md).
