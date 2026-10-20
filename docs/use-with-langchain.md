# Use with LangChain

LangChain chains and agents run through Otari with `langchain-otari`, whose
`ChatOtari` is LangChain's `ChatOpenAI` pointed at Otari's OpenAI-compatible API
(`POST /api/v1/chat/completions`). Every call then gets Otari's budgets, usage
tracking, fallback routing and guardrails. For graphs, see
[Use with LangGraph](use-with-langgraph.md).

## Quick start

Install the package:

```bash
pip install langchain-otari
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

Then create the chat model:

```python
from langchain_otari import ChatOtari

llm = ChatOtari(model="openai:gpt-4o")

print(llm.invoke("Say hello in one sentence.").content)
```

Without `OTARI_BASE_URL`, requests go to otari.ai's gateway,
`https://api.otari.ai/api/v1`. `base_url` and `api_key` arguments override both
variables. `ChatOtari` never falls back to `OPENAI_API_KEY`, so an OpenAI key in
your environment is not sent to the gateway by mistake, and it asks Otari for
token usage on streams too.

`ChatOtari` takes every `ChatOpenAI` argument and works for every provider Otari
routes to: Otari translates the Chat Completions request for Anthropic, Mistral,
Gemini and the rest, so the same chain switches provider by changing only
`model`. The names it accepts are listed under
[Choosing a model](use-with-pydantic-ai.md#choosing-a-model).

## Agents and tools

`create_agent` runs the tool loop; every model call in it goes through Otari:

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

Give each project (or each client) its own Otari key. Its usage and spend are
then reported apart, a budget ceiling on the key caps that project alone, and
revoking the key stops only that project. See [Access control](access-control.md)
and [Budgets and API keys in config](configuration.md#budgets-and-api-keys-in-config).

## Without the package

`ChatOtari` is a thin layer over `ChatOpenAI`, so the same thing works with
`langchain-openai` alone:

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
