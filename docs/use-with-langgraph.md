# Use with LangGraph

LangGraph calls models through LangChain's chat models, so `ChatOtari` from
`langchain-otari` is all it takes (see [Use with LangChain](use-with-langchain.md)):
every model call in every node of your graph then gets Otari's budgets, usage
tracking, fallback routing and guardrails.

## Quick start

Install LangGraph and the package:

```bash
pip install langgraph langchain-otari
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

## A tool-calling graph

A model node and a tool node, looping until the model stops calling tools:

```python
from langchain_core.tools import tool
from langchain_otari import ChatOtari
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition


@tool
def get_weather(city: str) -> str:
    """Return the current weather for a city."""
    return f"Sunny and 21 C in {city}"


llm = ChatOtari(model="openai:gpt-4o").bind_tools([get_weather])


def call_model(state: MessagesState) -> dict:
    return {"messages": [llm.invoke(state["messages"])]}


graph = StateGraph(MessagesState)
graph.add_node("model", call_model)
graph.add_node("tools", ToolNode([get_weather]))
graph.add_edge(START, "model")
graph.add_conditional_edges("model", tools_condition)
graph.add_edge("tools", "model")
app = graph.compile()

result = app.invoke({"messages": [{"role": "user", "content": "What is the weather in Paris?"}]})
print(result["messages"][-1].content)
```

## Streaming tokens

`stream_mode="messages"` streams the model's tokens out of the graph as Otari
streams them in:

```python
from langchain_otari import ChatOtari
from langgraph.graph import START, MessagesState, StateGraph

llm = ChatOtari(model="openai:gpt-4o")


def call_model(state: MessagesState) -> dict:
    return {"messages": [llm.invoke(state["messages"])]}


graph = StateGraph(MessagesState)
graph.add_node("model", call_model)
graph.add_edge(START, "model")
app = graph.compile()

for chunk, _metadata in app.stream(
    {"messages": [{"role": "user", "content": "Say hello in one sentence."}]},
    stream_mode="messages",
):
    print(chunk.content, end="", flush=True)
print()
```

## Different models per node

Each node can use its own model through the same gateway, for example a cheap
model to route and a stronger one to answer. Otari tracks the spend of both under
the same key, and a routing policy name works anywhere a model name does; see
[Choosing a model](use-with-pydantic-ai.md#choosing-a-model).

## See also

- [Use with LangChain](use-with-langchain.md): agents, structured output and streaming with `ChatOtari`.
- [Use with Pydantic AI](use-with-pydantic-ai.md).
