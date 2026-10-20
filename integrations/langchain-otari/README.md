# langchain-otari

Run [LangChain](https://www.langchain.com) and
[LangGraph](https://www.langchain.com/langgraph) through
[Otari](https://github.com/mozilla-ai/otari), the open-source LLM gateway, and get
its budgets, usage tracking, fallback routing and guardrails on every model call.

```bash
pip install langchain-otari
export OTARI_API_KEY=<your-otari-key>                # from otari.ai, or from your own gateway
export OTARI_BASE_URL=http://localhost:8000/api/v1   # only for a self-hosted gateway: its URL plus /api/v1
```

```python
from langchain_otari import ChatOtari

llm = ChatOtari(model="anthropic:claude-sonnet-4-6")
print(llm.invoke("Say hello in one sentence.").content)
```

- `ChatOtari` is `ChatOpenAI` pointed at Otari, so it takes every `ChatOpenAI`
  argument and works anywhere a LangChain chat model does: `create_agent`,
  `with_structured_output`, `bind_tools`, LangGraph nodes.
- `model` is any model your gateway routes: `provider:model`, or the name of a
  routing policy or alias.
- `base_url` and `api_key` override the environment. Without `OTARI_BASE_URL`,
  requests go to otari.ai's gateway, `https://api.otari.ai/api/v1`.
- `OPENAI_API_KEY` is never used as a fallback, so an OpenAI key is not sent to
  the gateway by mistake. Token usage is requested on streams too.

Full guides: [Use with LangChain](https://github.com/mozilla-ai/otari/blob/main/docs/use-with-langchain.md)
and [Use with LangGraph](https://github.com/mozilla-ai/otari/blob/main/docs/use-with-langgraph.md).

Developed in [mozilla-ai/otari](https://github.com/mozilla-ai/otari/tree/main/integrations/langchain-otari),
where every change is tested against a running Otari.
