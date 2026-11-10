# Agent memory

Agent memory gives every agent in a workspace one memory it shares, whatever
model or harness it runs on. An agent searches it before work someone may have
done before, and proposes what it learns. A consolidator decides what is kept,
so the memory improves with use without any one agent overwriting it.

Memory is off by default, for the deployment and for every workspace. Nothing an
agent writes is readable at rest: every entry, proposal and embedding is sealed
under a data key of its workspace.

## Turn it on

1. Turn it on for the deployment and name the embedding model, then restart:

   ```yaml
   memory_enabled: true
   memory_embedding_model: openai:text-embedding-3-small
   # Optional: a model that decides what a proposal does to memory.
   memory_consolidation_model: openai:gpt-4.1-mini
   ```

   Memory seals with the deployment's content key, as trace content does: set
   `OTARI_SECRET_KEY`, or `trace_content_key_backend` and its KMS key. Without a
   key, memory stores nothing.

2. Turn it on for the organization. Its owners and admins set one switch for
   every workspace of it; with it off, memory runs in none of them, whatever
   their own switches say, and nothing is deleted:

   ```bash
   curl -X PUT "$OTARI_URL/api/v1/organizations/me/memory" \
     -H "Authorization: Bearer $OTARI_MASTER_KEY" -H "Content-Type: application/json" \
     -d '{"enabled": true}'
   ```

3. Turn it on for a workspace. An organization owner or admin, or an owner or
   admin of the workspace, sets its switches:

   ```bash
   curl -X PUT "$OTARI_URL/api/v1/workspaces/$WORKSPACE_ID/memory" \
     -H "Authorization: Bearer $OTARI_MASTER_KEY" -H "Content-Type: application/json" \
     -d '{"enabled": true, "workspace_scope_enabled": true, "write_policy": "auto"}'
   ```

   | Switch | What it does |
   | --- | --- |
   | `enabled` | Memory on: each member's agents keep and recall entries of their own. |
   | `workspace_scope_enabled` | The memory every member's agents share. |
   | `write_policy` | `auto` keeps what the consolidator decides; `review`, the default, holds every shared proposal for an admin. |
   | `offer_to_every_request` | Offer memory's tools to every request a member sends through Otari, not only to one that asks. Off by default. See [Through Otari's tool loop](#through-otaris-tool-loop). |
   | `hide_external` | Leave entries resting on a web page, a fetched file or a third-party tool out of recall. |
   | `retention_days` | Days an unused entry is kept. Unset keeps the deployment's `memory_retention_days`. |

   `GET` on the same path shows the switches and how many entries the workspace
   holds. Turning memory off stops recall at once and deletes nothing; `DELETE`
   on the same path erases everything, data keys included.

## Connect an agent

An agent reaches memory in one of two ways, and both reach the same tools with
the same checks.

### Through Otari's tool loop

A request routed through Otari, on any model, gets memory's tools beside its
other tools, and Otari runs each call, as it runs web search. A request asks for
them in either of two ways:

- declare the built-in tool in `tools`:

  ```json
  {"model": "openai:gpt-4.1", "messages": [...], "tools": [{"type": "otari_memory"}]}
  ```

- or send the header `Otari-Memory: on`.

Where the workspace has `offer_to_every_request` on, every request a member sends
gets them without asking, and one that sends `Otari-Memory: off` does not. A
request that neither asks nor is offered them reaches the provider unchanged.

The request's API key must name a member of its workspace. Asking for memory
where it cannot be had (memory off for the workspace or its organization, or a
key with no member behind it) is refused before the provider is called; not
asking never is. No provider is sent the `otari_memory` declaration, and an MCP
server cannot expose a tool under one of memory's names.

### As an MCP server

Memory is also an MCP server at `/api/v1/memory/mcp`, reached with the API key of a
member of the workspace. The key decides the workspace and the member; no tool
argument can name either. A key that names no member of its workspace, such as
a service key, is refused.

For Claude Code:

```bash
claude mcp add --transport http otari-memory "$OTARI_URL/api/v1/memory/mcp" \
  --header "Authorization: Bearer $OTARI_API_KEY"
```

Any MCP client that speaks Streamable HTTP connects the same way. The server is
stateless and answers each request in its response, so it offers no stream to
open with `GET`.

## The tools

| Tool | What it does |
| --- | --- |
| `memory_search` | The entries that best match a query, each with its kind, scope, trust level, confidence and age. |
| `memory_get` | One entry, with the versions it replaced. |
| `memory_propose` | Offers one fact, decision or procedure. It is a proposal until the consolidator decides on it. |
| `memory_confirm` | Says an entry held up in use, which raises its confidence. |
| `memory_flag` | Says an entry is wrong, stale, a duplicate or unsafe, which lowers its confidence. |
| `memory_forget` | Deletes an entry and every version it replaced. Its author, its owner or a workspace manager may. |

A workspace viewer can search and read, and changes nothing. An entry in the
member scope is its owner's alone: no other member, and no admin, recalls it.

## How a proposal becomes memory

1. **Screening.** A proposal carrying a credential, a private key, an email
   address or a card number is refused before anything is stored, and the agent
   is told which kind it carried, never the match.
2. **Comparison.** The consolidator embeds the proposal and compares it with the
   nearest entries of its scope.
3. **Decision.** The consolidation model answers with one decision and nothing
   else: add, update (a new version of an entry), no change, or conflict. What
   is stored is the proposal's own text, so an instruction hidden in a proposal
   has nothing to write. Without a consolidation model, a proposal nearly the
   same as an entry is a no change, and anything else is added.
4. **Review.** A conflict, a shared proposal under the `review` policy, an
   update an agent proposed to an entry an admin reviewed, and a proposal whose
   model call failed five times wait for an admin instead.

Search ranks entries by their similarity to the query times their confidence.
Confidence starts at 0.5, rises by 0.1 for each member who confirms an entry
and falls by 0.15 for each who flags it, the rules of
[cq](https://github.com/mozilla-ai/cq). A member counts once per entry, and an
entry three members flag leaves recall until an admin looks at it.

## Data protection

- **Sealed at rest.** Text and embeddings are sealed with AES-256-GCM under a
  data key of the workspace, made and opened by the deployment's content key.
  Each value is bound to its workspace, scope, owner and row, so a value copied
  anywhere else opens as nothing.
- **Fenced by workspace.** On PostgreSQL the memory tables are under forced
  row-level security keyed on the workspace, beside each query's own workspace
  filter. SQLite has no row-level security; there the filter and the sealing are
  the bound.
- **Retention.** An entry or version nobody has used for the retention period is
  deleted, a retired entry at once, a proposal after
  `memory_proposal_max_age_hours`, and a read record after
  `memory_read_retention_days`. A data key that seals nothing any more is
  deleted too, so what was deleted opens from no backup.
- **Deletion.** Deleting a member deletes their own entries and clears them as
  author of the shared ones. Deleting a workspace deletes its memory.
- **Usage, no content.** Each embedding and consolidation call is recorded as
  usage of its workspace, outside any budget. Logs, traces and metrics carry ids
  and counts, never what an agent wrote.

## Settings

| Setting | Default | Purpose |
| --- | --- | --- |
| `memory_enabled` | `false` | Offer memory. Startup setting. |
| `memory_embedding_model` | unset | The one model every entry and search is embedded with. |
| `memory_consolidation_model` | unset | The model that decides what a proposal does to memory. |
| `memory_max_entries_per_workspace` | `5000` | Active entries one workspace holds; past it, the least used is retired. |
| `memory_entry_max_length` | `2000` | Longest proposal, in characters. |
| `memory_retention_days` | `180` | Days an unused entry or version is kept. |
| `memory_proposal_max_age_hours` | `24` | Hours a proposal is kept, decided or not. |
| `memory_read_retention_days` | `365` | Days a read record is kept. |
| `memory_key_rotation_days` | `30` | Days one workspace data key seals new entries. |
