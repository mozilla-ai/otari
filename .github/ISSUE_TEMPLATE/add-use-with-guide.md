---
name: "Docs: add a use-with guide"
about: "Write an integration guide for a coding tool or client, following the Claude Code and opencode guides"
title: "docs: add use-with guide for <Tool>"
labels: "good first issue, documentation"
---

## What

Add a guide at `docs/use-with-<tool>.md` showing how to point **&lt;Tool&gt;** at Otari, following the matching exemplar for its API surface:

- `docs/use-with-claude-code.md` for tools that speak the Anthropic Messages API
- `docs/use-with-opencode.md` for tools that speak an OpenAI-compatible API

## Why

Coding tools are the most common reason people reach for a gateway. Each one sets its base URL, auth token, and model differently, so a per-tool guide is what turns "does it work with X" into "yes, here's how."

## Steps

1. Copy the matching exemplar (`docs/use-with-claude-code.md` for Anthropic/Messages clients, or `docs/use-with-opencode.md` for OpenAI-compatible clients) to `docs/use-with-<tool>.md`.
2. Fill in each section for this tool:
   - [ ] **Intro** — which API surface the tool speaks (OpenAI `/v1/chat/completions`, Anthropic `/v1/messages`, etc.) and which Otari endpoints it hits
   - [ ] **Quick start** — cover both **Connected to otari.ai** and **Standalone**, either as separate subsections or clearly labeled examples, with the exact env vars or config file the tool uses (base URL, auth token, model)
   - [ ] **Choosing a model** — document the selector format this tool passes to Otari (for example `provider:model`, `provider/model`, or tool-prefixed forms like `otari/openai:gpt-4o`), with an example or two the tool supports
   - [ ] **Gotchas** — anything tool-specific (header names, base URL with or without `/v1`, user-id handling), like the `ANTHROPIC_AUTH_TOKEN` vs `ANTHROPIC_API_KEY` note in the Claude Code guide
   - [ ] **See also** — link back to the Quickstart and the provider setup
3. Don't re-explain provider setup — link to it. This guide assumes a provider is already configured.
4. Test it end to end against a running Otari.
5. Link the new page from the docs navigation (`docs/index.md`) and any nearby cross-references that should mention it.

## Acceptance criteria

- [ ] New file at `docs/use-with-<tool>.md`
- [ ] Follows the matching exemplar for the tool's API surface
- [ ] Both "Connected to otari.ai" and "Standalone" variants covered, either as separate subsections or clearly labeled examples
- [ ] The tool's selector format for Otari is explained with concrete examples
- [ ] At least one tool-specific gotcha called out
- [ ] Linked from `docs/index.md`

## Good picks to start

Cursor, Continue, Aider, Zed, Cline, Windsurf, Codex CLI, Gemini CLI — all support a custom OpenAI- or Anthropic-compatible base URL.

## Resources

- Exemplars: `docs/use-with-claude-code.md`, `docs/use-with-opencode.md`
- Model format and providers: `docs/models.md`
- Base config: `docs/quickstart.md`
