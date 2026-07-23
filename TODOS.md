# TODOS

Deferred work, captured so it isn't lost.

## From #378 (external usage ingestion + exclude-from-budget)

- [ ] **Per-model budget exclusion.** Extend the `exclude_from_budget` concept from API keys to
      specific models (exclude a model from budget enforcement everywhere). Needs a model-level
      flag, resolution in the budget path, and a Models-page UI toggle. Deferred from the #378 PR
      (which ships the per-key toggle only).
- [ ] **Per-request budget exclusion.** Allow a single request to opt out of budget enforcement
      (e.g. a header or body flag), for one-off internal/testing calls.
- [ ] **Subscription-vs-API break-even / comparison view.** A dashboard surface that compares a
      user's subscription cost against their imported API-equivalent cost ("you'd save/spend $Z by
      switching these models"). The strategic 10x idea; ingestion (#378) is the foundation for it.
- [ ] **More vendor special-cases.** OTLP ingestion handles the GenAI semantic conventions
      generically (`gen_ai.*` + `otari.*`) plus Claude Code and Codex special-cases. Add
      special-cases for other agents that emit non-standard attributes (e.g. Cursor, Copilot) if
      they don't adopt the `gen_ai.*` conventions.
- [ ] **OTLP gRPC receiver (evaluate).** The OTLP endpoints are HTTP (`/v1/traces`, `/v1/logs`,
      protobuf or JSON). gRPC OTLP is not accepted; exporters must use an `http/*` protocol.
      Revisit if a gRPC-only producer needs it.
