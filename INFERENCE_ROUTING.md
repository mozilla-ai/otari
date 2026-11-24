# Inference routing design

## Repository assessment

Otari owns selection, credentials, model access, fallback, pricing and usage
attribution. The any-llm SDK owns provider protocol translation and calls. This
change belongs in Otari under the existing routing domain. A new provider
protocol or a fix to an SDK adapter belongs in any-llm.
[Contribution guide](CONTRIBUTING.md#is-this-an-otari-change-or-an-any-llm-change)

| Area | Existing behavior and extension point |
| --- | --- |
| Logical model IDs | Callers already send aliases, named policies, bare catalog IDs or physical selectors in the existing `model` field. Policies support a default, conditional selection, priority and weighted ranking, ordered fallback, and guardrails. Stored scopes are workspace and optionally user. |
| Inference execution | Chat Completions, Anthropic Messages and Responses share the completion pipeline, with format adapters for dispatch and response handling. The walker handles retryable upstream failures, provider errors before stream commitment, rate-limit spillover, and lock-in once a tool loop is underway. |
| Other operations | Embeddings, image generation, multipart image editing, speech and transcription have their own any-llm calls and use the pass-through scaffold. Static policy names can resolve there as aliases. Dynamic selection belongs to the completion pipeline. Batch submission and asynchronous video jobs have distinct lifecycle and provider state contracts. |
| Provider connections | Deployment configuration supports named instances, custom endpoint URLs and keyless local endpoints on supported any-llm adapters. Hosted providers and organization credentials have separate access and dispatch paths. Tenant-owned provider endpoints are bound to a workspace or user, with public-address checks, and are outside the policy compiler. |
| Capability evidence | Catalog and model discovery describe offered models. The completion pipeline has an aggregate wrong-model-kind gate. Native image and PDF normalization uses model overrides, known exceptions, then provider flags where the flags are trusted. Those facts did not filter individual attempts in named policies. |
| Locality and availability | Existing country restrictions refer to caller reachability. Hosted provider regions carry data-location declarations. The provider health monitor reports cached model discovery, which can be unsupported despite inference being available. Automatic catalog routing uses worker-local observations for cooldown, load and latency alongside region, ownership and estimated price. Those observations are specific to interchangeable catalog offerings, and a cooldown is a ranking penalty there. |
| Authority and accounting | The completion pipeline derives workspace, organization, user and key access from authentication. The compiler checks allowed models, organization provider-key restrictions and country rules. Before dispatch the pipeline reserves budget, and the walker handles attempt-specific admission and pricing. Logs, usage rows and response metadata distinguish served, skipped and failed attempts. |
| Configuration and API | `PolicySpec` is validated for YAML and for writes through the deployment and organization policy APIs. Stored specs are versioned JSON. Explain already reports ordered and dropped candidates. The public policy API carries the spec as a JSON object. |

[Routing domain](DOMAINS.md#routing) · [Request lifecycle](src/gateway/AGENTS.md#request-lifecycle) · [Provider endpoints](docs/provider-endpoints.md) · [Routing tests](tests/unit/test_routing_compiler.py)

Agent model recommendations answer a separate decision request. The harness
uses the recommendation to choose a model for a subagent. Named policy
eligibility applies when a completion request arrives with that policy name.
[Agent model recommendations](docs/routing.md#agent-model-recommendations)

## Implemented design

The policy name continues to be the logical inference identity. A policy with
`eligibility` evaluates an operation and stream requirement against an exact
deployment and model before asking the existing router to rank candidates. The
compiler applies the same constraints to the final attempt plan, including the
default and explicit backups. A locally declared credential or keyless
endpoint ties the candidate to the instance whose locality was evaluated.

Capability metadata extends `model_capabilities` with an operation map. Each
operation carries its own streaming capability and operator-controlled
availability. The existing image and PDF
overrides serve as declarations of native support when a policy requires them.
The first consumers are the three completion dialects. Embedding, image and
speech operation names can be declared for future consumers. Grouped locality
is an operator declaration on the provider instance. A policy must list each
allowed category explicitly, including cloud. Existing priority and fallback
order express preference, including local-first and private-first. Weighted
routing keeps its existing algorithm over the eligible pool.

The walker executes the filtered plan using the existing retry, metering and
stream commitment rules. Conventional aliases and unconstrained policies keep
their current behavior. A constrained policy name requires a request plan,
including when a catalog spelling collides with the name. Per-key model access,
budgets, workspace retention requirements and organization scope continue to
apply. Authorized policy writers can control which policy is in force through
the existing scopes.

The policy document and operator configuration are sufficient for this slice.
Existing rows deserialize with eligibility disabled, and the database schema and
inference request models require no migration. The existing explain path reports
the exclusion reasons for a non-streaming chat case. Runtime drops use the same
reason values in gateway logs. Usage records continue to identify the selected
policy, serving attempt and fallback. The dashboard presents
constrained policies read-only so edits preserve their eligibility fields.

[Configuration and rollback](docs/routing.md#route-by-deployment-locality-and-capability) · [Compiler](src/gateway/services/routing/compiler.py) · [Execution](src/gateway/api/routes/_attempts.py)

## Follow-up decisions

- Use request-derived native input or format requirements once a completion
  adapter can distinguish requests that require original binary input from those
  that permit extraction. Evaluate tool calling, structured output and limits
  using per-operation declarations where they become useful.
- Consider passive readiness as a separate input after its identity includes
  the resolved endpoint, credential scope, model and operation. Define which
  failure classes are authoritative, when evidence expires, what unknown means,
  and how a target becomes eligible again. A discovery verdict alone is
  insufficient evidence about inference readiness.
- Reuse pricing and attempt observations if named policies later rank by
  estimated cost, latency or region. A comparison needs the relevant pricing
  units and interaction mode, and a stable rule for missing measurements. The
  existing automatic catalog scoring and budget ceilings can be evaluated
  independently of this change.
- Evaluate managed and tenant-owned targets only after metadata can be bound
  to the actual credential and regional endpoint before ranking. For explicit
  per-request consent, define authorization at the calling application and a
  verifiable, request-bound contract before adding a gateway signal.
- Extend one additional operation through its normal admission and settlement
  path before sharing the execution scaffold. Long-running jobs and cancellation
  need a stable upstream identity after submission. Binary endpoints need tests
  for replayability and the adapter's multipart or byte response contract.

## Upstream contribution plan

Lead with a small pull request titled
`feat(routing): add capability and locality eligibility to named policies`.
Include the policy examples, compatibility and rollback behavior, tests for
cloud authorization and access rejection, and the OpenAI-compatible mock
transport contract test. Ask maintainers to review the deployment-locality
attestation and completion-only scope. Keep dynamic readiness, managed
credentials and additional endpoints in independent proposals so each can be
reviewed against its own evidence and contract.
