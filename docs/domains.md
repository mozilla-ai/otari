# Backend domains

The gateway's backend is a modular monolith: one process and one deploy, with
the code cut by domain. This page gives the target shape of a domain, and
assigns every module under `services/`, `api/routes/`, `models/` and
`repositories/` in `src/gateway/` to one domain or to the shared set.

## The target shape

The layers stay the top-level folders. Services and repositories are one
package per domain inside their layer. Routes, schemas, exceptions and models
are one module per domain. A domain's package and module names are its name
with underscores (`api_keys`).

| Layer | Path | Does | Must not |
| --- | --- | --- | --- |
| Routes | `api/routes/<domain>.py` | Parse the request, call one service, return a schema | Run a query, hold business rules, define schemas inline |
| Services | `services/<domain>/`, whose `__init__.py` exports the one service and the types its public methods use | Use cases: business rules and orchestration | Run a query, hold the session, touch HTTP, import another domain's repositories |
| Repositories | `repositories/<domain>/`, with modules that end in `_repository.py` | Every query, over `BaseRepository`; flush, never commit | Hold business rules, commit |
| Schemas | `schemas/<domain>.py` | Pydantic request and response models | Anything else |
| Exceptions | `exceptions/<domain>_exceptions.py` | The domain's error classes, each with its HTTP status | Handle errors |
| Models | `models/<domain>.py` | ORM tables | Hold logic |

How a domain fits together:

- **One service per domain, with a small public API.** Each public method is
  one use case. The implementation sits in the package's private modules, whose
  names start with `_`.
- **Constructor injection.** The service receives its own domain's
  repositories, the Unit of Work, config, ports, and the services of other
  domains it needs. It never receives the session or another domain's
  repository, so it cannot run a query.
- **One Unit of Work per request or worker job.** Only a service opens a
  block.
  [Who commits](../.github/skills/backend-standards/SKILL.md#who-commits) gives
  the rules.
- **Builders** live in `api/deps.py`. A worker job calls the same builder with a
  Unit of Work over its own session. Nothing under `services/` may import
  `api/`, so the code that starts a worker passes the builder in.
- **Imports** follow the
  [layer and import rules](../ARCHITECTURE.md#the-modular-monolith).
- **Reacting to another domain.** Dependencies between domains run one way.
  When a domain must react to a change in a domain that does not depend on it,
  the domain where the change happens defines a listener interface and receives
  an implementation by constructor injection (Observer, and the Dependency
  Inversion Principle). The listener runs inside the caller's transaction and
  never commits. A package root may export a listener it offers another domain.
- **Divider comments** that cut a module into sections mean the module splits
  along them.
- **The domain test.** A domain that cannot offer a small public API is more
  than one domain.

Most modules are not in this shape yet. New and moved code follows the target,
not the module beside it.

## The shape today

Measured on `main` at `ecd07b13`, 2026-09-17. A module "runs queries" when it
imports a query builder (`select`, `update`, `delete` or `insert` from
SQLAlchemy or SQLModel) and calls `execute`, `exec`, `scalar`, `scalars` or
`get` on a session.

| Measure | Count |
| --- | --- |
| Service modules | 108, of which 71 sit flat at the top of `services/` |
| Service modules that run queries | 39, plus 2 that only call `session.get` |
| Route modules | 72 |
| Route modules that run queries | 17, plus 1 that only calls `session.get` |
| Route modules that define Pydantic models inline | 42 |
| Model modules | 19 |
| Repository modules | 9: a base, `users_repository.py`, and 7 under `tenancy/` |
| Service packages per domain | 1: `services/tools/`, which holds the built-in tool registry and no service yet. `services/mail/`, `services/routing/` and `services/tenancy/` are older subpackages |
| Repository packages per domain | None. `repositories/tenancy/` is an older subpackage |
| Modules in `schemas/` | None; the package does not exist |
| Modules in `exceptions/` | None; `services/tenancy/errors.py` holds every tenancy error in 1,297 lines |

## The domains

Fifteen domains plus a shared set. A module appears once. Paths are relative to
their layer's directory. A domain package is listed by its directory, which
covers every module inside it. A route module whose name starts with an underscore is
a shared helper, which the target shape moves out of the routes layer.

Two groups of modules fail the domain test and are split here. Tenancy holds
sign-in and organization management, which are separate sets of use cases, so
it is **identity** and **organizations**. Providers-and-models holds provider
credentials and the model catalog, so it is **providers** and **catalog**.

### identity

Who a person is and how they sign in: passwords, passkeys, OAuth, dashboard
sessions, email verification and reset, the profile, and deployment-wide
account administration.

- Routes: `admin.py`, `auth_oauth.py`, `auth_password.py`,
  `auth_password_reset.py`, `auth_profile.py`, `auth_session.py`,
  `auth_signup.py`, `auth_webauthn.py`; helper `_public_auth.py`
- Services: `tenancy/user_service.py`, `tenancy/webauthn_service.py`,
  `tenancy/deployment_user_service.py`, `tenancy/email_address.py`,
  `tenancy/tokens.py`, `tenancy/verification_email.py`,
  `tenancy/password_reset_email.py`, `oauth_service.py`, `password_service.py`,
  `dashboard_session_service.py`
- Repositories: `tenancy/user_repository.py`

Its tables sit in `models/tenancy.py` today, which organizations holds.

### organizations

Organizations, workspaces, members, invitations, email-domain claims, first-boot
provisioning, the setup guide, and the gateway's billing users.

- Routes: `organizations.py`, `workspaces.py`, `invitations.py`,
  `workspace_activation.py`, `users.py`
- Services: `tenancy/organization_service.py`, `tenancy/workspace_service.py`,
  `tenancy/authorization.py`, `tenancy/invitation_email.py`,
  `tenancy/organization_domain_service.py`, `tenancy/domain_verification.py`,
  `tenancy/provisioning_service.py`, `tenancy/workspace_activation_service.py`,
  `tenancy/errors.py`, `workspace_scope.py`
- Repositories: `tenancy/organization_repository.py`,
  `tenancy/organization_member_repository.py`,
  `tenancy/organization_domain_repository.py`,
  `tenancy/invitation_repository.py`, `tenancy/workspace_repository.py`,
  `users_repository.py`
- Models: `tenancy.py`, `users.py`

### api-keys

The deployment's and the members' API keys, and which models a key may reach.

- Routes: `keys.py`, `organization_keys.py`
- Services: `model_access.py`, `bootstrap_service.py`
- Models: `api_keys.py`

### budgets

Ceilings, reservations, reset periods and per-member policies.

- Routes: `budgets.py`, `scoped_budgets.py`, `organization_budgets.py`,
  `workspace_member_budget_policies.py`
- Services: `budget_service.py`, `scoped_budget_service.py`,
  `budget_periods.py`, `budget_reservation_ledger.py`, `budget_retiming.py`,
  `tenancy/organization_budget_service.py`,
  `tenancy/workspace_budget_default_service.py`
- Models: `budgets.py`

### pricing

The deployment price list, organization rate overrides and upstream price
snapshots.

- Routes: `pricing.py`, `organization_pricing.py`
- Services: `pricing_service.py`, `pricing_init_service.py`,
  `pricing_refresh_service.py`, `organization_pricing_service.py`
- Models: `pricing.py`, `pricing_schemas.py`

### providers

Provider credentials: instances configured at runtime, organization-scoped
provider keys, their health, and what a dispatch needs to reach a provider.

- Routes: `providers.py`, `org_provider_keys.py`
- Services: `provider_store_service.py`, `provider_health_service.py`,
  `provider_metadata_service.py`, `provider_kwargs.py`,
  `bedrock_gateway_auth.py`, `tenancy/org_provider_key_service.py`
- Repositories: `tenancy/org_provider_key_repository.py`
- Models: `providers.py`, `provider_keys.py`

`tenancy/org_provider_key_service.py` has three divider sections (organization
keys, workspace overrides, model restrictions) and splits along them.
`models/providers.py` also holds the model alias table, which moves to catalog.

### catalog

The models a caller may see and name: the catalog, discovery, capabilities,
short spellings and aliases.

- Routes: `models.py`, `catalog.py`, `aliases.py`
- Services: `model_catalog_service.py`, `model_discovery_service.py`,
  `model_capabilities.py`, `model_identity.py`, `merged_catalog_service.py`,
  `catalog_selectors.py`, `selector_index_service.py`, `alias_service.py`,
  `tenancy/organization_model_access.py`

### routing

Routing policies, their compiled plans and the router backends.

- Routes: `routing.py`, `routing_memory.py`, `organization_routing.py`
- Services: `routing/backends.py`, `routing/compiler.py`, `routing/decide.py`,
  `routing/knn.py`, `routing/weighted.py`, `policy_store.py`
- Models: `routing.py`

### tools

The tools the gateway runs itself: the tool loop, MCP, web search, web
retrieval, code execution and files.

- Routes: `tools.py`, `tool_settings.py`, `search.py`, `search_tools.py`,
  `web_search_backend.py`, `workspace_web_search.py`, `mcp.py`,
  `workspace_mcp_servers.py`, `workspace_code_execution_policy.py`,
  `files.py`; helper `_tools.py`
- Services: `_tool_loop.py`, `tools/`, `mcp_loop.py`, `mcp_loop_messages.py`,
  `mcp_loop_responses.py`, `mcp_client.py`, `mcp_stateless.py`,
  `sandbox_backend.py`, `search_backend.py`, `web_search_backend.py`,
  `web_search_budget.py`, `web_search_providers.py`, `web_extraction.py`,
  `web_fetch_service.py`, `web_retrieval_backend.py`,
  `web_retrieval_network.py`, `web_retrieval_policy.py`,
  `search_tool_store_service.py`, `tool_settings_service.py`,
  `tool_format.py`, `tool_usage.py`, `file_service.py`, `file_store.py`,
  `file_extractors.py`, `tenancy/workspace_mcp_server_service.py`,
  `tenancy/workspace_web_search_service.py`,
  `tenancy/workspace_code_execution_policy_service.py`
- Models: `tools.py`, `mcp.py`

### guardrails

Guardrails that run on a request, and an organization's guardrail
configuration.

- Routes: `organization_guardrails.py`
- Services: `guardrails.py`, `guardrail_catalog.py`,
  `tenancy/organization_guardrail_service.py`
- Models: `guardrails.py`

### agent-gates

The Hook Server that evaluates an Agent Gates policy against caller-submitted
evidence. The evaluator is pure policy code with no database, so the domain has
a service package and no repository.

- Routes: `hooks.py`
- Outside the four layers: `agent_runtime/`, which becomes the domain's service
  package

### usage-and-telemetry

Usage rows, the usage log writer, OTLP ingest and coding-agent telemetry.

- Routes: `usage.py`, `organization_usage.py`, `otlp.py`, `agent_telemetry.py`;
  helper `_billing_schemas.py`
- Services: `usage_admin_service.py`, `external_usage_service.py`,
  `log_writer.py`, `agent_telemetry_service.py`,
  `agent_telemetry_admin_service.py`, `claude_code_import.py`
- Models: `usage.py`

### inference

The request path: the completion dialects, the pass-through endpoints, batches
and the Playground.

- Routes: `chat.py`, `messages.py`, `responses.py`, `embeddings.py`,
  `images.py`, `audio.py`, `rerank.py`, `moderations.py`, `batches.py`,
  `playground.py`; helpers `_pipeline.py`, `_attempts.py`, `_platform.py`,
  `_passthrough.py`, `_normalize.py`, `_schema_derive.py`, `_helpers.py`
- Services: `batch_service.py`, `content_normalizer.py`, `vision.py`,
  `upstream_redaction.py`, `playground_service.py`, `playground_dispatch.py`
- Models: `inference.py`, `playground.py`

### platform

Deployment settings, health, modes, maintenance mode and mail.

- Routes: `settings.py`, `bootstrap.py`, `health.py`, `maintenance_mode.py`,
  `hosted_mode.py`, `hybrid_mode.py`, `mail.py`
- Services: `runtime_settings_service.py`, `maintenance_mode_service.py`,
  `master_key_service.py`, `mail/mailer.py`, `mail/message.py`,
  `mail/templates.py`, `mail/transports.py`
- Models: `platform.py`

### alerts

Alert destinations, rules, send-once delivery and the test send. It knows
nothing about what triggers an alert and imports no other domain. Each trigger
lives in the domain it watches and calls the alerts service.

- No code yet. It arrives in the target shape.

### Shared

Cross-cutting modules that several domains import. They stay where they are.

- Services: `url_safety.py`, `secret_box.py`
- Repositories: `base_repository.py`
- Models: `base.py`, `money.py`, `secret_fields.py`

## Order of work

One domain at a time, in this order:

1. budgets, the pilot for the steps below.
2. providers and catalog, the largest.
3. tools, together with the built-in tool interface.
4. usage-and-telemetry.
5. pricing.
6. routing.
7. identity and organizations: splitting `organization_service.py` and
   `errors.py`.
8. api-keys, guardrails, agent-gates and platform, which are small.
9. inference last, after the tool loop is split by dialect.

## What one domain change does

The same six steps for every domain, as separate commits, or separate pull
requests for a large domain:

1. Move the schemas out of the route modules into `schemas/<domain>.py`.
2. Move the queries out of the routes and services into
   `repositories/<domain>/`, over `BaseRepository`.
3. Build the service package `services/<domain>/`: one service with a small
   public API, built with its repositories and the Unit of Work. Helper modules
   become its private modules, split along any divider comments.
4. Move the commits into Unit of Work blocks.
5. Move the errors into `exceptions/<domain>_exceptions.py`, keeping each
   class's status.
6. Remove the domain's modules from every baseline in the boundary check.

Code moves and behavior changes stay in separate commits. The counts above go
down for the domain, and the pull request records them.

## Sources

- Simon Brown, "Package by component and architecturally-aligned testing"
  (2016), republished as "The Missing Chapter" in Robert C. Martin, *Clean
  Architecture* (2017): https://simonbrown.je/modular-monolith/
- Erich Gamma, Richard Helm, Ralph Johnson and John Vlissides, *Design
  Patterns: Elements of Reusable Object-Oriented Software* (1994): Observer
- Robert C. Martin, *Agile Software Development, Principles, Patterns, and
  Practices* (2002): the Dependency Inversion Principle
- Martin Fowler, *Patterns of Enterprise Application Architecture* (2002):
  Service Layer, Repository and Unit of Work,
  https://martinfowler.com/eaaCatalog/
- Martin Fowler, "Inversion of Control Containers and the Dependency Injection
  pattern" (2004): https://martinfowler.com/articles/injection.html
- John Ousterhout, *A Philosophy of Software Design* (2018), chapter 4,
  "Modules Should Be Deep"
- Harry Percival and Bob Gregory, *Architecture Patterns with Python* (2020),
  chapters 2, 4 and 6: https://www.cosmicpython.com/book/
