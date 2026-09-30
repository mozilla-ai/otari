# Backend domains

The gateway's backend is a modular monolith: one process and one deploy, with
the code cut by domain. This page gives the target shape of a domain and what
each domain owns.

A module in its domain's target location belongs to that domain by its path. A
module still in the old shape moves to its target location when its domain does.

## The target shape

The layers stay the top-level folders. Services and repositories are one
package per domain inside their layer. Routes, schemas, exceptions and models
are one module per domain. A domain's package and module names are its name
with underscores (`api_keys`).

| Layer | Path | Does | Must not |
| --- | --- | --- | --- |
| Routes | `api/routes/<domain>.py` | Parse the request, call one service, return a schema | Run a query, hold business rules, define schemas inline |
| Services | `services/<domain>/`, whose `__init__.py` exports the one service and the types its public methods use | Use cases: business rules and orchestration | Run a query, hold the session, touch HTTP, import another domain's repositories |
| Repositories | `repositories/<domain>/`, with modules that end in `_repository.py`, and the bundle a service receives in `<domain>_repositories.py` | Every query, over `BaseRepository`; flush, never commit | Hold business rules, commit |
| Schemas | `schemas/<domain>.py` | Pydantic request and response models, and their mapping from ORM rows | Anything else |
| Exceptions | `exceptions/<domain>_exceptions.py` | The domain's error classes, each with its HTTP status | Handle errors |
| Models | `models/<domain>.py` | ORM tables, and the closed vocabulary of each string column that has one | Hold logic |
| Core | `core/<subject>.py` | What a deployment is, and the vocabulary its own wiring is written in | Hold a domain's business rules, run a query |

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

## The domains

Each domain has a section below. The shared set follows them. A route module
whose name starts with an underscore is a shared helper, which the target shape
moves out of the routes layer.

The boundary check reads each `###` heading below as a domain name, so a
heading is the domain's name in lower case with hyphens. A section names a
module only when neither its path nor the section's prose shows its domain.

Two groups of modules fail the domain test and are split here. Tenancy holds
sign-in and organization management, which are separate sets of use cases, so
it is **identity** and **organizations**. Providers-and-models holds provider
credentials and the model catalog, so it is **providers** and **catalog**.

### identity

Who a person is and how they sign in: passwords, passkeys, OAuth, dashboard
sessions, email verification and reset, the profile, and deployment-wide
account administration.

Its tables sit in `models/tenancy.py` today, which organizations holds.

### organizations

Organizations, workspaces, members, invitations, email-domain claims, first-boot
provisioning, the setup guide, and the gateway's billing users.

It defines `MembershipListener`, the interface budgets implements to react to a
membership change without organizations importing budgets. It also owns
`models/users.py`, `repositories/users_repository.py` and
`services/workspace_scope.py`.

### api-keys

The deployment's and the members' API keys, and which models a key may reach.

It also owns `services/model_access.py` and `services/bootstrap_service.py`.

### budgets

Ceilings, reservations, reset periods and per-member policies.

`models/budgets.py` holds the scope, reset-alignment and reservation-status
vocabularies, with the columns they name, and the schemas and services import
them from there.

### pricing

The deployment price list, organization rate overrides and upstream price
snapshots.

It also owns `models/pricing_schemas.py`.

### providers

Provider credentials: instances configured at runtime, organization-scoped
provider keys, their health, and what a dispatch needs to reach a provider.

`tenancy/org_provider_key_service.py` has three divider sections (organization
keys, workspace overrides, model restrictions) and splits along them.
`models/providers.py` also holds the model alias table, which moves to catalog.

### catalog

The models a caller may see and name: the catalog, discovery, capabilities,
short spellings and aliases.

It also owns `services/tenancy/organization_model_access.py`.

### routing

Routing policies, their compiled plans and the router backends.

### files

Uploaded files: the Files API, the `file_objects` table, and a file's
lifecycle, including its expiry and the sweep that gives its storage back.
The bytes sit in a pluggable blob backend. The row holds the metadata and
the reference to them. The domain owns `ports/file_storage_port.py` and
`adapters/file_storage_adapter.py`.

The model never calls files, so it is not a tool. Inference normalizes an
uploaded file into a request. Tools hands one to a sandbox and returns one
from a tool call. The sandbox bridge and retention worker use `FileService`
for output registration and retention sweeping; neither receives a
Files repository. The service accepts produced-file metadata as `NewOutput`
and maps it to the repository's row type internally. It owns short database
transactions, with output compensation and cleanup storage calls outside them.

### tools

The tools the gateway runs itself: the tool loop, MCP, web search, web
retrieval and code execution.

It owns the code execution, MCP server and web search policy ports in `ports/`,
and their adapters in `adapters/`.

`mcp_server_port.py` names where a workspace's MCP servers come from. One
deployment holds those rows and another asks a peer that holds them for it, so
the composition root binds the implementation and no caller reads a mode. A
resolved server is connected to directly; neither implementation proxies MCP
traffic.

`web_search_policy_port.py` names where a workspace's web search policy comes
from, in the same two ways. The policy says who may search and how far. A
tools service applies it to a request with one rule on every plane, and
neither implementation carries a search.

**The tool test.** A tool is something the model calls during a request. The
domain holds the registry, the loop and each tool's settings. A capability the
model does not call is not a tool, even when a tool uses it. A large tool
splits inside tools, behind the registry interface, not into a new domain.

### guardrails

Inference Guardrails: checks that run on a request before the provider is
called, and an organization's guardrail configuration. Distinct from
`agent-guardrails` below, which checks what a coding agent did to a
repository; the two share no route, table or identifier.

### agent-guardrails

The Hook Server that evaluates an Agent Guardrails policy against caller-submitted
evidence. The evaluator is pure policy code with no database, so the domain has
a service package and no repository.

Its evaluator is `otari_agent.domain`, in the `otari-agent` workspace member
(`cli/`), so `otari hook` can run without the gateway.

### usage-and-telemetry

Usage rows, the usage log writer, OTLP ingest and coding-agent telemetry.

It also owns the route helper `api/routes/_billing_schemas.py`.

### inference

The request path: the completion dialects, the pass-through endpoints, batches
and the Playground.

### platform

Deployment settings, health, modes, maintenance mode and mail.

`control_plane/` is how a deployment asks the control plane a peer runs for it
what a workspace may do. It is a Gateway in Fowler's sense and an
anticorruption layer in Evans's: it holds how a deployment asks a peer for
policy and credentials, and raises this codebase's own errors so the peer's
status codes stop at its edge. Usage reporting still builds its own call from
the API layer. `ResolveEndpoint` is a closed set, so it answers questions and cannot
grow into a route for the data plane's own traffic.

`deployment.py` holds `Plane`, which names the control plane and the data
plane, and the value that says which of them a process serves. `surface.py`
says which deployments publish a dashboard page and `feature.py` shapes an
optional feature, so the three together are how a build describes itself.

### alerts

Alert destinations, rules, send-once delivery and the test send. It knows
nothing about what triggers an alert and imports no other domain. Each trigger
lives in the domain it watches and calls the alerts service.

Nothing is built yet. Its code goes in the target shape when it is built.

### overview

The dashboard overview's summary: the counts and the budget health that the
page shows, in one answer. It is a read model across api-keys, organizations
and budgets, so it has no tables of its own and writes nothing.

`overview/overview_repository.py` reads the tables of those three domains
directly, and it takes the session instead of extending `BaseRepository`. The
target shape has the overview service ask each domain's service for its data.

The overview has no slot of its own in the order of work. Its queries move with
each domain it reads, and budgets is the first.

### feedback

This domain holds the messages a signed-in dashboard user sends to the Otari
team. The gateway forwards each one to otari.ai without the caller's
credentials, and limits how often one caller may send. It is a core feature,
switched by `feedback_enabled`.

### Shared

Cross-cutting modules that several domains import. They stay where they are.

- Services: `url_safety.py`, `secret_box.py`, `file_extractors.py`
- Repositories: `base_repository.py`
- Exceptions: `_base.py`, `shared_exceptions.py`
- Models: `base.py`, `money.py`, `secret_fields.py`

`file_extractors.py` turns bytes into text and holds no state. Inference
imports it from `content_normalizer.py`, and tools from `web_extraction.py`.
The files code does not import it.

## Order of work

One domain at a time, in this order:

1. budgets, the pilot for the steps below.
2. providers and catalog, the largest.
3. files, which tools depends on.
4. tools, together with the built-in tool interface.
5. usage-and-telemetry.
6. pricing.
7. routing.
8. identity and organizations: splitting `organization_service.py` and
   `errors.py`.
9. api-keys, guardrails, agent-guardrails and platform, which are small.
10. inference last, after the tool loop is split by dialect.

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

Code moves and behavior changes stay in separate commits.

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
