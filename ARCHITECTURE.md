# Architecture

Otari is an OpenAI-compatible LLM gateway, built to be extended. Its core defines a small set of interfaces, called **ports**, and delegates real work to **adapters** that implement them. The core depends only on the ports, never on a specific adapter, so you can supply your own adapter and run it with Otari without changing Otari's code. If you have your own code-execution platform, for example, an adapter that satisfies `CodeExecutionPort` lets Otari drive it; the same is true for routing, identity, and the other ports below.

An **overlay** is a build that layers its own adapters (and its own routers and pages) on top of Otari through these extension points. otari.ai's edition is one such overlay, but it is one among possible many, and nothing in this document is specific to it. This is the open-core shape: Otari is the Apache-2.0 base that stands on its own, and an overlay adds capabilities along a published boundary.

This document is for contributors changing the *shape* of the system. It explains where that boundary is, what Otari's core provides versus what an overlay supplies, and how to add a capability without crossing the line. For the request-level, user-facing picture, start with [docs/index.md](docs/index.md) and [docs/modes.md](docs/modes.md).

## Status: this is a north-star document

This describes the architecture the codebase is **heading toward**, not the whole of what exists today. Otari today is the gateway (the data plane) plus a standalone management API; the ports package, the composition container, and the bootstrap hook are in the tree, and the rest, including the control-plane UI described below, arrives as Otari grows into the full open-source base.

Read it with that in mind:

- **"Today" statements** are grounded in what is actually in `src/gateway/` right now, and say so.
- **"Planned" statements** describe target structures that do not exist in the tree yet. Where a structure is planned, the doc names where it will live rather than pretending it is already there.
- A few capability-line rows are explicitly marked **provisional**: the split between what Otari ships and what an overlay adds is a working assumption for those rows, pending a decision, not settled fact.

As the seam is built out, this document and the mechanical boundary check that enforces it (see [Cardinal rules](#cardinal-rules-for-contributors)) are updated together, so the human-readable boundary and the enforced one do not drift.

## The two planes

Otari separates a **data plane** from a **control plane**.

- **Data plane**: the gateway hot path. It authenticates the request, applies input guardrails, runs any built-in tools, and dispatches the provider call through [`any-llm`](https://github.com/mozilla-ai/any-llm). It is stateless and scales with traffic. Today it runs in `src/gateway/`: the request lifecycle in `src/gateway/api/routes/chat.py` (and its siblings `messages.py`, `responses.py`), with streaming in `src/gateway/streaming.py`. `src/gateway/` today also houses the standalone control plane described below, so the package as it stands is not the data plane alone.
- **Control plane**: everything off the hot path. It owns users, orgs, workspaces, keys, budgets, usage, and the routing decision. It is stateful and database-backed. It makes the policy decisions; the data plane executes them.

On the request hot path the two meet through the **resolve protocol**. Before a provider call, the data plane asks the control plane which provider and credentials to use, and gets back an ordered list of attempts:

```http
POST /gateway/provider-keys/resolve  ->  attempts[]
```

Each attempt carries `provider`, `model`, `api_key`, `api_base`, and `managed`. The gateway walks the list in order (retrying the next attempt on a retryable failure, committing on success) and makes **zero routing decisions of its own**. The full wire contract is in [docs/hybrid-mode-protocol.md](docs/hybrid-mode-protocol.md).

The resolve protocol is the seam made concrete on a live boundary: a simple control plane returns a plain ordered list; a richer one could return a smartly selected list; and the gateway cannot tell which answered. The decision-making sits behind the protocol, on the control-plane side, and the gateway stays identical no matter what answers.

Provider resolution is the seam on the request hot path, but it is not the only one. Other capabilities (authorization, identity, entitlements, billing, code execution) sit behind their own ports; several of them back admin-facing and control-plane surfaces rather than the LLM call path.

Today the control plane resolves two ways, selected by [mode](docs/modes.md):

- **Standalone** (default): the gateway resolves against its own local database (users, keys, budgets, usage in `src/gateway/models/`). This is the open-source control plane in its simplest form.
- **Hybrid** (`OTARI_AI_TOKEN` set): the gateway delegates resolution to a peer over HTTP (`src/gateway/api/routes/_platform.py`). Any service that implements the protocol can answer; otari.ai is the reference peer.

Hybrid mode is a *network* form of this seam, and it is worth not conflating it with an overlay. In hybrid mode a remote control plane answers the resolve protocol over HTTP, out of the gateway's process; the peer can be any service that implements the protocol. An overlay, by contrast, is an *in-process* build that binds its own adapters into the composition container (see [How a port is resolved](#how-a-port-is-resolved)) and runs in the same process as the core. Both let something other than the plain local logic answer; the difference is whether that something runs over the network or in the same process. So a hybrid peer and an overlay are two ways to reach the seam, not the same thing.

```mermaid
%%{init: {'theme':'base','themeVariables':{'fontFamily':'-apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif','fontSize':'14px','lineColor':'#5e8496'}}}%%
flowchart LR
    clients("Your apps<br/>(OpenAI or Anthropic SDKs)")
    otari("<b>Otari gateway</b><br/>data plane: stateless hot path")
    providers("Providers<br/>via any-llm, 40+")
    control("Control plane<br/>stateful, off the hot path<br/>users, keys, budgets, routing")
    clients -->|requests| otari
    otari -->|dispatch| providers
    otari -. "resolve (once per request)" .-> control
    classDef hero fill:#3c6678,stroke:#2c4d5c,color:#ffffff;
    classDef cp fill:#f4f0e8,stroke:#d8c9a8,color:#22303a;
    classDef plain fill:#eef4f7,stroke:#b8ccd6,color:#22303a;
    class otari hero;
    class control cp;
    class clients,providers plain;
```

## How a data plane reaches what it does not own

A data plane holds no database and no store of its own. That is the design, not a gap to work around. When a capability needs a remote service or leaves durable state, the data plane gets authority from the control plane and then acts directly. The two rules below apply to every capability on the seam: model inference, MCP, web search, code execution, files, and every capability added after them.

### The narrowest authority that lets it go direct

The control plane answers policy questions, and it never carries customer traffic. Completions, search queries, sandbox code, file bytes and captured content go from the data plane straight to where they are served or stored. The control plane returns the narrowest authority that lets the data plane do that, and there are three cases. Refer to a case by its name.

| Name | Case | The control plane returns | Examples |
|---|---|---|---|
| Workspace key | The workspace owns the credential | That credential | A workspace's own provider keys, its MCP servers, search with its own key |
| Platform grant | The platform owns the resource and can mint a grant | A scoped grant | The code execution front door, the files bucket |
| Platform key | The platform owns the credential and the upstream cannot accept a grant | The credential, to the platform's own data plane only | Managed models, managed search |

The platform is whoever runs the control plane: otari.ai, or anyone who builds their own edition on Otari. The case depends on who runs the resource in that deployment, not on the seam. A resource that someone else runs, such as an operator's own sandbox backend, never sees a grant and keeps its own credential.

A **scoped grant** covers one resource, only the operations named for it, and a short deadline. A file read and a file write need separate grants, while a sandbox session is one grant for the operations of that session, and no grant permits listing, because a grant that can enumerate is a tenant isolation hole. The data plane receives grants and never mints them. A grant is never logged, never stored on a record, and never sent to a client.

MCP is the reference shape. The resolve returns the server's own URL and the workspace's credential, and the data plane connects direct.

### Durable state: Try-Confirm/Cancel

Where the work leaves durable state, such as a stored file or a recorded run, the exchange is **Try-Confirm/Cancel (TCC)**. The model is Helland's tentative operations ("Life beyond Distributed Transactions: an Apostate's Opinion", CIDR 2007). The name is Pardon's (Atomikos, and Pardon and Pautasso, 2011).

1. **Try.** The data plane asks the control plane to authorize the operation. The control plane checks policy and quota, allocates any identifier the operation needs, records the intent in a state that is not yet usable, and returns the identifier with a grant.
2. The data plane does the work directly against the resource the grant names.
3. **Confirm.** The data plane reports the outcome, and the control plane makes the record usable. Confirm carries an idempotency key, so a retry cannot create a second record. A Confirm that keeps failing is retried with backoff and then parked for reconciliation, never dropped.
4. **Cancel.** A failure cancels explicitly where it can. Where it cannot, a sweep cancels each Try record that is past its deadline.

The budget path already runs this inside one database (`services/budgets/`). `reserve_budget` is Try, `reconcile_reservation` is Confirm, `refund_reservation` is Cancel, and the reservation sweep reclaims a hold that nobody released. Across the plane boundary there is no shared transaction, so the idempotency key and the sweep are obligations, not options. Try comes first so that a record always exists before its side effect, which keeps the sweep a walk of stale Try records rather than a scan of the remote store.

TCC is the wrong tool where one transaction already covers the work. There, use a listener that runs inside the caller's transaction, as `services/tenancy/membership_listener.py` does. Work that leaves nothing durable, such as a web search, needs only the authorization and the direct call.

> **Where this stands.** The rules are decided, and the grant is not built yet. The grant object, the token a front door verifies, the Try-Confirm/Cancel rules and content encryption are specified in [docs/hybrid-mode-protocol.md](docs/hybrid-mode-protocol.md#grants). Files is the first capability to implement them ([#1750](https://github.com/mozilla-ai/otari/issues/1750)), and the code execution front door follows ([docs/code-execution-protocol.md](docs/code-execution-protocol.md)). Web search and code execution can still be configured to send traffic through the control plane, and [#1688](https://github.com/mozilla-ai/otari/issues/1688) removes that. Each capability's wire contract is written in [docs/hybrid-mode-protocol.md](docs/hybrid-mode-protocol.md) before its code.

## The extension seam: ports and adapters

Otari's core defines interfaces called **ports**, and delegates the real work to **adapters** that implement them. Because the core depends only on the ports and never on a specific adapter, you change what a capability does by binding a different adapter, and the core itself does not change. That dependency line, with the core on one side and the adapters on the other, is the boundary between what Otari ships and what an overlay plugs in.

There are two ways to extend the system across that boundary, because there are two different kinds of thing sitting on it:

- **Work which the core hands off: change it by swapping the adapter.** Some work the core relies on but does not implement itself, such as authorizing a request, choosing the provider attempts, running the inference call, executing sandboxed code, or billing for usage. For each of these the core defines a port and depends on it, and an adapter supplies the actual implementation. Binding a different adapter changes the behavior, and the core stays as it is.
- **Surfaces exposed to callers: change it by adding a surface and gating it.** The API routes and frontend pages that callers reach. Each is registered into a central list the core owns (a router list on the backend, a nav registry on the frontend) and switched on by an entitlement. Extending here means registering a new route or page and gating it, whether that surface ships in Otari's core or comes from an overlay. Nothing is swapped; a surface is added and made conditional.

A **port** is a domain-named interface (a Python `Protocol`), named for what it does, never for how it is implemented (`RoutingPort`, not `SmartRouterClient`). The seam is built around this set of ports:

| Port | Responsibility |
|---|---|
| `AuthzPort` | Authorization decisions (who may do what). |
| `EntitlementPort` | Which capabilities a deployment is entitled to. |
| `IdentityProviderPort` | Authenticating users/sign-in. |
| `RoutingPort` | Choosing the provider/model attempts for a request. |
| `ModelProviderPort` | Resolving the deployment-owned credential that serves a request bringing none. |
| `CodeExecutionPort` | Running model-generated code in a sandbox. Two core adapters: the published protocol over HTTP, and E2B's hosted sandboxes. |
| `BillingPort` | Metering and charging for usage. |
| `GrowthSignalPort` | Telling an outside CRM or support messenger about a user's lifecycle. |
| `TelemetryStoragePort` | Where captured agent telemetry is stored and read back. |
| `TraceStoragePort` | Where agent traces (a session and the spans inside it) are stored and read back. |
| `DataKeyPort` | Where the keys that encrypt captured trace content come from. Two core adapters: `OTARI_SECRET_KEY`, and an AWS KMS key. |
| `FileStoragePort` | Where the bytes behind an uploaded file are kept. Three core adapters: a local directory, S3, and any fsspec filesystem. |
| `ProviderFilePort` | Reading and writing the files a provider holds: a copy an attached file gets for the provider's own code execution, and a file that code produced. The core adapter reaches them through any-llm. |
| `ApiKeyFormatPort` | The shape of the API keys a build mints, and where a presented key is checked. |
| `RateLimitStorePort` | Where rate-limit counts and concurrency slots are kept. Two core adapters: this process's memory, and Redis for a count every replica shares. |
| `AgentModelRecommenderPort` | Which model a coding agent's subagent should start on, and at what price. The core adapter puts one choice question to the decision model the deployment configured and quotes no price of its own. |

The cardinal property: **every port ships with a working adapter in Otari's core**, a real lightweight implementation or an honest [Null Object](https://en.wikipedia.org/wiki/Null_object_pattern). Otari must stand alone with no overlay present. `BillingPort`, for example, is a Null Object in the core: it is present and callable, and does nothing, so nothing in the core needs to know whether real billing exists anywhere.

```mermaid
flowchart LR
    subgraph driving["Surfaces that call in: add + gate"]
        UI["Frontend + nav registry"]
        API["API routers"]
    end
    subgraph core["Layered core"]
        SVC[Services]
        REPO[Repositories]
        MOD[Models]
    end
    subgraph driven["Work behind ports: swap the adapter"]
        AP[AuthzPort]
        EP[EntitlementPort]
        IP[IdentityProviderPort]
        RP[RoutingPort]
        MP[ModelProviderPort]
        CP[CodeExecutionPort]
        BP[BillingPort]
        GP[GrowthSignalPort]
        TP[TelemetryStoragePort]
        TR[TraceStoragePort]
        DK[DataKeyPort]
        FP[FileStoragePort]
        PF[ProviderFilePort]
        KP[ApiKeyFormatPort]
        RL[RateLimitStorePort]
    end
    UI --> API --> SVC --> REPO --> MOD
    SVC --> AP
    SVC --> EP
    SVC --> IP
    SVC --> RP
    SVC --> MP
    SVC --> CP
    SVC --> BP
    SVC --> GP
    SVC --> TP
    SVC --> TR
    SVC --> DK
    SVC --> FP
    SVC --> PF
    SVC --> KP
    SVC --> RL
```

> **Where the ports are today.** `src/gateway/ports/` holds seventeen of them, `ModelProviderPort`, `BillingPort`, `EntitlementPort`, `GrowthSignalPort`, `TelemetryStoragePort`, `TraceStoragePort`, `DataKeyPort`, `IdentityProviderPort`, `ApiKeyFormatPort`, `CodeExecutionPort`, `CodeExecutionPolicyPort`, `FileStoragePort`, `ProviderFilePort`, `McpServerPort`, `WebSearchPolicyPort`, `RateLimitStorePort` and `AgentModelRecommenderPort`, each with a working core adapter in `src/gateway/adapters/`. Fifteen have core callers. `ApiKeyFormatPort` is asked by every key-mint site (the two key routers, the setup guide, the playground and the first-run bootstrap) and by the verify path, which routes a presented key before it looks it up; the core adapter mints the open-source `tk-` shape and checks every key locally, and a hosted overlay binds a region-tagged, checksummed format behind it (otari-ai#1665). `TelemetryStoragePort` is resolved by the OTLP receiver, the telemetry read endpoints and the purge paths. `TraceStoragePort` is resolved by the trace writer, which the lifespan starts and each request's recorder hands its spans to. Its database adapter settles each call on a Unit of Work of its own from the metering pool, and where the deployment serves no control plane the binding is a Null store. `DataKeyPort` is bound from `trace_content_key_backend` and has no core caller yet: content capture resolves it when it lands. Both of its adapters are core, because a data plane seals content and runs no overlay; the KMS one sits behind the `kms` extra, as the S3 file store sits behind `s3`. `ModelProviderPort` is asked on the standalone dispatch path (`resolve_dispatch_provider` in `api/routes/_pipeline.py`) for a candidate the credential ladder could not serve, after an organization's key, a stored instance and `config.yml` have all missed; a routed multi-candidate chain does not reach it yet, because those candidates are credentialed by the routing compiler, which is synchronous and does no I/O. It is also asked on every catalog read (`catalog_scope` in `services/merged_catalog_service.py`) for the models the deployment advertises, and by the selector-index lifespan worker, which resolves it per tick through a closure over the container that `main.py` hands in, rather than through `gateway.api.deps`. `IdentityProviderPort` is resolved by the OAuth sign-in route. Its core adapter follows `open_signup`, as the signup form does: when the setting is disabled, the adapter signs in only an existing account, and when it is enabled, the adapter also creates an account with its own organization and workspace. `CodeExecutionPort` is resolved on the completion path, and only where the deployment configured a sandbox to reach. `FileStoragePort` is bound from `files_backend` and resolved twice: by the lifespan, which hands the request path its store, and by the retention sweep. The binding builds one store per app and reuses it, so the bytes a request wrote are the bytes the sweep reclaims. Above it, the `/v1/files` routes, the content normalizer resolving a `file_id` back to bytes, and the sandbox file bridge all take that store as a dependency. `ProviderFilePort` is resolved beside it, by the lifespan and the retention sweep, and the files service and the sandbox file bridge receive both together as `FileBackends`. It opens a session for one provider account with the credential a request dispatches with, which is what lets a build that must not hand a managed credential to its data plane bind a transfer of its own. `McpServerPort` is resolved on the completion path and by the MCP routes, and `WebSearchPolicyPort` and `CodeExecutionPolicyPort` on the completion path. Each reads the workspace's rows where this deployment holds them and asks its control plane where it does not. The request preamble's early read of the code execution policy, which decides where an uploaded file goes, does not reach `CodeExecutionPolicyPort` yet. `RateLimitStorePort` is resolved once by `create_app`, which counts `rate_limit_rpm` in the store it returns; the in-process limits on sign-in, feedback and the public catalog do not use it yet. `AgentModelRecommenderPort` is resolved by the subagent recommendation route and run inside the decisions scaffold, which meters the call: at the deployment's own rate where the adapter quotes no price, which is the core adapter, or at the price the adapter quotes, held and charged through `BillingPort`. That is `BillingPort`'s one core caller so far; the completion path meters nothing through it yet. `EntitlementPort` is still the seam's mechanism rather than its surface, so the choice it describes (what a deployment is entitled to) is made by the mode switch and the hand-wired dependencies described next. The remaining ports in the table above arrive as they gain callers.
>
> `IdentityProviderPort` is also where the table's own wording is narrower than it reads. It is listed as "authenticating users/sign-in", but authenticating is not behind it: proving somebody controls a Google or GitHub account is protocol work that does not vary by edition, so it stays a plain service. What the port carries is the *policy* applied to a proven identity, which is the half an overlay replaces.

## How a port is resolved

Ports are resolved through a **composition root** and a small **container**, not by scattering concrete class names through the code.

- The **container** is a process-level registry of `Port -> factory` bindings, built **once at startup**. It is a plain Python object (a mapping of port to factory), not a third-party dependency-injection framework and not entry-point auto-discovery. Only a handful of ports ever need swapping, and only at startup, so a thin explicit registry is preferred: a contributor can read the whole wiring in one file, and there is no install-time magic to trace.
- The **composition root** is the single place allowed to name a concrete adapter. It binds the defaults at startup. Everywhere else refers to the *port*, and asks the container for whichever adapter is bound.

Shape of a resolution:

```python
# composition root (src/gateway/container.py), at startup: bind the core adapters
def _billing_adapter(session: AsyncSession | None) -> BillingPort:
    return NullBillingAdapter(session)

container.bind(BillingPort, _billing_adapter)

# dependency (src/gateway/api/deps.py): names the port and the container, never an adapter
def get_billing_port(db: PortSessionDep, container: ContainerDep) -> BillingPort:
    return container.resolve(BillingPort, db)
```

The session a factory receives is the request's, and it is `None` in hybrid mode, where the gateway has no local database at all. An adapter that needs one has to say what it does without.

An overlay (or your own deployment) rebinds ports to its own adapters **without editing any Otari source file**, by supplying a bootstrap module that the container invokes at startup, selected declaratively by configuration: `OTARI_BOOTSTRAP=module:callable`. The callable is imported once, after the core defaults are bound, and receives the container:

```python
def register(container: Container) -> None:
    container.bind(BillingPort, _wallet_billing_adapter)
    container.contribute_router(RouterContribution(capability="billing", router=wallet_router))
```

With nothing configured nothing is imported, the defaults stand, and Otari boots standalone. A selector that is set but cannot be loaded fails startup rather than quietly falling back, because a build nobody chose is worse than a gateway that will not start.

A contributed router is the additive half of the seam, and it is gated rather than swapped: Otari mounts it behind `require_capability(...)`, which resolves `EntitlementPort` and answers a request for an unentitled capability with the same 404 a path nothing serves gets. That gate is the server-side half of the entitlement axis; hiding a nav item in the dashboard is not authorization.

Entitlement is not authentication either, and the mount point adds none. A capability names no caller, so on an entitled deployment a contributed route is reachable by anyone unless the router says otherwise. A contribution declares the credential each of its routes needs on the route, the way Otari's own routers do; there is no router-level default to mount, because the right answer differs per route, a contributed route may be deliberately public, and the header check resolves a database session a hybrid gateway does not have.

> **Where this lives in the tree.** The composition root is `src/gateway/container.py`; it is built once per app in `create_app` (`src/gateway/main.py`) and attached to `app.state` beside the other shared resources, so two apps in one process never share one. Ports are resolved from it through dependencies in `src/gateway/api/deps.py`, which is also where the rest of composition is still hand-wired: the container took over the ports, not every dependency, and a plain single-implementation service stays wired directly.

Not every service goes through a port. Most code has a single implementation and stays plain (see [when a capability earns a port](#cardinal-rules-for-contributors)); only capabilities with a real second implementation are resolved through the container.

## Capability lines: what the core ships vs what an overlay adds

This is the open-core line: for each capability, what Otari's core ships and what an overlay can add. Most of the management plane is plain core code with no port of its own, because each of those features has a single implementation that an overlay has no reason to replace. Having no port of its own does not mean a feature is ungoverned: managing users or budgets is still authorized through `AuthzPort` and gated by `EntitlementPort` like everything else. A feature earns its own port only when a genuine second implementation is in play (see [when a capability earns a port](#cardinal-rules-for-contributors)).

| Capability | Where it lives | Notes |
|---|---|---|
| Users, orgs, workspaces, teams, invitations, budgets, usage, BYO provider keys | **Core** (plain, no port) | The management plane: plain CRUD, one implementation each. Routing and telemetry storage have their own rows below because, unlike these, they sit behind a port. Usage rows stay here in every build: they are the money path, not analytics. |
| RBAC | **Core base + overlay adapter** *(provisional)* | Base roles and org scoping in the core; deeper roles, fine-grained permissions, and audit from an overlay adapter. Split pending an open decision. |
| SSO | **Core base + overlay adapter** *(provisional)* | Social sign-in and passkeys in the core; enterprise SSO (for example SAML, enterprise OIDC, directory provisioning) from an overlay adapter. Split pending an open decision. |
| Routing | **Core base + overlay adapter** *(provisional)* | Ordered fallback and policies in the core; a richer model-selection strategy from an overlay adapter. Split pending an open decision. |
| Model inference | **Core port + hosted adapter** | Self-hosting your own backends is a first-class path in the core; a hosted, metered inference backend comes from an overlay. See the managed-models section of [docs/modes.md](docs/modes.md). |
| Code execution | **Core port and adapters, managed service from an overlay** *(provisional)* | The core adapters reach a sandbox backend through the published protocol or E2B. A hardened, managed sandbox is a service an overlay runs behind the published protocol, and the data plane reaches it with a platform grant. An overlay binds no adapter here, because the adapter sees the code. Interface still provisional. |
| Trace storage | **Core port, store from an overlay** | Agent traces are content, so the data plane writes them, best-effort after accounting and never a billing source: to this deployment's own database in the core, or with a platform grant to a store an overlay runs for many tenants. Where a data plane has no store to write to, the core binds a Null store. A person reads them back through the control plane. |
| Content keys | **Core port, two core adapters** | The key-encryption key for captured trace content comes from `OTARI_SECRET_KEY` or from an AWS KMS key, chosen by `trace_content_key_backend`. Both are core because a hybrid data plane, which seals content, runs the plain build. |
| Telemetry storage | **Core port, store from an overlay** | Captured telemetry is content, so the data plane writes it: to this deployment's own database in the core, or with a platform grant to a store that an overlay runs for many tenants' retention and query volume. A person reads it back through the control plane. |
| Billing (wallet/payments) | **Overlay-only** | A Null Object (no-op) adapter in the core; real billing exists only in an overlay. |

The **provisional** rows (RBAC, SSO, routing) share one open question: how deep the core base goes before an overlay adapter takes over. That is an open design decision for the project maintainers, not settled yet and not a contributor's to assume; treat those lines as a working assumption until it is decided and recorded here.

The inference and code-execution ports share a shape: a compute-heavy backend behind a port, chosen by the control plane, executed in the data plane. Self-hosting is a first-class path in the core, not a degraded one. In the resolve protocol this seam is already latent: `managed: false` with an `api_base` is the self-hosted path, and `managed: true` is a hosted, metered backend (see [docs/modes.md](docs/modes.md)).

## Deployment and entitlements

Two gates decide whether a piece of behavior runs. They **compose but never merge**, because they answer different questions:

- **Surface** is the topology axis: *does the process serving this URL host this surface at all?* It is answered by the deployment bootstrap (`GET /api/v1/bootstrap`), whose `surfaces` list the dashboard shell reads once before it renders, and it is the reason a hybrid gateway shows no management UI: its control plane is otari.ai, not itself. Standalone and hosted deployments host the surface; a data-plane gateway does not. It is named `surface` and never `capability`, because otari.ai already spends that word on the entitlement axis below, down to a nav item's `capability` field, and the two vocabularies meet in one shell when the control-plane UI converges.
- **Entitlement** is the licensing axis: *is this capability enabled for this deployment at all?* It is scoped per deployment, never per user. It is resolved by `EntitlementPort`, whose core adapter grants every base capability and reports every overlay-only one as absent; a real resolver is an overlay adapter.

The first is about *where the code is running* and the second about *what this customer bought*. Both are client-side conveniences over server-side authorization: hiding a surface never grants access to it, and the server authorizes every request behind one regardless.

In the dashboard both meet on one nav entry, which is where the vocabulary earns its keep: `web/src/app/nav/registry.ts` declares a destination's `surface` and `capability`, and `useNavVisibility` composes them as AND, so either one hides the link and the shell answers the route behind it with a panel rather than a page. A third field, `operatorOnly`, sits beside them and belongs to neither axis: it is about who is signed in rather than about the deployment, it is answered by the one surface it gates rather than by a rule of its own, and it is described in [web/AGENTS.md](web/AGENTS.md). An overlay replaces `web/src/app/nav/overlaySections.ts` (or `overlayNavItems.ts`, for a destination that belongs inside a section the base owns) to register its own destinations, without editing a base source file. The same build-time module override carries a contribution that is not a destination at all but something inside a piece of chrome the base owns: `web/src/app/nav/overlayWalletSlot.tsx` is the slot the top bar mounts for a balance this gateway has none of, and rule 6 is why it exists, since a chip there could otherwise be contributed only by editing the top bar itself. [web/AGENTS.md](web/AGENTS.md) lists the seams and the rule that a base module reaches one by its `@/…` specifier rather than relatively.

Be clear about how much of that is built. The surface axis is real and served: `GET /api/v1/bootstrap` answers it. `EntitlementPort` now exists on both sides of the wire, but **no endpoint serves entitlements to the browser**, so the two halves answer independently:

- **In the browser**, the axis resolves from the constant that is the default value of the context in `web/src/shared/hooks/useEntitlements.tsx`. It grants `BASE_CAPABILITIES` and reports everything else absent. An overlay answers it for real by rendering `EntitlementProvider`, through `web/src/app/overlayEntitlementResolver.tsx`: the seam the shell mounts above the navigation and the routes, whose base default renders its children unchanged so this build falls through to the constant. A resolver that answers asynchronously reports `isLoading` and the shell waits on it, rather than telling a visitor an entitled page is not served here while the answer is still in flight.
- **On the server**, `EntitlementPort` resolves through the container, and its core adapter (`src/gateway/adapters/entitlement_adapter.py`) answers with its own `BASE_CAPABILITIES`. That is what `require_capability` gates a contributed router on, so a route an overlay mounts into this process refuses for itself rather than trusting a hidden link. Hiding a link is not authorization; this is the half that is.

Both constants are empty, and deliberately so: no base nav entry and no base route is gated on a capability, because the one candidate is routing, whose split this document still marks provisional. They are meant to agree, so a capability the base grows is added to both at once.

The axis does not enforce against the operator, who owns the process; do not let a surface built on the client gate assume a resolver exists.

## Cardinal rules for contributors

These are the rules that keep the boundary from eroding. They apply to anyone adding or moving code across the seam.

1. **Anything that will have more than one implementation lands as a port plus a working core adapter.** If a capability will have a richer alternative behind it, its interface (the port) and a working default both live in the core.
2. **The core ships the base; the richer or specialized implementation lives in an adapter, not in the core.** Fallback routing is core; a model-selection strategy is an adapter. Keep the depth in the adapter.
3. **Every port has a working core adapter.** A real lightweight one or an honest Null Object. Otari must stand alone.
4. **Ports live in the core, in domain terms.** Name the port for the domain (`RoutingPort`), never for an implementation.
5. **Only the composition root names a concrete adapter.** Services, routers, and the frontend refer to ports; the composition root is the single place that binds a concrete one.
6. **An overlay never edits an Otari source file.** It registers into the extension points Otari exposes (the container, the router list, the nav registry) and supplies configuration. If extending an overlay *requires* editing an Otari file, that is a missing seam, and the seam belongs in Otari. Supplying configuration or a bootstrap module is not editing the core.
7. **Introduce a port only when it earns one.** A port that will only ever have one implementation is ceremony with no benefit. A capability earns a port only when a genuine second implementation is real, or a hard boundary (intellectual property, or a hosted service) runs through it. Where the adapter sees customer content, every adapter is in the core, and a hosted service sits behind a published protocol, not behind an overlay adapter. Plain CRUD and infrastructure (user, team, and trace management, and the bulk of orgs/workspaces management) stay concrete in the core. Where such data needs a store a hosted build replaces, only the store goes behind a port, as telemetry and trace storage do; the service that manages the data stays plain. "Most of the management plane is core" and "most services are not ports" are the same statement.

### The patterns these rules are built from

The rules above are this codebase's wording of four patterns. Naming them saves a contributor from deriving the shape again, and gives a review something to appeal to.

- **Ports and adapters**, also called hexagonal architecture (Cockburn). A port names a capability in domain terms; adapters implement it; the composition root picks one. Rules 1 through 5 are this pattern.
- **Gateway** (Fowler, *Patterns of Enterprise Application Architecture*). An object that encapsulates access to an external system. `services/control_plane/` is one: it holds how a deployment asks a peer for policy and credentials, so no module that asks has to know a peer is reached over HTTP. Usage reporting and the sandbox and web-search URL checks still build their own calls, and moving them is the remaining work.
- **Anticorruption layer** (Evans, *Domain-Driven Design*). A translation at the edge, so a peer's vocabulary does not spread inward. A Gateway here raises this codebase's own errors rather than the peer's HTTP statuses, and the API layer renders them. Without it, the peer's protocol reaches every caller and rule 4 is lost.
- **Replace conditional with polymorphism** (Fowler, *Refactoring*). Where a deployment's mode is read to choose behavior, the choice belongs in the binding rather than at the branch. A mode branch repeated across modules is the smell this removes.

These rules are enforced mechanically, not only in review. The boundary check (`scripts/check_architecture.py`, run by `make lint` locally and as its own step in CI) asserts the layering: ports may import models, exceptions, and core, but not services, the API, or any adapter; services may import ports but not a concrete adapter; only the composition root may import an adapter. This document is the human-readable companion to that check; the two are kept in step so the boundary the doc describes is the boundary CI enforces.

## The modular monolith

Otari's backend is a modular monolith: one process and one deploy, with the code cut into modules by domain (Simon Brown, "Package by component and architecturally-aligned testing", 2016, republished as "The Missing Chapter" in Robert C. Martin, *Clean Architecture*, 2017; Brown, "Modular Monoliths", GOTO Berlin 2018; <https://simonbrown.je/modular-monolith/>). Otari differs from Brown in one way: he packages by component first, and Otari keeps the layers as the top-level folders under `src/gateway/` and packages by domain inside the service and repository layers. A domain is held together across layers by its name and by the import rules below.

Services and repositories are one package per domain inside their layer, and a service package's root exports the domain's one service. Routes, schemas, exceptions and models are one module per domain, because they hold no hidden implementation. [DOMAINS.md](DOMAINS.md#the-target-shape) gives the path and job of each layer, what each domain owns, and the steps for moving one. The [backend standards](.github/skills/backend-standards/SKILL.md#layering) give the rules for writing code in each layer, including how a service is built and who commits.

**Layer and import rules.**

1. Nothing under `services/` imports `sqlalchemy` or `sqlmodel`.
2. Nothing under `api/routes/` imports `sqlalchemy` or `sqlmodel`.
3. Only the Unit of Work calls `commit()` or `rollback()`, only a repository imports `session_for`, and only `get_unit_of_work` or a worker factory constructs a `UnitOfWork`. The [backend standards](.github/skills/backend-standards/SKILL.md#who-commits) say how it is used.
4. Only a domain's own service package and the builders in `api/deps.py` import `repositories.<domain>`.
5. Code outside a domain imports only the package root of `services.<domain>`. A module whose name starts with `_` is private to its package.
6. Domain service packages never import each other in a cycle.

A service imports its own domain's repositories, the services of other domains, ports, models and exceptions, and never the API layer or an adapter. A repository imports models, and never a service or the API layer. A route imports its domain's service, schemas and exceptions.

**What the boundary check enforces.** `scripts/check_architecture.py` enforces part of the rules above today. It refuses a service that imports the API layer or an adapter, a repository that imports a service or the API layer, and a route that imports `sqlalchemy.orm`. It enforces rules 1 and 2. SQLModel counts as well as SQLAlchemy because it re-exports `select` and `Session`. Each of the two starts from a baseline, `SERVICE_DATABASE_IMPORT_BASELINE` and `ROUTE_DATABASE_IMPORT_BASELINE`, which names the modules still in the old shape, and a baseline only shrinks. It also refuses a new top-level module in `services/` or `repositories/`, so new code goes in its domain's package, starting from `FLAT_MODULE_BASELINE`, the flat modules still in those layers. It enforces rule 3 from `TRANSACTION_CONTROL_BASELINE`, the modules that still end a transaction themselves, and refuses an import of `session_for` outside `repositories/`, which needs no baseline. It also refuses a `UnitOfWork(...)` call outside `get_unit_of_work` and the module that defines the Unit of Work, where the worker factories live, so a request or a job has exactly one. It refuses a package in `services/` or `repositories/`, or a module in `schemas/` or `exceptions/`, whose name is not a domain that [DOMAINS.md](DOMAINS.md#the-domains) gives a section, starting from `DOMAIN_NAME_BASELINE`. A package on that baseline holds only the modules its entry lists, so new code cannot join a package that is not a domain yet. It enforces rule 4 for each domain [DOMAINS.md](DOMAINS.md#the-domains) gives a section, and lets a domain's repository package import its own modules. Service code outside every domain package is still in the old shape, and its path does not say which domain owns it, so rule 4 skips it until it moves. It enforces rule 5 for the same domains. It refuses a read of `is_hosted_mode`, `is_hybrid_mode`, `effective_mode` or `configured_mode`, or a call to `deployment_for`, under `services/`, starting from `SERVICE_MODE_READ_BASELINE`, because the behavior for each mode belongs in a binding, not in a branch. It gives each ORM model one repository module that may construct or query it, in `MODEL_ACCESS`, so a table has one writer. It refuses any other module that names the model where it does more than name a type, as an annotation, a cast or an `isinstance` check does, starting from the baseline on each model's entry. It also refuses a new model until the map gives it an entry. A model whose queries no repository holds yet names no repository. It refuses a default on a parameter or a class field typed as a listener, starting from `LISTENER_DEFAULT_BASELINE`, so a caller cannot skip a listener by leaving it out. Rule 6 is a review rule until the check covers it. No check can tell whether a service's API is small, so review carries that rule too. New and moved code follows the target shape, and never copies code named on a baseline. `make lint` runs the check, and CI runs it as its own step in `otari-lint.yml`, so a failure is named as an architecture failure. Three packages sit outside these layers by design. The laptop-side CLI at `cli/` (distribution `otari-agent`), which the gateway depends on for the Agent Guardrails evaluator and which the check keeps from importing the gateway or its server stack back, so the `otari` command can ship alone. And the web search and fetch libraries at `any-search/` and `any-fetch/`, which the check holds the same way, tests included, so each runs without otari.

**The feature registry.** A core feature is a domain in this shape plus one entry in `src/gateway/features.py`, the one wiring list for optional features. That list is a literal tuple edited by hand. The app asks each listed feature once, when it is built, whether it is enabled, and router registration, the lifespan and the deployment bootstrap all use that one answer. A feature's switch is therefore a startup setting, never one the dashboard can change. A listed feature mounts as core routes, ungated, and hosts its surface when its own setting enables it. `OTARI_BOOTSTRAP` is not involved. Nothing is discovered from installed packages, and a feature never registers itself on import.

## Where new code goes

Split a feature into its pieces of work before you choose a mechanism. One feature often has a piece in each plane, or a core piece beside an overlay piece. Ask these questions of each piece.

1. **Does Otari need it to stand alone?** Yes: the core. No: an overlay.
2. **Does it see customer content while it serves a caller's request?** Yes: the data plane. No: the control plane. Customer content is what a caller sends and receives: prompts, completions, tool inputs and outputs, search queries, sandbox code, file bytes and captured agent telemetry. A person who reads stored content is the one exception: the control plane decrypts it for them after an access check and an audit record, as [the protocol](docs/hybrid-mode-protocol.md#content-encryption) says.
3. **Is it an overlay piece that sees content?** Then it is a service, not an adapter. An overlay loads into a control plane, and a data plane runs the core alone. So the adapter that sees the content is in the core and speaks a published protocol. The overlay runs a service behind that protocol, and its control-plane piece mints the grant that the data plane presents to it. The data plane goes direct to the service.

Code execution is this shape. `CodeExecutionPort`'s core adapter speaks [the code execution protocol](docs/code-execution-protocol.md), and a platform runs a front door behind it. [How a data plane reaches what it does not own](#how-a-data-plane-reaches-what-it-does-not-own) gives the authority the control plane returns.

A data-plane piece that reaches a remote resource takes one of the three authority cases, and one that leaves durable state uses [Try-Confirm/Cancel](#durable-state-try-confirmcancel).

Then choose each piece's mechanism from the table below, by what you are adding, not by the extension point you already know.

| You are adding | Home | Mechanism |
|---|---|---|
| An optional feature any deployment may run, with its own routes, tables, settings, worker or page | Core | An entry in `src/gateway/features.py`, switched by a startup setting. See [How to add a core feature](#how-to-add-a-core-feature). |
| An always-on route or service with one implementation, such as a new endpoint in an existing domain | Core | The domain's own modules, with a new router registered in `register_routers` (`src/gateway/api/main.py`). No port and no registry entry. |
| A second implementation of work the core hands off, such as billing, code execution or an identity policy | A port and its adapters | See [How to add a capability](#how-to-add-a-capability). |
| An adapter, route or page that only an overlay ships, and that sees no customer content | The overlay | `OTARI_BOOTSTRAP`: bind a port, or contribute a router gated on a capability. Pages register through the dashboard's overlay modules (see [Deployment and entitlements](#deployment-and-entitlements)). |
| Work that only an overlay ships and that must see customer content | A service behind a published protocol | A core adapter speaks the protocol. The data plane calls the service with the authority the control plane returns, usually a [platform grant](#the-narrowest-authority-that-lets-it-go-direct). The overlay's control-plane piece owns the policy and mints the grant. |
| A built-in tool the model calls inside the tool loop | Core | **Planned:** one built-in tool interface that the tool loop dispatches through. Today each built-in tool is wired by hand into the tool loop (`src/gateway/services/mcp_loop.py` and its per-dialect siblings). |
| Code loaded at runtime from an installed package, a directory or a download | Not supported | None. The boundary check refuses `importlib.metadata`, `importlib_metadata` and `pkg_resources` under `src/gateway/`. |

`OTARI_BOOTSTRAP` is the overlay's hook, and it never loads or switches a core feature. It takes one module and an overlay already holds it, so a feature wired through it either displaces the overlay or makes the overlay register the feature a second time. A capability gates a route on what a deployment is entitled to, not on whether a feature is switched on, so a core feature names none.

A feature that lives in this repository lives under `src/gateway/`. The boundary check refuses any other top-level package under `src/`, where none of its layer rules would reach.

A new service or repository module goes in its domain's package, `src/gateway/services/<domain>/` or `src/gateway/repositories/<domain>/`, never beside the flat modules already in those layers. The boundary check refuses a new top-level module in either layer.

## How to add a capability

A step-by-step recipe for adding a capability without crossing the boundary. The `AuthzPort` seam is the reference template every later capability copies.

1. **Decide whether you need a port at all.** Apply rule 7. If the capability is plain management CRUD with no second implementation on the horizon, skip the ceremony: write a normal service and repository in the core and stop here. An optional feature follows [How to add a core feature](#how-to-add-a-core-feature). Only continue if a second implementation is genuinely on the table, or a hard boundary runs through it.
2. **Define the port.** Add a domain-named `Protocol` to `src/gateway/ports/`, one module per port. Its methods are `async`. It may depend on models, exceptions, and core only, never on services, the API, or an adapter.
3. **Route callers through the port.** Services depend on the port, resolved from the container; they never name a concrete adapter.
4. **Ship a working core adapter.** Add a real lightweight implementation, or an honest Null Object, to `src/gateway/adapters/`. Verify the capability behaves correctly with only this adapter present.
5. **Bind the default in the composition root.** Register `Port -> core factory` in `build_container` (`src/gateway/container.py`), through a named factory function whose return type is the port, so the type checker verifies the adapter satisfies it. Resolve it through a dependency in `deps.py`.
6. **If the capability has API or UI surface, add and gate it.** A core router is registered in `register_routers` (`src/gateway/api/main.py`); an overlay's is recorded on the container with `contribute_router` and mounted behind the capability it names. Register its nav item into the nav registry (`web/src/app/nav/registry.ts`), gated by the same entitlement. Do not swap anything on this side; add surface and make it conditional.
7. **Verify Otari still stands alone.** It must boot standalone with only the core adapters bound (no overlay bootstrap configured) and pass its smoke suite, and the boundary check must pass. Both are automated: `uv run --frozen --no-dev python scripts/oss_edition_smoke.py` is the smoke suite (run by `.github/workflows/otari-oss-edition.yml` on any pull request that touches the app, the migrations, or dependency resolution), and `make check-architecture` is the boundary check.

Once the seam exists, an overlay adds its own adapter by registering it through these same extension points, with zero edits to Otari's source. Building the seam is core work; using it is overlay work.

## How to add a core feature

A recipe for an optional feature the core ships. [The modular monolith](#the-modular-monolith) defines a core feature; a capability that needs a port follows [How to add a capability](#how-to-add-a-capability) instead.

1. **Write the feature's modules** in its domain's shape, as [The modular monolith](#the-modular-monolith) lays it out. One of them declares the feature's `CoreFeature` (`src/gateway/core/feature.py`). None of them imports `src/gateway/features.py`: the boundary check refuses a service or a route that imports the registry. Its queries go in its repository package, and its service is built with its own repositories and a Unit of Work, never the database session. The boundary check refuses a service or a route that imports `sqlalchemy` or `sqlmodel`, so neither can build a query or name the session type.
2. **Add its settings.** A setting goes in its domain's module under `src/gateway/core/settings/`. A domain with no module there adds one, and adds its class to the bases of `GatewayConfig` in `src/gateway/core/config.py`. Every field declares its settings view (`src/gateway/core/settings_view.py`). The feature's switch is one of these settings, and it stays out of `_SPECS` in `src/gateway/services/runtime_settings_service.py`, because `enabled` is asked once when the app is built and a dashboard override would never take effect.
3. **Add its tables.** A table goes in its domain's model module under `src/gateway/models/`, and a new module joins the import list in `src/gateway/models/__init__.py`. Its migration goes in `alembic/versions/`, chained to the current head. The tables live on core's metadata and core's chain, so switching the feature off leaves its tables and rows in place, and switching it back on changes no schema.
4. **Declare its metrics** in the module that increments them, on `REGISTRY` from `src/gateway/metrics.py`.
5. **List it.** Add the feature's `CoreFeature` to `CORE_FEATURES` in `src/gateway/features.py`. The entry gives the feature's name, its `surface` (a `Surface` from `src/gateway/core/surface.py`, or `None` for a feature with no page), its `enabled` check, its routers and an optional worker. Its routers mount beside the management routers and its worker runs under the lifespan, in standalone and hosted mode only; a hybrid gateway runs neither.
6. **Add its page.** Its nav entry in `web/src/app/nav/registry.ts` names the feature's surface, so wherever the feature is off the dashboard hides the entry and answers its route with a panel rather than a page. A surface whose name is not its route prefix needs an entry in `SURFACE_ROUTE_PREFIXES` (`tests/unit/test_deployment_bootstrap.py`), which checks that every published surface names a mounted route.
7. **Regenerate the artifacts** a new route owes, as "Generated Artifacts" in [AGENTS.md](AGENTS.md) lists them.
8. **Verify it switched on and switched off:** `make lint`, the tests, and `uv run --frozen --no-dev python scripts/oss_edition_smoke.py`.

## Glossary

- **Capability**: the unit of division along the seam. A vertical slice: a port, one or more adapters, optional API and UI surface, and the entitlement that governs it. "Routing", "billing", "code execution" are capabilities.
- **Port**: a domain-named interface (a Python `Protocol`) the core depends on. Ports live in the core.
- **Adapter**: a concrete implementation of a port. Otari ships a lightweight, always-present adapter for each; an overlay (or your own deployment) can supply a richer one.
- **Overlay**: a build that layers its own adapters, routers, and pages on top of Otari through its extension points, without editing Otari's source.
- **Composition root**: the single place that decides which adapter answers a port. Only it may name a concrete adapter.
- **Container**: the process-level registry of `Port -> factory` bindings, built once at startup.
- **Entitlement**: whether a capability is enabled for a deployment, at capability grain. The licensing axis.
- **Scoped grant**: a short-lived authority for the named operations on one resource, which the control plane mints and the data plane presents. See [How a data plane reaches what it does not own](#how-a-data-plane-reaches-what-it-does-not-own).
- **Try-Confirm/Cancel (TCC)**: how a data plane with no database leaves durable state. The control plane records the intent, the data plane acts, and a Confirm or a Cancel settles the record.
