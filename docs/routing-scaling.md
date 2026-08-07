# Routing at scale and on the hosted platform

A design note on how the learned router's conversation stickiness behaves as
Otari scales out, and the target design for making it durable (including hybrid
mode). This is forward-looking: v1 ships the in-process design described first,
which is correct for a single standalone process. The rest is the plan, not yet
built. Tracked alongside [#187](https://github.com/mozilla-ai/otari/issues/187).

Context: a router is named by a [routing policy](routing.md#let-a-router-choose-learned-routing)
(`select: [{router: knn, candidates: [...]}]`). What follows concerns the decision
cache, not the policy itself, which is plain config.

## What "trace stickiness" is

With `trace_sticky` granularity (the default), the router decides once per
conversation and reuses that decision for the conversation's later turns, so a
multi-turn agent run is not re-routed (and possibly flipped to another model) on
every step. Reuse needs two things:

1. A **trace identity** that is stable across a conversation's turns.
2. A place to **remember** `trace identity -> chosen model`.

## The v1 design (in-process)

- **Identity.** The client may send `Otari-Conversation-Id`; the router keys on
  that, namespaced per user. Absent the header, it hashes the conversation's
  opening messages (everything before the first assistant turn), which is stable
  across turns but cannot disambiguate two conversations that open identically.
- **Memory.** A per-process, in-memory map (`trace key -> model`), bounded by an
  LRU, held on a `KnnRoutingMemory` instance that is cached per backend-config
  signature so it survives across requests in that process.

This is correct and cheap for a **single standalone process**. A miss is always
safe: the router simply re-routes, and the safety gates keep a re-decision from
being worse than serving the policy's default target.

## Where it breaks as you scale

1. **Stickiness does not survive horizontal scaling.** The map is process-local.
   Behind a load balancer with more than one replica, consecutive turns of one
   conversation can land on different workers; the second worker has no memory of
   the first's decision and re-routes. Trace stickiness silently degrades toward
   per-step routing, which is exactly the behavior it exists to prevent. Not a
   correctness bug, but the value evaporates in the deployment that needs it.

2. **Restarts and autoscaling reset it.** Every deploy or scale event starts
   workers with an empty map, so in-flight conversations lose their stickiness at
   that boundary. On a busy, frequently-deployed service this means stickiness is
   perpetually partial.

3. **The content-hash fallback collides on agent traffic.** Without a
   conversation id, two conversations whose opening (system preamble + first user
   turn) is byte-identical share a trace key and therefore a routing decision.
   Agent frameworks that open every run with the same scaffolding hit this
   routinely, so one user's traffic can collapse onto a handful of trace keys.
   This bites even in a single process. The fix is a real conversation id; the
   opener hash is only a fallback.

4. **The in-process bounds are process-global, not per-user.** The decision map is
   one bounded LRU shared by every user, so a noisy caller can evict another's
   still-active decisions. The per-config backend instance cache has no eviction or
   TTL either. Fine for one standalone config; a leak surface under many users.

5. **No TTL.** A cached decision is reused for the life of the trace regardless
   of newer preference data, and is only dropped by LRU pressure. A long-lived
   agent trace can hold a stale decision indefinitely.

## Why hybrid mode needs the durable design from day one

Routing policies, and therefore routing memory, are standalone-only today: in
hybrid mode the connected platform resolves the model for every request, so a
local policy name is refused with a 400. When learned routing reaches the hosted
platform it lands in an environment that is multi-tenant, multi-replica, and
autoscaled by definition, so every problem above is the default case rather than
an edge case. The in-process map cannot be the system of record there.

## Target design

- **Conversation id as the identity.** Thread a real conversation/session id
  (the `Otari-Conversation-Id` header today; a platform-provided trace id when
  available), namespaced per user. Fall back to the opener hash only when no id
  is supplied, and document that fallback as best-effort.

- **Shared decision store.** Move `(user, conversation) -> {model, decided_at}`
  into a store every replica can read: Redis with a TTL, or a
  `routing_trace_decisions` table. The TTL bounds staleness (#5) and reclaims
  space without a global LRU (#4). This makes stickiness survive replicas and
  restarts (#1, #2).

- **In-process map demoted to an L1 cache.** Keep the per-process map as a hot
  read-through cache in front of the shared store, with per-user bounds so one
  caller cannot evict another's entries. A miss falls through to the shared store
  before it falls through to a re-route.

- **Embedding signal stays granularity-scoped.** `step` routes on the current
  turn; `trace_sticky` anchors on the conversation opener so that a store miss on
  another replica reproduces the trace's first decision deterministically rather
  than drifting.

None of this changes the safety model: a miss at any layer degrades to a safe
re-route, and a decline degrades to the policy's default target. Neither can be
worse than the policy would have been with no router at all.
