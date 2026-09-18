# Naming conventions: `web/`

Names are the documentation that cannot go stale. The goal is that a reader can tell what a
thing is and what it does without opening it.

## Files

- Components: `PascalCase.tsx`, named for the component (`ProvidersPage.tsx`, `RowActions.tsx`).
- Hooks: `useCamelCase.ts(x)` (`useDeployment.tsx`, `useEntitlements.tsx`).
- Helpers and pure modules: `camelCase.ts` (`urlState.ts`, `tableSelection.ts`, `format.ts`).
- Tests: `<Name>.test.ts(x)`, beside the file they cover. One file is deliberately
  not: `design-system/navigation/tabs.test.tsx` covers `Tab`, `TabRow` and
  `Segmented` under a single header docstring arguing all three ARIA departures
  together, so it has no one subject to be named after and splitting it would
  duplicate that rationale across two files. A test with one subject takes its
  subject's name.
- Routes: whatever file-based routing requires (`tools.guardrails.tsx`), never renamed by hand.

## Variables

Say what the value is, not what type it happens to be. `data`, `info`, `item`, `obj`, and
`res` describe nothing. Booleans read as a predicate, a yes/no question in English:
`isPending`, `hasBudget`, `canRevokeKey`, `shouldShowBanner`. That covers every boolean a
reader meets, including props and the boolean fields of a hook's return value
(`useKeysScope()` answers `isDeploymentWide` and `isReady`, not `deploymentWide` and
`ready`): a bare noun phrase reads as the thing itself rather than as a question about it.

```ts
const isSignedIn = session !== undefined
const hasEnforcedBudget = budget?.limit_usd !== undefined
```

**A single letter says less than any of those nouns**, so it is held to the same rule. A
lambda binding a domain object names it: `(entry) => entry.latency_ms`, not
`(e) => e.latency_ms`, which in a React file reads as an event to everyone who has written a
handler. The same goes for `p` over a placement or a provider, `o` over an option, `w` over a
workspace.

Two idioms are exempt and stay:

- `(a, b)` in a comparator. `rows.sort((a, b) => a.cost - b.cost)` is read the same way
  everywhere and naming the pair adds nothing.
- The parameter of a `setState` updater, which is the previous value of the thing already
  named on the left: `setCount((n) => n + 1)`.

## Functions

Start with a verb: `formatCost`, `parseSearch`, `revokeKey`, `describeSurface`. A function
named for its return value (`cost()`, `search()`) reads like a variable at the call site.

Pure domain logic belongs in a module of its own rather than inside a component, because that
is what makes it directly testable: `features/organization/roles.ts`,
`shared/helpers/tableSelection.ts`, `shared/helpers/format.ts` are the pattern.

## Callback props are `on*`, handlers are what they do

This tree names the prop for the event (`onConfirm`, `onRevoke`, `onSelectAllMatching`,
`onSaved`), and that is the convention to follow. It is deliberately not otari-ai's
`handle<Subject><Verb>` prop naming: renaming ~200 call sites would be churn with no reader
benefit, and the two trees converge on component structure rather than on prop spelling.

Inside a component, name the function for the intent, not the event: `revokeSelected`,
`applyFilters`, `dismissBanner`. `handleClick` says only that a click happened.

React DOM props keep their DOM names (`onClick`, `onKeyDown`); HeroUI's interactive components
take `onPress`, not `onClick` (see [components.md](./components.md)).

## Constants

`SCREAMING_SNAKE_CASE` for a true compile-time constant, `camelCase` for a derived
configuration object:

```ts
const PRICING_PAGE_SIZE = 1000
const MOBILE_QUERY = "(max-width: 767px)"

const usageFilters = { model: models, workspace: workspaceId }
```

Query keys are module constants (`const MODELS = "models"`), never inline literals, so a
query and the mutation that invalidates it reference one symbol instead of repeating a string
that can diverge. A mistyped *identifier* fails to compile; a mistyped *literal* would not,
which is exactly why the literal is written once. See [data-fetching.md](./data-fetching.md).

A literal table that drives a union carries `as const`, so the values stay literals and the
derived type is the set rather than `string[]`: `THEME_PREFERENCES` in
`shared/hooks/useTheme.tsx`, and the nav registry's
`] as const satisfies readonly NavSection[]`.

## Vocabulary that carries meaning

Three words mean specific things here and are not interchangeable:

- A **surface** is the deployment axis: a group of management APIs a standalone gateway
  serves. It comes from `GET /v1/bootstrap` through `useSurfaces()`.
- A **capability** is the entitlement axis: what a build or a customer is licensed for, from
  `useEntitlements()`.

A nav entry may gate on both and they compose as AND. Calling a surface a capability in a
variable name is how the two vocabularies start merging, which is exactly what the M5
convergence has to avoid (see [web/AGENTS.md](../../../web/AGENTS.md)).

Similarly: the credential an operator pastes into the sign-in screen is the **master key**
until the deployment is claimed and an **email** and **password** after it (`sign_in_methods`
in the deployment bootstrap says which); the credentials the gateway stores for upstream
providers are **provider credentials**; the keys the gateway issues to callers are **API
keys**. The UI, the tests, and these docs use those names and no synonyms.

## Array work reads declaratively

In [typescript-and-react.md](./typescript-and-react.md), with the rest of the language
conventions: it governs how a transformation is written rather than what it is called.
