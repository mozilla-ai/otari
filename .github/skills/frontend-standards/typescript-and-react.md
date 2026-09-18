# TypeScript & React conventions: `web/`

TypeScript runs in `strict` mode; `pnpm --dir web run typecheck` must pass. React 19.

## TypeScript

- **An absent value is an empty value first, `undefined` second, never `null`**, in your own
  types and props. Reach for the empty value the type already carries (`""`, `[]`, `{}`) before
  widening it: a union grows a case every reader and every call site has to handle, where an
  empty value is the one the code around it already handles. `ProvidersPage`'s add-provider
  form spells "no provider chosen" as an empty `providerId` rather than as
  `string | undefined`, and every check around it reads the same either way. Widen to
  `undefined` only where the type has no empty value that cannot collide with a real one, and
  say so where it is declared. `null` stays out either way: the API layer hands it back because
  it mirrors the server JSON, so convert at the boundary rather than letting it spread through
  the component tree. (`ApiError`-style third-party shapes that explicitly use `null` are the
  exception.) This applies to a `null` **carried over** by a refactor as much as to a new one:
  moving it is what puts it in your diff.
- **Named exports**, not default exports, for components/hooks/helpers, consistent names
  across imports, better tooling and tree-shaking. (`web/` already does this throughout.)
- **Named imports**, not namespace imports (`import * as …`).
- Take the API surface from the generated client (`import type { … } from "@/client"`,
  regenerated from the OpenAPI spec) and thread those types through `apiFetch<T>(…)`;
  don't fetch into `any` and don't hand-write a wire shape. The one sanctioned
  exception is `web/src/client/local.ts`, for the few shapes the spec does not
  describe (routing-policy bodies, `/dashboard-build.json`, `UsageFilters`), and each
  entry there says why; adding to it is a decision, not a shortcut. See
  [web/AGENTS.md](../../../web/AGENTS.md).
- Let inference work for locals; annotate function signatures and exported values.
- **`unknown`, not `any`, at a boundary you cannot type** (a thrown value, an opaque payload),
  and narrow it with a guard before use. `errorMessage(error: unknown)` in
  `design-system/feedback/errorMessage.ts` is the pattern: one place turns an unknown
  into a display string.
  Biome's `noExplicitAny` is off in `web/biome.jsonc` because the tree still has older `any`s,
  which makes it a convention rather than a lint error; do not add to the pile.
- **A discriminated union beats a bag of optionals** for anything with states. `{ status:
  "error"; message: string } | { status: "success"; data: T }` makes `state.data` on the error
  branch a compile error, where `{ status, message?, data? }` makes it a runtime `undefined`.
- **`as const` on a literal table** that drives a union (`THEME_PREFERENCES` in
  `shared/hooks/useTheme.tsx`, the nav registry's `as const satisfies readonly NavSection[]`),
  so the values stay literals and the derived type is the set rather than `string[]`.

## React

- **`onPress`, not `onClick`**, for HeroUI interactive components (see
  [components.md](./components.md)).
- **Correct dependency arrays** on `useEffect`/`useMemo`/`useCallback`. Clean up subscriptions,
  intervals, and event listeners in the effect's return.
- **Derive, don't duplicate.** Compute values from props/query data during render instead of
  copying them into `useState` and syncing with effects. Server state lives in TanStack Query,
  not in component state (see [data-fetching.md](./data-fetching.md)).
- **Stable `key`s** for lists, a stable id, not the array index.
- **The React Compiler is enabled** (`babel-plugin-react-compiler`, wired up in
  `vite.config.ts`), so memoization is the build's job by default and the plain expression is
  what to write. Reach for `useMemo`, `useCallback` or `React.memo` where it earns its place
  rather than by reflex or never: an expensive computation, a reference something else
  identity-checks, or a component the compiler could not optimize.
  It also means the rules of hooks are load-bearing: the compiler silently skips a component it
  cannot verify. See [performance.md](./performance.md).
- Keep a component per file, colocated with its test.

## A draft that resets when the value behind it moves

A control that holds an editable draft over a value the caller owns has to decide what happens
when that value moves. **Adjust the state during render. Do not run an effect**, and reach for a
`key` only when the draft is genuinely worth nothing.

```tsx
const [lastSeenValue, setLastSeenValue] = useState(value)
if (value !== lastSeenValue) {
  setLastSeenValue(value)
  // the condition is the design decision; see below
}
```

The effect form renders the stale value once, commits it, then renders again. It is also a
second source of truth that only converges after paint, and it is the shape the rule above is
about.

Three of these are in the tree and they differ only in the condition, which is where the thought
goes:

| Where | Condition | What it protects |
| --- | --- | --- |
| `design-system/forms/ComboBoxField.tsx` | `if (value !== typed)` | a value this field itself reported, so a list that drops a row does not move a value under a mounted field |
| `features/settings/SettingsPage.tsx` (`useDraft`) | `if (draft === lastSeenValue)` | an unsaved edit, so another operator's save does not take a half-typed value out from under the cursor |
| `design-system/data/TablePagination.tsx` | unconditional | nothing: the page box commits on Enter or blur, so a half-typed number is uncommitted by definition |

Because the condition differs at every site, these do not share a hook. One would need the
condition passed in, which is a knob on shared code to serve one caller and puts the three back
on different behavior by a different route.

**A `key` is not the general answer.** Keying on the value remounts the control exactly when the
value changes, which throws the draft away and drops focus mid-edit. That is correct only when
the draft should be discarded, and it is the wrong fix for anything the operator is still
typing: otari#1341 was a draft being discarded, so a `key` would have preserved the bug.

An effect remains right for synchronizing with something outside React, a subscription, a timer,
an observer, and those still owe cleanup.

## Array work reads declaratively

A loop that produces a value is a transformation written the long way. `map`, `filter`,
`reduce`, `some`, `every`, `find`, `findLastIndex`, `flatMap` say which transformation it is in
the first word, where `for` says only that something repeats and makes the reader hold an
accumulator to find out.

That covers **`for...of`, `for...in` and the index loop alike**:

```ts
// for...of building a lookup
const byWorkspace = new Map(placements.map((placement) => [placement.workspaceId, placement]))

// for...in over an object
const enabled = Object.entries(settings).filter(([, value]) => value.isEnabled)

// an index scan keeping the last match
const startIndex = timestamps.findLastIndex((value) => value <= target)
```

`for...in` has a second reason: it walks inherited enumerable keys and gives you strings, so
`Object.keys`, `Object.values` and `Object.entries` are both clearer and narrower.

**`forEach` is right when the body is genuinely only a side effect** and there is no value
coming back: aborting each controller in a set, appending each id to a `URLSearchParams`.
Reach for it there and a reader knows at the first word that nothing is being produced. What
it must not be is a `map`, a `filter` or a `reduce` with the result pushed into a variable
declared above it, which is the shape that hides the transformation from the reader and gives
the accumulator a chance to escape.

**Give that `forEach` a block body.** Biome's `suspicious/useIterableCallbackReturn` rejects a
concise arrow whose body is a call, because the arrow syntactically returns that call's value.
It reads the shape rather than the type, so a call returning `void` is flagged too:

```ts
ids.forEach((id) => params.append("request_group_id", id))    // lint error
ids.forEach((id) => {
  params.append("request_group_id", id)
})                                                            // correct
```

The rule is right to read the shape: a concise arrow in a `forEach` is one keystroke from
being a `map` whose result nobody took. (`(id) => void params.append(...)` is the rule's own
escape hatch and does pass lint, so it is not a review finding, but nothing here is written
that way and new code should not start. The `void` this codebase does use is the other one,
discarding a floating promise: `void queryClient.invalidateQueries(...)`.)

Two cases stay imperative, because each iteration decides whether there is a next one and no
array method expresses that:

- consuming a stream (`shared/api/playground.ts`'s SSE reader)
- a bounded request walk (`shared/api/paging.ts`), where the loop ends on a short page

## Testing

Vitest and Testing Library, colocated with the code they cover, mocking the transport rather
than the hooks. The rules, the harnesses in `src/tests/`, and the two Playwright suites are in
[testing.md](./testing.md).
