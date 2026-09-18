# Performance: bundle, rendering, and the compiler

## The React Compiler is enabled

`web/vite.config.ts` runs `babel-plugin-react-compiler` as a Babel pass of its own
(`@rolldown/plugin-babel`, configured by the `reactCompilerPreset` helper `@vitejs/plugin-react`
exports), so components and derived values are memoized at build time. It sits beside the React
plugin rather than inside it, which is where that plugin hosted it before its version 6.
otari-ai/frontend runs the same pass; keeping both configured alike is deliberate, because
components move between the trees. The plugin's other route, `react({ compiler: true })`, swaps
in oxc-transform-react, a Rust reimplementation still labeled experimental, and would change
which compiler decides what to memoize; the config says so at the site.

What follows from that:

- **Memoize where it earns its place: not by reflex, and not never.** The compiler handles the
  ordinary cases, so the default is to write the plain expression and let it decide. That is a
  default, not a ban, and a `useMemo` with a reason behind it is correct code rather than a
  finding.
- **What "earns its place" looks like.** A computation expensive enough that you can see it, a
  reference crossing into something that runs its own identity check (a memoized third-party
  component, a dependency array that cannot be flattened), or a component the compiler could
  not optimize, which is the case below. When the reason is not obvious from the call site,
  a short comment saves the next reader from deleting it to find out.
- **What it costs, so the judgment is an informed one.** A dependency array allocated and
  compared on every render, a hook slot per instance, and one more array that can go stale.
  For a cheap expression that is a net loss: `useMemo(() => a + b, [a, b])` does strictly more
  work than `a + b`. The compiler must also preserve the semantics of whatever you wrote, so a
  boundary drawn badly by hand stays drawn badly; it cannot reason through one and fix it.
- **Correct dependency arrays still matter**, because `useEffect` is not memoization. The
  compiler does not fix an effect that re-subscribes on every render or one that misses a
  dependency it reads.
- Rules of hooks still apply, and now they are load-bearing: the compiler bails out of a
  component it cannot prove follows them, silently, so a conditional hook costs optimization
  as well as correctness. **That silence is also the third case for memoizing by hand**: in a
  component the compiler skipped there is no build-time memoization to defer to, and nothing
  reports which components those are. A component with a conditional hook, or one the compiler
  otherwise could not verify, is on its own.

## Code splitting

Route-level splitting is automatic: the TanStack Router plugin runs with `autoCodeSplitting`,
which lifts each route's component into its own chunk. **That only works while a route file
exports nothing but `Route`** (see
[component-architecture.md](./component-architecture.md)), and `src/routes.test.ts` is what
keeps it true.

`vite.config.ts` additionally pins five vendor chunks by hand (`heroui`, `react`,
`tanstack-query`, `tanstack-router`, `recharts`), as `output.codeSplitting.groups` matched on
module id: Vite 8 bundles with Rolldown, which takes groups rather than Rollup's map of chunk
name to entry module. The comments there explain why the router is separate from React (an
evaluation-order bug at first paint), why recharts is on its own (it is ~368 kB raw, and only
the chart-bearing routes should pay for it), and why React's group is listed first. Read them
before changing the groups.

For anything heavy that is not a route, lazy-load it:

```tsx
const ShareDialog = lazy(() => import("./ShareDialog"))

{isSharing && (
  <Suspense fallback={null}>
    <ShareDialog onClose={close} />
  </Suspense>
)}
```

Two rules for the fallback: `null` for something the operator just opened (a warm chunk
resolves within a frame, and a spinner is a flash), and a fixed-height placeholder for
anything above the fold, so the swap does not shift the page.

A `React.lazy` chunk is the one sanctioned default export (see
[imports-and-modules.md](./imports-and-modules.md)).

## Mount modals when they open

```tsx
// Good: nothing in the DOM until it is needed
{isConfirming && <ConfirmDialog isOpen onClose={close} />}

// Bad: mounted always, hidden by a prop
<ConfirmDialog isOpen={isConfirming} onClose={close} />
```

`isOpen` still drives the open/close animation; what changes is whether the dialog's subtree,
its listeners, and its queries exist while it is closed.

## The server does the shaping

**The endpoint returns what the page needs and the dashboard renders it.** Filtering,
searching, sorting, joining and aggregating server data belong to the endpoint, not to the
browser.

Doing any of them here fails in one of three ways, depending on what was fetched. Filter the
page on screen and every match on another page is missed. Walk the collection first and the
answer is right, paid for with a full scan, and there is still no server-side count to put
under it. Walk a collection past the hundred-page cap and the tail is gone with nothing said,
so the answer is wrong and looks right. A small list already in memory, rendered in a table,
is fine.

**Nor does it assemble a view out of several responses.** Reading members, users, budgets and
ceilings to join them by id in the browser is four round trips and four whole tables to render
one page, and the joins are the server's work done with less information. One endpoint returns
the page's shape.

**If the endpoint you need does not exist, that is the change to make.** Not a walk, not a
client-side join, not a filter over everything. The gateway is in this repository.

**Reading a whole collection is not an exception to that, it is the thing it forbids.** A cap
on such a walk stops it looping forever against a backend that ignores `skip`; it does not make
the read paginated, and everything downstream still sorts and filters in the browser. A hook
that looks like it has to read everything is a hook whose endpoint does not offer what the page
needs, and that is a reason to change the endpoint.

What to reach for instead, by what the page is doing:

- **A table** takes `skip` and `limit` from the URL state, with
  `placeholderData: (previous) => previous` so the rows do not blank between pages. Every list
  route in the gateway already accepts both, capped at `limit=1000`, so this needs nothing
  from the backend.
- **A picker** needs the endpoint to search. Fetching every option to filter in memory offers
  a subset the moment the collection passes one page, and says nothing about it. No list route
  accepts a search term today except the two usage ones, so a picker over a large collection is
  a gateway change before it is a dashboard one.
- **A lookup** (a row carries a `user_id`, the page wants a name) belongs in the response.
  Either the row carries its own label or there is a batch endpoint to resolve the ids on the
  page. Reading the whole table to join it in the browser is the dashboard doing the server's
  join, and it pays for every row to answer for the handful on screen.
- **A page built from several reads** wants one endpoint shaped for it.
  `OrganizationMembersPage` currently takes seven and joins them by id; that is the case this
  rule is about, not an edge of it.

Nineteen reads in `shared/api/` predate this rule and still walk. #1376 tracks working them
down; `fetchAllPaged` and `fetchAllRows` in `shared/api/paging.ts` exist to be deleted, not to
be reached for.

## Long lists

Past a few hundred rows, paginate at the endpoint rather than rendering them. If a view ever
genuinely needs thousands of rows on screen at once, virtualization is the answer, and it is a
deliberate addition to discuss, not something to slip into a page.

## Reading layout is a synchronous flush

`getBoundingClientRect()`, `offsetWidth`, `scrollTop` and their siblings force the browser to
resolve pending layout before they can answer. One read is nothing. A read in a handler that
fires per pointer move, interleaved with a state write that dirties layout again, is a
read-write cycle per event for the length of a drag.

- **Measure once, outside the loop.** A value that cannot change during an interaction is read
  where the interaction starts and carried in the ref that already tracks it.
- **Batch reads, then write.** `LoginBackground.tsx:50` is the worked example: both
  `getBoundingClientRect()` calls sit together in one `measure()`, everything else runs off
  dirty flags, and a single `requestAnimationFrame` coalesces the lot. It is also the model for
  respecting `document.hidden` and `prefers-reduced-motion` in a render loop.
- **Prefer an observer to a listener.** `ResizeObserver` hands you `contentRect` without
  forcing anything; a `resize` listener that measures does force it, on every event.
- **`{ passive: true }` on `scroll`, `wheel` and `touchmove`**, unless the handler genuinely
  calls `preventDefault`. Without it the browser waits for the handler before it can scroll.
  `design-system/layout/TableScrollFrame.tsx:37` is the tree's only scroll listener and has it.

## Effects clean up after themselves

Every listener, interval, subscription, observer **and timer** that an effect creates is
removed in the function it returns, and that includes a timer started from an event handler or
a mutation callback rather than from an effect. Two failures, not one: a timer outliving its
component touches state or a DOM node that is gone, and a second trigger inside the window
stacks a timer rather than replacing it, so the first one's expiry cuts the second one short.
`design-system/actions/CopyField.tsx:215` names both and solves both with a ref that holds
the handle, cleared on the next trigger and on unmount. `AppShell` (the mobile media query)
and `useTheme` (the `prefers-color-scheme` query) are the worked examples in this tree,
including the Safari fallback for the deprecated listener API. A leak here is per-navigation,
so it compounds in a
dashboard people leave open all day.

## Watch the bundle when you add a dependency

`pnpm run build` prints every chunk. A new dependency that lands in the entry chunk, or a
route chunk that suddenly doubles, is a review finding. Import icons and utilities from their
subpaths rather than a package's barrel, for the same tree-shaking reason as
[imports-and-modules.md](./imports-and-modules.md).
