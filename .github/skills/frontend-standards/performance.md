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

- **Do not add `useMemo`, `useCallback`, or `React.memo` by reflex.** The compiler already
  does the ordinary cases, and hand-memoization it cannot see through is how a stale value
  survives a prop change.
- Reach for them when you can point at the reason: a profiler measurement, or a reference the
  compiler cannot prove stable that you are passing into a memoized third-party component.
- **Correct dependency arrays still matter**, because `useEffect` is not memoization. The
  compiler does not fix an effect that re-subscribes on every render or one that misses a
  dependency it reads.
- Rules of hooks still apply, and now they are load-bearing: the compiler bails out of a
  component it cannot prove follows them, silently, so a conditional hook costs optimization
  as well as correctness.

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

Filtering, sorting, searching, and pagination of server data belong to the endpoint. Pulling
ten thousand rows to `.filter()` them in the browser is both slow and wrong: it filters only
the page that was fetched, so the result is a subset of a subset and the count is a lie. A
small list already in memory, rendered in a table, is fine.

When a hook genuinely has to walk everything, bound the walk. `fetchAllPricing` in
`shared/api/hooks.ts` is the shape to copy, and [data-fetching.md](./data-fetching.md)
explains the cap.

## Long lists

Past a few hundred rows, paginate at the endpoint rather than rendering them. If a view ever
genuinely needs thousands of rows on screen at once, virtualization is the answer, and it is a
deliberate addition to discuss, not something to slip into a page.

## Effects clean up after themselves

Every listener, interval, subscription, and observer that an effect creates is removed in the
function it returns. `AppShell` (the mobile media query) and `useTheme` (the
`prefers-color-scheme` query) are the worked examples in this tree, including the Safari
fallback for the deprecated listener API. A leak here is per-navigation, so it compounds in a
dashboard people leave open all day.

## Watch the bundle when you add a dependency

`pnpm run build` prints every chunk. A new dependency that lands in the entry chunk, or a
route chunk that suddenly doubles, is a review finding. Import icons and utilities from their
subpaths rather than a package's barrel, for the same tree-shaking reason as
[imports-and-modules.md](./imports-and-modules.md).
