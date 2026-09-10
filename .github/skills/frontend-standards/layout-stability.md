# Layout stability: no flash, no shift

An operator watches this dashboard while a gateway is doing something they care about. Every
flash of a skeleton over data that was already on screen, and every row that jumps as a query
settles, costs them their place. Most of it comes from three habits, and all three are cheap
to avoid.

## Query guards: `&& !data`

TanStack Query's `isPending` means "no data in this cache entry yet", and it is `true` again
whenever the key changes, including a key that only changed its filters. Gate on the data, not
on the flag alone:

```tsx
const { data, isPending, isError } = useUsage(filters)

// Good: cached data keeps rendering while the next page loads
if (isPending && !data) return <Skeleton />
if (isError && !data) return <ErrorBanner error={error} />
return <UsageTable rows={data} />

// Bad: a filter change blanks the page to a skeleton and back
if (isPending) return <Skeleton />
```

`isLoading` is `isPending && isFetching`, which is `false` for a disabled query and for one
restoring from cache, so a guard written on it flashes an empty or error state at exactly the
moments a cache should have prevented one. Use `isPending && !data`.

## Keep the previous page during a refetch

For any query whose key carries filters, pagination, or a search term, hold the last result
while the next one loads:

```tsx
useQuery({
  queryKey: [USAGE, filters],
  queryFn: () => apiFetch<UsageResponse>(usagePath(filters)),
  placeholderData: (previous) => previous,
})
```

Without it, every keystroke in a filter empties the table. The dashboard already does this in
a few hooks under `web/src/shared/api/`; a new filtered query should not be the exception.

## Skeletons match what they replace

A skeleton is a promise about the size of the thing arriving. If it is shorter than the
content, everything below it jumps on swap, which is worse than no skeleton at all. Measure
the loaded component and match its height, and prefer HeroUI's `Skeleton` over a hand-rolled
pulse so the motion-safe handling comes with it.

## Theme is applied before the first paint

`web/index.html` carries an inline script that reads `otari.dashboard.theme` from
localStorage, resolves `system` against `prefers-color-scheme`, and sets `data-theme`, the
`dark` class, and `color-scheme` on `<html>` before the bundle loads. `useTheme`
(`src/shared/hooks/useTheme.tsx`) sets the same three properties once React has mounted.

**The duplication is deliberate and the two must stay in step.** Moving the initial resolution
into the effect alone would paint the boot screen light for every operator who chose dark. If
you change the storage key, the three-state resolution, or which attributes carry the theme,
change both, and `useTheme.test.tsx` plus the screenshot matrix's dark projects are what
notice when you do not.

The boot markup in `index.html` is the one sanctioned `<style>` block in the app, for the same
reason: it paints the loading state before the bundle's stylesheet is a certainty. It reads
`var(--color-background, #ffffff)` so it tracks the token where the stylesheet is already
loaded and falls back where it is not.

## Chrome lives in the shell, not in a route

`AppShell` renders the sidebar, header, and breadcrumbs around the router outlet. Putting any
of that inside a route component remounts it on every navigation, which shifts the layout and
throws away the drawer's open state. A route renders its page and nothing around it.

## Reload is not a state-recovery tool

`window.location.reload()` in response to a stale-looking UI throws away every cache the
operator has warmed and re-runs the bootstrap round trip. Invalidate the affected query keys
instead. The one legitimate reload is at the root error boundary, where the app cannot know
what state it is in. `UpdatePrompt` is a separate case: it asks first, because it is telling
the operator a new build exists rather than recovering from an error.

## Anti-patterns

- A bare `isPending` or `isError` loading guard.
- A filtered or paginated query with no `placeholderData`.
- A skeleton whose height is not the content's height.
- Theme, or any other pre-paint decision, resolved only in a React effect.
- Global chrome mounted inside a route component.
- `window.location.reload()` as a refresh button.
