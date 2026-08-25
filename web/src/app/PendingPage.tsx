/**
 * The wait a route renders while something it needs is still on its way.
 *
 * One declaration, because there are two of those waits and they are the same
 * wait to the person looking at them: the router shows it while a page's chunk
 * loads (`defaultPendingComponent` in `app/router.tsx`), and the shell shows it
 * while the entitlement axis is still resolving and the route would otherwise be
 * answered with a panel asserting the deployment does not serve the page
 * (`app/AppShell.tsx`). Written out in both places they drifted into two
 * different waits with nothing to notice it.
 *
 * Its own module rather than an export from `router.tsx`, which is where the
 * first of the two lives: `AppShell` is what `routes/__root.tsx` names, and
 * `router.tsx` reaches that file through `routeTree.gen.ts`, so importing the
 * router here would close that circle. Nothing imports this one.
 */
export function PendingPage() {
  return <div role="status">Loading page…</div>
}
