import { EmptyState } from "@/design-system/feedback/EmptyState"
/**
 * A destination this build declares but does not serve.
 *
 * The shell already answers a *gated-off* registered path with its own panel, so
 * in a standalone gateway this never renders: the surface is absent, the sidebar
 * drops the link, and the shell intercepts the route. It is what a deployment
 * that reports the surface but has not composed the overlay serving it would
 * see, which is the one case that would otherwise paint a blank page.
 */
export function UnavailableHere({ title }: { title: string }) {
  return (
    <EmptyState
      title={`${title} is not available here`}
      description="This deployment declares the page but does not serve it. Pick a destination from the sidebar."
    />
  )
}
