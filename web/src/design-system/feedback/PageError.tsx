import type { ReactNode } from "react"

import { ErrorBanner } from "./ErrorBanner"

/**
 * A failure that took the whole page rather than a band inside one.
 *
 * `PageLoading`'s counterpart, and for the same reason it exists: three places
 * reach this state (a gateway that never answered, a throw above the router,
 * and a throw inside it) and none of them should be drawing its own centered
 * box. `children` is the sentence about what to do next, and `action` the one
 * control that does it, where the page has one to offer (the gateway case's
 * Retry; a throw has none).
 */

// `ErrorBanner` renders nothing at all for a falsy value, and both catch
// boundaries hand over whatever was thrown: `throw ""` and `throw 0` are as
// legal as `throw null`, and none of them carries a message to show. Substituted
// here rather than at each caller, so neither can reach the panel with an empty
// banner in it.
const UNTYPED_FAILURE = new Error("Something went wrong.")

export function PageError({
  error,
  children,
  action,
}: {
  error: unknown
  children?: ReactNode
  action?: ReactNode
}) {
  return (
    <div className="flex min-h-full items-center justify-center p-6">
      <div className="flex w-full max-w-md flex-col gap-3">
        <ErrorBanner error={error || UNTYPED_FAILURE} />
        {children ? <p className="text-caption">{children}</p> : null}
        {action ? <div className="flex">{action}</div> : null}
      </div>
    </div>
  )
}
