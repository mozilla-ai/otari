import type { ReactNode } from "react"

import { ErrorBanner } from "./ErrorBanner"

/**
 * A failure that took the whole page rather than a band inside one.
 *
 * `PageLoading`'s counterpart, and for the same reason it exists: three places
 * reach this state (a gateway that never answered, a throw above the router,
 * and a throw inside it) and none of them should be drawing its own centered
 * box. `children` is the sentence about what to do next, which the gateway case
 * puts in the banner itself and the two throwing cases cannot.
 */
export function PageError({
  error,
  children,
}: {
  error: unknown
  children?: ReactNode
}) {
  return (
    <div className="flex min-h-full items-center justify-center p-6">
      <div className="flex w-full max-w-md flex-col gap-3">
        <ErrorBanner error={error} />
        {children ? <p className="text-caption">{children}</p> : null}
      </div>
    </div>
  )
}
