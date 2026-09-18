import { useQueryClient } from "@tanstack/react-query"
import { useCallback, useSyncExternalStore } from "react"
import { FiAlertTriangle } from "react-icons/fi"

import { ApiError } from "@/shared/api/client"

// apiFetch normalizes an unreachable gateway to ApiError status 0 ("Network
// error: could not reach the gateway."). A 401/403 is a different failure: the
// backend answered, it just rejected the key, and that already bounces to
// sign-in, so it is not "can't connect".
function isUnreachable(error: unknown): boolean {
  return error instanceof ApiError && error.status === 0
}

// True while at least one query is currently failing to reach the gateway. It
// watches the whole query cache rather than any one page, so the alert is the
// same wherever the operator is standing, and it clears itself the moment a
// request succeeds again.
//
// The cache is an external store and is read as one. Subscribing in an effect
// and calling `setState` from the listener does not hold here: the cache emits
// synchronously when an observer is created, and an observer is created when a
// component calls `useQuery` during its render, so the listener runs inside
// another component's render pass. React calls that a bad `setState` in render.
// `useSyncExternalStore` is the primitive for this shape, and React owns the
// timing, so the same notification arrives safely.
function useGatewayUnreachable(): boolean {
  const queryClient = useQueryClient()

  // `useSyncExternalStore` identity-checks `subscribe` and re-subscribes when it
  // changes, so this is held stable rather than rebuilt every render. That is
  // the "a reference something else identity-checks" case performance.md keeps,
  // not memoization by reflex.
  const subscribe = useCallback(
    (onStoreChange: () => void) =>
      queryClient.getQueryCache().subscribe(onStoreChange),
    [queryClient],
  )

  // Runs on every notification and more than once per render, so it stays a
  // scan that returns a boolean. React compares snapshots with `Object.is`, so
  // returning anything carrying an identity of its own would loop.
  const getSnapshot = useCallback(
    () =>
      queryClient
        .getQueryCache()
        .getAll()
        .some(
          (query) =>
            query.state.status === "error" && isUnreachable(query.state.error),
        ),
    [queryClient],
  )

  return useSyncExternalStore(subscribe, getSnapshot)
}

// A bottom-right toast that surfaces a lost backend connection at the app level,
// instead of leaving each page to render its own inline error. The gateway not
// answering is a whole-app condition, so it belongs above any single page. Not
// dismissible: it is tied to live state and disappears on its own once the
// gateway responds.
export function ConnectionStatus() {
  const isUnreachable = useGatewayUnreachable()
  if (!isUnreachable) {
    return null
  }

  return (
    <div
      role="alert"
      aria-live="assertive"
      className="fixed right-4 bottom-4 z-50 flex max-w-sm items-start gap-2.5 rounded-lg border border-danger bg-danger-subtle px-4 py-3 text-sm text-danger shadow-elevation-lg"
    >
      <FiAlertTriangle aria-hidden="true" className="mt-0.5 h-5 w-5 shrink-0" />
      <span>
        <strong className="font-semibold">Can’t reach the gateway.</strong> The
        backend isn’t responding; data won’t load or save until the connection
        is restored.
      </span>
    </div>
  )
}
