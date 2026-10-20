import type { Query, QueryClient } from "@tanstack/react-query"
import { useQueryClient } from "@tanstack/react-query"
import { useEffect, useState, useSyncExternalStore } from "react"
import { FiAlertTriangle } from "react-icons/fi"

import { ApiError } from "@/shared/api/client"

// apiFetch normalizes an unreachable gateway to ApiError status 0 ("Network
// error: could not reach the gateway."). A 401/403 is a different failure: the
// backend answered, it just rejected the key, and that already bounces to
// sign-in, so it is not "can't connect".
function isUnreachable(error: unknown): boolean {
  return error instanceof ApiError && error.status === 0
}

function failedToConnect(query: Query): boolean {
  return query.state.status === "error" && isUnreachable(query.state.error)
}

// Whether the most recent request anywhere in the cache failed to connect. Kept
// from the cache's events rather than read off each query's state: any answer
// from the gateway, an error included, clears it, so an errored query left
// behind by a page the operator moved on from cannot hold the alert, and a
// query refetching (which puts one with no data back to `pending`) does not
// flip it on its way to the same failure. `error` fires once retries are spent.
function trackConnection(queryClient: QueryClient) {
  let failing = false
  return {
    subscribe: (onStoreChange: () => void) =>
      queryClient.getQueryCache().subscribe((event) => {
        if (event.type !== "updated") return
        const { action } = event
        // A manual cache write (`setQueryData`) reaches no gateway, so it says
        // nothing about the connection.
        if (action.type === "success" && !action.manual) failing = false
        else if (action.type === "error") failing = isUnreachable(action.error)
        else return
        onStoreChange()
      }),
    getSnapshot: () => failing,
  }
}

// It watches the whole query cache rather than any one page, so the alert is
// the same wherever the operator is standing.
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
  // Built once per client: `useSyncExternalStore` identity-checks `subscribe`
  // and re-subscribes when it changes, and the tracker holds the state.
  const [connection] = useState(() => trackConnection(queryClient))
  return useSyncExternalStore(connection.subscribe, connection.getSnapshot)
}

// A bottom-right toast that surfaces a lost backend connection at the app level,
// instead of leaving each page to render its own inline error. The gateway not
// answering is a whole-app condition, so it belongs above any single page. Not
// dismissible: it is tied to live state and disappears on its own once the
// gateway responds.
//
// Shown only once the failing queries have been asked again and failed again.
// One dropped request is routine (a laptop waking, a proxy closing an idle
// connection) and the build poll does not retry, so alarming on the first
// failure put the banner up over a healthy gateway.
export function ConnectionStatus() {
  const queryClient = useQueryClient()
  const failing = useGatewayUnreachable()
  const [confirmed, setConfirmed] = useState(false)

  useEffect(() => {
    if (!failing) {
      setConfirmed(false)
      return
    }
    let current = true
    // A recheck that succeeds clears `failing`, which ends this run before the
    // banner is confirmed.
    void queryClient.refetchQueries({ predicate: failedToConnect }).then(() => {
      if (current) setConfirmed(true)
    })
    return () => {
      current = false
    }
  }, [failing, queryClient])

  if (!failing || !confirmed) {
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
