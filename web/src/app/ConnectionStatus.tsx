import { useQueryClient } from "@tanstack/react-query"
import { useSyncExternalStore } from "react"

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
function useGatewayUnreachable(): boolean {
  const cache = useQueryClient().getQueryCache()

  // Read through `useSyncExternalStore` rather than by writing a subscription
  // into state, and that is load-bearing rather than tidying: mounting a
  // component that holds a query builds that query into this cache *during
  // render*, and the cache notifies its subscribers synchronously. A `setState`
  // in that callback is therefore an update to this component while a different
  // one is rendering, which React warns about. Swapping the sidebar's head
  // (leaving the mobile drawer's organization submenu remounts the workspace
  // switcher, which holds two queries) is one way to reach it. This is the API
  // for an external store that can change mid-render.
  return useSyncExternalStore(
    (onStoreChange) => cache.subscribe(onStoreChange),
    () =>
      cache
        .getAll()
        .some(
          (query) =>
            query.state.status === "error" && isUnreachable(query.state.error),
        ),
  )
}

// A bottom-right toast that surfaces a lost backend connection at the app level,
// instead of leaving each page to render its own inline error. The gateway not
// answering is a whole-app condition, so it belongs above any single page. Not
// dismissible: it is tied to live state and disappears on its own once the
// gateway responds.
export function ConnectionStatus() {
  const unreachable = useGatewayUnreachable()
  if (!unreachable) {
    return null
  }

  return (
    <div
      role="alert"
      aria-live="assertive"
      className="fixed right-4 bottom-4 z-50 flex max-w-sm items-start gap-2.5 rounded-lg border border-danger bg-danger-subtle px-4 py-3 text-sm text-danger shadow-lg"
    >
      <svg
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        strokeWidth="2"
        aria-hidden="true"
        className="mt-0.5 h-5 w-5 shrink-0"
      >
        <path
          d="M12 9v4M12 17h.01"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
        <path
          d="M10.3 3.9 1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0z"
          strokeLinejoin="round"
        />
      </svg>
      <span>
        <strong className="font-semibold">Can’t reach the gateway.</strong> The
        backend isn’t responding; data won’t load or save until the connection
        is restored.
      </span>
    </div>
  )
}
