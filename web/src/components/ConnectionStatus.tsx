import { useQueryClient } from "@tanstack/react-query";
import { useEffect, useState } from "react";

import { ApiError } from "@/api/client";

// apiFetch normalizes an unreachable gateway to ApiError status 0 ("Network
// error: could not reach the gateway."). A 401/403 is a different failure: the
// backend answered, it just rejected the key, and that already bounces to
// sign-in, so it is not "can't connect".
function isUnreachable(error: unknown): boolean {
  return error instanceof ApiError && error.status === 0;
}

// True while at least one query is currently failing to reach the gateway. It
// watches the whole query cache rather than any one page, so the alert is the
// same wherever the operator is standing, and it clears itself the moment a
// request succeeds again.
function useGatewayUnreachable(): boolean {
  const queryClient = useQueryClient();
  const [unreachable, setUnreachable] = useState(false);

  useEffect(() => {
    const cache = queryClient.getQueryCache();
    const compute = () =>
      cache.getAll().some((query) => query.state.status === "error" && isUnreachable(query.state.error));
    setUnreachable(compute());
    return cache.subscribe(() => setUnreachable(compute()));
  }, [queryClient]);

  return unreachable;
}

// A bottom-right toast that surfaces a lost backend connection at the app level,
// instead of leaving each page to render its own inline error. The gateway not
// answering is a whole-app condition, so it belongs above any single page. Not
// dismissible: it is tied to live state and disappears on its own once the
// gateway responds.
export function ConnectionStatus() {
  const unreachable = useGatewayUnreachable();
  if (!unreachable) {
    return null;
  }

  return (
    <div
      role="alert"
      aria-live="assertive"
      className="fixed right-4 bottom-4 z-50 flex max-w-sm items-start gap-2.5 rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700 shadow-lg"
    >
      <svg
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        strokeWidth="2"
        aria-hidden
        className="mt-0.5 h-5 w-5 shrink-0"
      >
        <path d="M12 9v4M12 17h.01" strokeLinecap="round" strokeLinejoin="round" />
        <path
          d="M10.3 3.9 1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0z"
          strokeLinejoin="round"
        />
      </svg>
      <span>
        <strong className="font-semibold">Can’t reach the gateway.</strong> The backend isn’t responding; data won’t
        load or save until the connection is restored.
      </span>
    </div>
  );
}
