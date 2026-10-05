import { FiAlertTriangle } from "react-icons/fi"

import { isGatewayUnreachable } from "@/shared/api/client"
import { useGatewayLiveness } from "@/shared/api/deployment"

// A bottom-right toast that surfaces a lost backend connection at the app level,
// instead of leaving each page to render its own inline error. The gateway not
// answering is a whole-app condition, so it belongs above any single page. Not
// dismissible: it is tied to live state and disappears on its own once the
// gateway responds.
export function ConnectionStatus() {
  const { isError, error } = useGatewayLiveness()
  // Missing or invalid liveness replies confirm an outage. Genuine HTTP
  // refusals mean the gateway answered, even if the request was unsuccessful.
  const isUnreachable = isError && isGatewayUnreachable(error)
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
