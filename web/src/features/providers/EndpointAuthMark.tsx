import type { EndpointAuth } from "@/client"
import { formatDate } from "@/shared/helpers/format"

/**
 * What a key's endpoint credential says about its own rotation date, where it
 * has something to say.
 *
 * A credential still inside its window draws nothing. The date is a declaration
 * the organization made, and no value of it refuses a request, so a mark on
 * every key carrying one would be noise on the rows that need no attention.
 */
export function EndpointAuthMark({
  auth,
}: {
  auth: EndpointAuth | null | undefined
}) {
  if (!auth || auth.state === "valid") return null
  const isExpired = auth.state === "expired"
  return (
    <span className={isExpired ? "text-danger" : "text-warning"}>
      {isExpired ? "Endpoint credential expired " : "Endpoint credential due "}
      {formatDate(auth.expires_at)}
    </span>
  )
}
