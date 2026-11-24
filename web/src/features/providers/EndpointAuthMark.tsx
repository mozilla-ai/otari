import type { EndpointAuth } from "@/client"
import { formatDate } from "@/shared/helpers/format"

/**
 * What a key's endpoint credential says about itself, where it has something
 * to say.
 *
 * A credential still inside its window draws nothing. The date is a declaration
 * the organization made, and no value of it refuses a request, so a mark on
 * every key carrying one would be noise on the rows that need no attention.
 *
 * One this deployment cannot read is the exception, and it outranks the date: a
 * credential nobody can send has no window left to be inside, and the row is
 * otherwise indistinguishable from a key that carries none.
 */
export function EndpointAuthMark({
  auth,
  isUnreadable,
}: {
  auth: EndpointAuth | null | undefined
  /** Whether the key stores a credential this deployment cannot read. */
  isUnreadable: boolean
}) {
  if (isUnreadable)
    return <span className="text-danger">Endpoint credential unreadable</span>
  if (!auth || auth.state === "valid") return null
  const isExpired = auth.state === "expired"
  return (
    <span className={isExpired ? "text-danger" : "text-warning"}>
      {isExpired ? "Endpoint credential expired " : "Endpoint credential due "}
      {formatDate(auth.expires_at)}
    </span>
  )
}
