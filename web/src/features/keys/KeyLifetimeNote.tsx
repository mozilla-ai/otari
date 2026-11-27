import { formatDateTimeMinutes } from "@/shared/helpers/format"

/**
 * Where a clamped key write landed, beside the place that asked for it.
 *
 * The gateway sends `Otari-Key-Lifetime-Capped` only when it moved an expiry to
 * the deployment's maximum key lifetime, so `cappedTo` is "" whenever there is
 * nothing to say.
 */
export function KeyLifetimeNote({ cappedTo }: { cappedTo: string }) {
  if (cappedTo === "") return null
  return (
    <p role="status" className="text-caption">
      This deployment caps key lifetime, so the expiry is{" "}
      {formatDateTimeMinutes(cappedTo)}.
    </p>
  )
}
