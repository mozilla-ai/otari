import type { ProviderHealth } from "@/client"
import { Dot } from "@/design-system/indicators/Dot"
import { formatRelative } from "@/shared/helpers/format"

// A provider's reachability, from the shared model-discovery health path, for
// the deployment providers and hosted providers pages alike. A provider that
// answers no model listing (its backend never implemented /v1/models) is not
// unreachable: only discovery is broken, and it may still serve requests, so it
// gets the amber warning state rather than the red one (issue #447).

// A repeated column's words, in their own casing. The uppercase these carried
// was emphasis applied to every row alike, including the reachable ones, which
// is the state a column watching for failures wants to draw the eye to least.
// Same treatment, and same reasoning, as Activity's status column.
const HEALTH_LABELS = {
  ok: "Reachable",
  degraded: "No model discovery",
  unreachable: "Unreachable",
} as const

export function HealthPill({ health }: { health: ProviderHealth | undefined }) {
  if (!health) {
    return <span className="text-caption">—</span>
  }
  const isDegraded = !health.ok && health.discovery_unsupported
  // Only a real failure colors its text. Degraded is a provider that answers
  // requests but lists no models, which is a fact about discovery rather than an
  // outage, so it reads on the muted rung with the danger dot that says "worth
  // noticing" without the ink that says "broken".
  const styles = health.ok || isDegraded ? "text-muted" : "text-danger"
  const dot = health.ok ? "bg-success" : "bg-danger"
  // The last-checked time lives in the top summary banner; the row just shows the
  // status. The error (and time) stay available on hover as the pill's tooltip.
  const checked = health.checked_at
    ? `Last checked ${formatRelative(health.checked_at)}`
    : "Not checked yet"
  const reason = isDegraded
    ? `${health.error ?? "This provider does not list models."} Requests to it may still work.`
    : (health.error ?? "Unreachable")
  const title = health.ok ? checked : `${reason} · ${checked}`
  return (
    <span
      title={title}
      className={`flex items-center gap-2 text-mono-caption ${styles}`}
    >
      <Dot className={dot} />
      {
        HEALTH_LABELS[
          health.ok ? "ok" : isDegraded ? "degraded" : "unreachable"
        ]
      }
    </span>
  )
}
