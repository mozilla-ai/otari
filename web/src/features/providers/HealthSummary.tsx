import { Button } from "@/design-system/actions/Button"
import { Dot } from "@/design-system/indicators/Dot"
import { Section } from "@/design-system/layout/Section"
import { useRecheckProviderHealth } from "@/shared/api/providers"
import { formatRelative } from "@/shared/helpers/format"

// A one-line "N of M providers reachable" summary with a live re-check, above the
// table. The healthy/degraded/total counts come precomputed from the gateway, and
// the same counts feed the overview page's summary tile (issue #302). `degraded`
// providers are not reachable-by-discovery but are not failures either, so they
// are called out separately and keep the dot amber rather than red.
export function HealthSummary({
  healthy,
  degraded,
  total,
  checkedAt,
}: {
  healthy: number
  degraded: number
  total: number
  /** Empty before the first check. */
  checkedAt: string
}) {
  const allHealthy = healthy === total
  const dot = allHealthy
    ? "bg-success"
    : healthy + degraded === total
      ? "bg-warning"
      : "bg-danger"
  const recheck = useRecheckProviderHealth()
  return (
    // A band of the page rather than a card, which is what the rest of this page
    // became: the summary is a region between rules, so the rules are what bound
    // it and the fill goes.
    <Section
      className="border-y border-border py-3"
      contentClassName="flex flex-wrap items-center gap-3 text-sm"
    >
      <Dot className={dot} />
      <span className="font-medium text-foreground">
        {healthy} of {total} provider{total === 1 ? "" : "s"} reachable
      </span>
      {degraded > 0 ? (
        <span className="text-warning">{degraded} without model discovery</span>
      ) : null}
      {checkedAt ? (
        <span className="text-muted">
          Last checked {formatRelative(checkedAt)}
        </span>
      ) : null}
      <Button
        size="sm"
        variant="ghost"
        className="ml-auto"
        isDisabled={recheck.isPending}
        onPress={() => recheck.mutate()}
      >
        {recheck.isPending ? "Re-checking…" : "Re-check all"}
      </Button>
    </Section>
  )
}
