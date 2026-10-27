import { Skeleton } from "@/design-system/feedback/Skeleton"
import { Chip } from "@/design-system/indicators/Chip"
import { SpendMeter } from "@/design-system/metrics/SpendMeter"
import { canManage } from "@/features/organization/roles"
import { useApiKeyCeiling, useApiKeyCeilings } from "@/shared/api/budgets"
import { useOrganizationContext } from "@/shared/api/organizations"
import { formatPct, formatUsd } from "@/shared/helpers/format"
import { useIsDeploymentScope } from "@/shared/hooks/DeploymentScope"

import { CostByTool } from "./CostByTool"
import { formatRelativeTime } from "./format"
import { SessionAtAGlance } from "./SessionAtAGlance"
import type { Trace } from "./types"

type Gateway = NonNullable<Trace["gateway"]>

/**
 * The key's own ceiling over all of its spend, from the surface of the page
 * this renders on: the deployment's on a deployment page, the organization's
 * spend ceilings otherwise. The organization's list is for its owners and
 * admins, so a member is not asked and sees no ceiling sentence.
 */
function useKeyCeiling(apiKeyId: string) {
  const isDeploymentWide = useIsDeploymentScope()
  const managesOrganization = canManage(useOrganizationContext().data)
  const deployment = useApiKeyCeiling(apiKeyId, isDeploymentWide)
  const organization = useApiKeyCeilings(
    "organization",
    !isDeploymentWide && managesOrganization && apiKeyId !== "",
  )
  if (isDeploymentWide) return deployment
  return {
    data: organization.data?.find((row) => row.scope_id === apiKeyId),
    isSuccess: organization.isSuccess,
    isLoading: organization.isLoading,
  }
}

function Spend({
  cost,
  apiKeyId,
  isAtApiPrices,
}: {
  cost: number
  apiKeyId: string
  isAtApiPrices: boolean
}) {
  const ceiling = useKeyCeiling(apiKeyId)
  const limit = ceiling.data?.max_budget ?? null
  // At a glance states the total; this says what it means for the key.
  const sentences = [
    isAtApiPrices
      ? "The cost above is at API prices: the agent reported its requests rather than sending them through Otari, so on a subscription plan it is what they are worth, not what was charged."
      : null,
    ceiling.data && limit !== null
      ? `This session is ${formatPct(limit > 0 ? cost / limit : 0)} of this key's ${formatUsd(limit)} ceiling, which has ${formatUsd(ceiling.data.current_spend)} spent this period.`
      : ceiling.isSuccess && !ceiling.data
        ? "This key has no spend ceiling of its own."
        : null,
  ].filter((sentence) => sentence !== null)
  // The ceiling's sentence is held as a line while it loads, so the section
  // does not appear only once the read lands.
  if (sentences.length === 0 && !ceiling.isLoading) return null
  return (
    <section className="flex flex-col gap-1.5">
      <h3 className="text-overline">Spend</h3>
      {sentences.length > 0 ? (
        <p className="text-body">{sentences.join(" ")}</p>
      ) : (
        <Skeleton className="h-5 w-3/4" />
      )}
      {ceiling.data && limit !== null ? (
        <SpendMeter
          spent={ceiling.data.current_spend}
          allocated={limit}
          ariaLabel="This key's spend against its ceiling"
          className="max-w-80"
        />
      ) : null}
    </section>
  )
}

// Where the gateway stepped in, each row opening the span it is about. This is
// what the panel shows before a span is picked: the API stores no request or
// response content, so there is no trace-level input or output to show instead.
export function GatewayPanel({
  gateway,
  startedAt,
  onSelect,
}: {
  gateway: Gateway
  startedAt: string
  onSelect: (spanId: string) => void
}) {
  const start = Date.parse(startedAt)
  return (
    <>
      <SessionAtAGlance stats={gateway.stats} cost={gateway.cost} />
      <Spend
        cost={gateway.cost}
        apiKeyId={gateway.apiKeyId}
        isAtApiPrices={gateway.isAtApiPrices}
      />
      {gateway.toolCosts ? (
        <CostByTool
          breakdown={gateway.toolCosts}
          total={gateway.cost}
          onSelect={onSelect}
        />
      ) : null}
      <section className="flex flex-col gap-1.5">
        <h3 className="text-overline">Where the gateway stepped in</h3>
        {gateway.events.length === 0 ? (
          <p className="text-caption text-subtle">
            No guardrail flags, fallbacks or model switches in this session.
          </p>
        ) : (
          <ul className="flex flex-col border-t border-border">
            {gateway.events.map((event) => (
              <li
                key={`${event.spanId}-${event.word}`}
                className="border-b border-border"
              >
                <button
                  type="button"
                  onClick={() => onSelect(event.spanId)}
                  className="flex w-full items-start gap-3 px-2 py-2 text-left outline-none hover:bg-surface-alt focus-visible:otari-focus-ring"
                >
                  <span className="flex w-24 shrink-0 pt-0.5">
                    <Chip tone={event.tone}>{event.word}</Chip>
                  </span>
                  <span className="flex min-w-0 flex-1 flex-col gap-0.5">
                    <span className="break-words text-body">{event.title}</span>
                    <span className="text-caption">{event.detail}</span>
                  </span>
                  {event.at ? (
                    <span className="shrink-0 pt-0.5 text-mono-caption text-subtle">
                      {formatRelativeTime(
                        Math.max(0, Date.parse(event.at) - start),
                      )}
                    </span>
                  ) : null}
                </button>
              </li>
            ))}
          </ul>
        )}
      </section>
    </>
  )
}
