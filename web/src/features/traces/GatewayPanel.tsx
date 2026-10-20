import { Chip } from "@/design-system/indicators/Chip"
import { SpendMeter } from "@/design-system/metrics/SpendMeter"
import { useApiKeyCeiling } from "@/shared/api/budgets"
import { useDeploymentOperator } from "@/shared/api/organizations"
import { formatCost, formatPct, formatUsd } from "@/shared/helpers/format"

import { formatRelativeTime } from "./format"
import type { Trace } from "./types"

type Gateway = NonNullable<Trace["gateway"]>

function Spend({ cost, apiKeyId }: { cost: number; apiKeyId: string }) {
  // The ceilings read is deployment-wide, so only an operator asks for it.
  const { isOperator } = useDeploymentOperator()
  const ceiling = useApiKeyCeiling(apiKeyId, isOperator)
  const limit = ceiling.data?.max_budget ?? null
  return (
    <section className="flex flex-col gap-1.5">
      <h3 className="text-overline">Spend</h3>
      <p className="text-body">
        This session cost {formatCost(cost)}.
        {ceiling.data && limit !== null
          ? ` That is ${formatPct(limit > 0 ? cost / limit : 0)} of this key's ${formatUsd(limit)} ceiling, which has ${formatUsd(ceiling.data.current_spend)} spent this period.`
          : ceiling.isSuccess && !ceiling.data
            ? " This key has no spend ceiling of its own."
            : null}
      </p>
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
      <Spend cost={gateway.cost} apiKeyId={gateway.apiKeyId} />
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
