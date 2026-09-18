import { Button } from "@heroui/react"

import type { UsageEntry } from "@/client"
import { isTokenChargeLine, isUnitChargeLine } from "@/client"
import { useMemberAttributionLabels } from "@/features/organization/attribution"
import { userDisplay } from "@/features/users/userDisplay"
import { formatUnitRate } from "@/shared/helpers/format"
import { providerDisplayName } from "@/shared/helpers/providers"
import {
  buildTokenComposition,
  computeToolCost,
  describeSource,
  findPricingSelector,
  formatLatencyCell,
  formatTokenCount,
  formatToolUsage,
  formatUSD,
  listToolUsage,
  sortChargeLines,
} from "./activityModel"
import { DetailField } from "./DetailField"
import { RoutingPlan } from "./RoutingPlan"

// The detail panel for one request: the failure diagnostic plus the metadata
// that does not fit the row. The dashboard is master-key admin-only, so the
// stored `error_message` is shown verbatim; it is source-neutral by nature,
// carrying either a fixed gateway rejection string (e.g. a model with no pricing
// under `require_pricing`) or the raw upstream provider error, so the heading
// stays "Error" rather than blaming the provider for every failure.
export function RequestDetail({
  entry,
  onPriceModel,
}: {
  entry: UsageEntry
  /**
   * Null for a caller who does not operate the deployment. Pricing a model is a
   * deployment-wide write (`/pricing`), so offering the button to a tenant
   * would be offering a refusal; the sentence beside it is still theirs to read,
   * because "this row cost nothing" is a fact about their own request.
   */
  onPriceModel: ((model: string) => void) | null
}) {
  const memberLabels = useMemberAttributionLabels()
  // A row with no cost (cost IS NULL, the same test the "Priced?" filter uses)
  // is either a model the gateway has no price for or a request refused before
  // it could be billed. Both are the same fix, and the row holds what the
  // selector was, which a provider without model discovery would never have put
  // in the catalog. A $0 cost is a real price, so it is deliberately not
  // treated as uncosted.
  const isUncosted = entry.cost === null
  const pricingKey = findPricingSelector(entry)
  return (
    <div className="flex flex-col gap-4 px-4 py-4">
      {entry.error_message ? (
        <div className="flex flex-col gap-1.5">
          <span className="text-overline">
            Error{entry.status_code !== null ? ` (${entry.status_code})` : ""}
          </span>
          <pre className="max-h-48 overflow-auto rounded-lg border border-danger bg-danger-subtle p-3 text-xs whitespace-pre-wrap break-all text-danger">
            {entry.error_message}
          </pre>
        </div>
      ) : null}
      {/* Routed requests only. Placed above the metadata grid, and above the
          per-row fields, because on a failed attempt it answers the first question
          the failure raises: what served the request instead. */}
      {entry.policy_name !== null && entry.policy_name !== undefined ? (
        <RoutingPlan entry={entry} />
      ) : null}
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <DetailField label="Provider">
          {entry.provider ? providerDisplayName(entry.provider) : "—"}
        </DetailField>
        <DetailField label="Endpoint">{entry.endpoint}</DetailField>
        <DetailField label="Source">{describeSource(entry.source)}</DetailField>
        {entry.source_label ? (
          <DetailField label="Session">{entry.source_label}</DetailField>
        ) : null}
        {/* The name reads, the id stays copyable: this is the one place an
            operator goes for the raw id, so naming the person here must not
            take it away. */}
        <DetailField label="User" copyValue={entry.user_id} copyLabel="user id">
          {entry.user_id === null
            ? "—"
            : userDisplay(entry.user_id, entry.user_alias, memberLabels).label}
        </DetailField>
        <DetailField
          label="API key"
          copyValue={entry.api_key_id}
          copyLabel="api key id"
        >
          {entry.api_key_id ?? "—"}
        </DetailField>
        <DetailField label="Prompt tokens">
          {formatTokenCount(entry.prompt_tokens)}
        </DetailField>
        <DetailField label="Completion tokens">
          {formatTokenCount(entry.completion_tokens)}
        </DetailField>
        <DetailField label="Total tokens">
          {formatTokenCount(entry.total_tokens)}
        </DetailField>
        {/* The Tokens column's number, spelled out here because it can exceed the
            provider-reported total above: the row's composition counts the cache
            buckets, which an additive-convention provider reports outside the prompt. */}
        <DetailField label="Billed tokens">
          <span title="Fresh input, cache reads and writes, and output: the tokens this request was priced on, and the total the activity row's bar splits.">
            {formatTokenCount(buildTokenComposition(entry)?.total ?? null)}
          </span>
        </DetailField>
        <DetailField label="Cost">{formatUSD(entry.cost)}</DetailField>
        {listToolUsage(entry).length ? (
          <>
            <DetailField label="Tools">
              {listToolUsage(entry).map(formatToolUsage).join(" \u00b7 ")}
            </DetailField>
            <DetailField label="Tool cost">
              {computeToolCost(entry) === null ? (
                <span
                  className="text-warning"
                  title="No per-request price is configured for this tool, so its calls were recorded at zero cost. Set one on the Tools & Guardrails screen."
                >
                  unpriced
                </span>
              ) : (
                formatUSD(computeToolCost(entry))
              )}
            </DetailField>
          </>
        ) : null}
        <DetailField label="Cache read tokens">
          {formatTokenCount(entry.cache_read_tokens)}
        </DetailField>
        <DetailField label="Cache write tokens">
          {formatTokenCount(entry.cache_write_tokens)}
        </DetailField>
        <DetailField label="1h cache writes">
          {formatTokenCount(entry.cache_write_1h_tokens ?? null)}
        </DetailField>
        <DetailField label="Total time">
          {formatLatencyCell(entry.latency_ms)}
        </DetailField>
        <DetailField label="Request ID" copyValue={entry.id}>
          {entry.id}
        </DetailField>
      </div>
      {isUncosted ? (
        <div className="flex flex-wrap items-center gap-3">
          {onPriceModel ? (
            <Button
              size="sm"
              variant="ghost"
              onPress={() => onPriceModel(pricingKey)}
            >
              Price this model
            </Button>
          ) : null}
          <span className="text-caption">
            {onPriceModel ? (
              <>
                This request carries no cost. Set a price for{" "}
                <code className="break-all">{pricingKey}</code> so later
                requests are metered and count against budgets. Rows already
                logged keep the cost they were served with.
              </>
            ) : (
              <>
                This request carries no cost, because no price is set for{" "}
                <code className="break-all">{pricingKey}</code>. A deployment
                operator sets one; rows already logged keep the cost they were
                served with.
              </>
            )}
          </span>
        </div>
      ) : null}
      {entry.pricing_breakdown?.length ? (
        <div className="flex flex-col gap-2">
          <span className="text-overline">Billed meters</span>
          <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
            {sortChargeLines(entry.pricing_breakdown).map((line) => {
              // A line of neither known shape was written by an older gateway.
              // Its cost is still real, so it is shown as a charge with no rate
              // rather than rendered through one of the two rate formats, which
              // would print an undefined rate as "NaN / 1M".
              const meter = String(line.meter ?? "")
              return (
                <DetailField key={meter} label={meter.replaceAll("_", " ")}>
                  {isUnitChargeLine(line)
                    ? `${formatTokenCount(line.units)} at ${formatUnitRate(line.unit_rate)} each, ${formatUSD(line.cost)}`
                    : isTokenChargeLine(line)
                      ? `${formatTokenCount(line.units)} at ${formatUSD(line.rate_per_million)} / 1M, ${formatUSD(line.cost)}`
                      : formatUSD(Number(line.cost ?? 0))}
                </DetailField>
              )
            })}
          </div>
        </div>
      ) : null}
    </div>
  )
}
