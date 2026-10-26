import { Button } from "@heroui/react"

import type { UsageEntry } from "@/client"
import { isTokenChargeLine, isToolChargeLine, isUnitChargeLine } from "@/client"
import { useMemberAttributionLabels } from "@/features/organization/attribution"
import { userDisplay } from "@/features/users/userDisplay"
import { formatUnitRate } from "@/shared/helpers/format"
import { providerDisplayName } from "@/shared/helpers/providers"
import {
  buildTokenComposition,
  computeToolCost,
  describeSkip,
  describeSource,
  findPricingSelector,
  formatLatencyCell,
  formatTokenCount,
  formatToolUsage,
  formatUSD,
  isSkippedAttempt,
  listToolUsage,
  sortChargeLines,
} from "./activityModel"
import { DetailField } from "./DetailField"
import { RoutingPlan } from "./RoutingPlan"
import { apiKeyLabel } from "./requestColumns"

// The detail panel for one request: the failure diagnostic plus the metadata
// that does not fit the row. The stored `error_message` is shown as stored: the
// gateway redacts upstream text before persisting it, so nothing here needs to.
// It is source-neutral by nature, carrying either a fixed gateway rejection
// string (e.g. a model with no pricing under `require_pricing`) or the upstream
// provider error, so the heading stays "Error" rather than blaming the provider
// for every failure.
export function RequestDetail({
  entry,
  onPriceModel,
}: {
  entry: UsageEntry
  /**
   * Null for a caller who does not operate the deployment. Pricing a model is a
   * deployment-wide write (`/pricing`), so offering the button to a tenant
   * would be offering a refusal; the sentence beside it is still theirs to read,
   * because "this row cost nothing" is a fact about their own request. The
   * endpoint goes with the model, so the price can open in the unit that
   * endpoint is billed in.
   */
  onPriceModel: ((model: string, endpoint: string) => void) | null
}) {
  const memberLabels = useMemberAttributionLabels()
  // A served request with no cost (cost IS NULL, the same test the "Priced?"
  // filter uses) was metered without a price, and pricing the model is the fix;
  // the row holds the selector, which a provider without model discovery would
  // never have put in the catalog. A failed attempt carries no cost because
  // nothing was billed, whatever the cause (a provider 404 for an unknown model,
  // a provider 402 for an exhausted balance), so offering a price there points
  // at the wrong fix; its error, shown above, names the cause, including the
  // gateway's own refusal of an unpriced model. A $0 cost is a real price, so it
  // is deliberately not treated as uncosted.
  const isUnpricedServed = entry.status === "success" && entry.cost === null
  const pricingKey = findPricingSelector(entry)
  const isSkipped = isSkippedAttempt(entry)
  const user =
    entry.user_id === null
      ? null
      : userDisplay(entry.user_id, entry.user_alias, memberLabels)
  return (
    <div className="flex flex-col gap-5 px-3 pt-3 pb-4">
      {/* A skip is the walk moving past a full model, not a failure, so it is
          worded and styled as a note rather than an error. */}
      {isSkipped ? (
        <div className="flex flex-col gap-1.5">
          <span className="text-overline">Skipped</span>
          <p className="rounded-lg border border-border bg-surface p-3 text-xs text-foreground">
            {describeSkip(entry)}. Routing moved this request to the next
            candidate without calling this one.
          </p>
        </div>
      ) : entry.error_message ? (
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
      <div className="grid gap-x-6 gap-y-3 sm:grid-cols-2 lg:grid-cols-4">
        {/* In full: the row's Model and API key lanes cut them short, and a
            title is no help to a keyboard or a touch screen. */}
        <DetailField label="Model">{entry.model}</DetailField>
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
        <DetailField
          label="User"
          copyValue={entry.user_id}
          copyLabel="user id"
          isId={user !== null && user.id === undefined}
        >
          {user?.label ?? "—"}
        </DetailField>
        <DetailField label="API key">{apiKeyLabel(entry)}</DetailField>
        <DetailField label="Prompt tokens">
          {formatTokenCount(entry.prompt_tokens)}
        </DetailField>
        <DetailField label="Completion tokens">
          {formatTokenCount(entry.completion_tokens)}
        </DetailField>
        <DetailField label="Total tokens">
          {formatTokenCount(entry.total_tokens)}
        </DetailField>
        {/* The row's Input and Output added up, spelled out here because it can
            exceed the provider-reported total above: the cache buckets count, and
            an additive-convention provider reports them outside the prompt. */}
        <DetailField label="Billed tokens">
          <span title="Fresh input, cache reads and writes, and output: the tokens this request was priced on, which the row splits into Input and Output.">
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
        <DetailField label="Reasoning tokens">
          {formatTokenCount(entry.reasoning_tokens ?? null)}
        </DetailField>
        <DetailField label="Total time">
          {formatLatencyCell(entry.latency_ms)}
        </DetailField>
        {/* Provider-reported compute time, best-effort (Groq, Ollama today).
            formatLatencyCell's null placeholder means not reported, not zero.
            Diagnostic only: Total time above stays the canonical measure. */}
        <DetailField label="Provider time">
          {formatLatencyCell(entry.provider_latency_ms)}
        </DetailField>
      </div>
      {/* The two ids on a row of their own, each half the panel: a 36-character
          id does not fit a quarter of it beside its copy control. */}
      <div className="grid gap-x-6 gap-y-3 sm:grid-cols-2">
        <DetailField
          label="API key ID"
          copyValue={entry.api_key_id}
          copyLabel="api key id"
          isId
        >
          {entry.api_key_id ?? "—"}
        </DetailField>
        <DetailField label="Request ID" copyValue={entry.id} isId>
          {entry.id}
        </DetailField>
      </div>
      {isUnpricedServed ? (
        <div className="flex flex-wrap items-center gap-3">
          {onPriceModel ? (
            <Button
              size="sm"
              variant="ghost"
              onPress={() => onPriceModel(pricingKey, entry.endpoint)}
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
          <div className="grid gap-x-6 gap-y-3 sm:grid-cols-2 lg:grid-cols-4">
            {sortChargeLines(entry.pricing_breakdown).map((line) => {
              // A line of neither known shape was written by an older gateway.
              // Its cost is still real, so it is shown as a charge with no rate
              // rather than rendered through one of the two rate formats, which
              // would print an undefined rate as "NaN / 1M".
              const meter = String(line.meter ?? "")
              // A tool runs under one line per label, so the label keeps the
              // lines of one meter apart.
              const label = isToolChargeLine(line) ? line.label : null
              return (
                <DetailField
                  key={label === null ? meter : `${meter} ${label}`}
                  label={meter.replaceAll("_", " ")}
                >
                  {label === null ? null : (
                    <>
                      <code className="break-all">{label}</code>:{" "}
                    </>
                  )}
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
