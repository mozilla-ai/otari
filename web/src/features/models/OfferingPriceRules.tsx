import type { CatalogOffering } from "@/client"
import { priceRules } from "@/features/models/priceRules"
import { providerDisplayName } from "@/shared/helpers/providers"
import { providerModelId } from "./modelKey"

/**
 * The overrides and multipliers each offering's price carries, read-only and
 * in words. Nothing renders when no offering has one, which is most models.
 * They are set through the pricing API or the config file; this page shows
 * them so a price that moves with the request is not a surprise on the bill.
 */
export function OfferingPriceRules({
  offerings,
}: {
  offerings: readonly CatalogOffering[]
}) {
  const ruled = offerings.flatMap((offering) => {
    const rules = priceRules(offering.pricing?.rates)
    return rules.length === 0 ? [] : [{ offering, rules }]
  })
  if (ruled.length === 0) return null
  return (
    <section
      id="price-rules"
      aria-labelledby="price-rules-title"
      className="flex scroll-mt-4 flex-col gap-3"
    >
      <div className="flex flex-col gap-1">
        <h2 id="price-rules-title" className="text-heading">
          Price rules
        </h2>
        <p className="max-w-prose text-sm text-muted">
          Some prices change with the request. An override replaces a price when
          its condition holds, and a multiplier then scales the prices it names.
          The table above shows the base prices.
        </p>
      </div>
      <ul className="flex flex-col gap-3">
        {ruled.map(({ offering, rules }) => (
          <li
            key={offering.selector}
            className="flex min-w-0 flex-col gap-2 border border-border bg-surface p-3"
          >
            <span className="flex min-w-0 flex-wrap items-baseline gap-x-2 text-body">
              {providerDisplayName(offering.provider)}
              <code className="break-all text-mono-caption text-muted">
                {providerModelId(offering.selector)}
              </code>
            </span>
            <ul
              aria-label={`Price rules of ${offering.selector}`}
              className="flex flex-col gap-1 text-sm"
            >
              {rules.map((rule) => (
                <li key={`${rule.when}|${rule.change}`} className="break-words">
                  <span className="tabular-nums">{rule.change}</span>
                  <span className="text-muted"> when {rule.when}</span>
                </li>
              ))}
            </ul>
          </li>
        ))}
      </ul>
    </section>
  )
}
