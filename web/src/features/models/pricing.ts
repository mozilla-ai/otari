import type { PricingResponse } from "@/client"

export function providerFromModelKey(modelKey: string): string {
  const idx = modelKey.indexOf(":")
  return idx > 0 ? modelKey.slice(0, idx) : "—"
}

// The pricing endpoint returns the full history (one row per effective_at). For
// the table we want the price in effect now per model: the newest row whose
// effective_at is in the past, falling back to the earliest future-dated row if
// every entry is scheduled for later. Results are sorted by model key.
export function currentPricing(
  rows: PricingResponse[],
  now: number = Date.now(),
): PricingResponse[] {
  const byModel = rows.reduce((groups, row) => {
    const list = groups.get(row.model_key)
    if (list) list.push(row)
    else groups.set(row.model_key, [row])
    return groups
  }, new Map<string, PricingResponse[]>())

  return [...byModel.values()]
    .map((list) => {
      const sorted = [...list].sort(
        (a, b) => Date.parse(a.effective_at) - Date.parse(b.effective_at),
      )
      return (
        sorted.findLast((row) => Date.parse(row.effective_at) <= now) ??
        sorted[0]
      )
    })
    .sort((a, b) => a.model_key.localeCompare(b.model_key))
}
