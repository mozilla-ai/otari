import type { PricingResponse } from "@/api/types";

export function providerFromModelKey(modelKey: string): string {
  const idx = modelKey.indexOf(":");
  return idx > 0 ? modelKey.slice(0, idx) : "—";
}

// The pricing endpoint returns the full history (one row per effective_at). For
// the table we want the price in effect now per model: the newest row whose
// effective_at is in the past, falling back to the earliest future-dated row if
// every entry is scheduled for later. Results are sorted by model key.
export function currentPricing(rows: PricingResponse[], now: number = Date.now()): PricingResponse[] {
  const byModel = new Map<string, PricingResponse[]>();
  for (const row of rows) {
    const list = byModel.get(row.model_key) ?? [];
    list.push(row);
    byModel.set(row.model_key, list);
  }

  const current: PricingResponse[] = [];
  for (const list of byModel.values()) {
    const sorted = [...list].sort((a, b) => Date.parse(a.effective_at) - Date.parse(b.effective_at));
    const active = [...sorted].reverse().find((row) => Date.parse(row.effective_at) <= now);
    current.push(active ?? sorted[0]);
  }

  return current.sort((a, b) => a.model_key.localeCompare(b.model_key));
}
