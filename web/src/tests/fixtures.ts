// Builders for the API shapes a test needs whole.
//
// These live here because the shapes now come from the generated client, so a
// field the gateway adds arrives in every test at once. Written out inline, that
// meant six files failing to compile over a field none of them cared about. A
// builder fills the whole shape with neutral values and takes an override for
// the part under test, so a test states what it is about and nothing else.

import type { PricingResponse, UsageSeriesPoint, UsageTotals } from "@/client"

export function usageTotals(overrides: Partial<UsageTotals> = {}): UsageTotals {
  return {
    cost: 0,
    prompt_tokens: 0,
    completion_tokens: 0,
    total_tokens: 0,
    cache_read_tokens: 0,
    cache_write_tokens: 0,
    cache_write_1h_tokens: 0,
    billed_input_tokens: 0,
    billed_output_tokens: 0,
    request_count: 0,
    error_count: 0,
    unpriced_requests: 0,
    avg_latency_ms: null,
    ...overrides,
  }
}

export function seriesPoint(
  overrides: Partial<UsageSeriesPoint> & Pick<UsageSeriesPoint, "bucket_start">,
): UsageSeriesPoint {
  return {
    cost: 0,
    tokens: 0,
    requests: 0,
    errors: 0,
    input_tokens: 0,
    output_tokens: 0,
    cache_read_tokens: 0,
    cache_write_tokens: 0,
    ...overrides,
  }
}

export function pricingResponse(
  overrides: Partial<PricingResponse> & Pick<PricingResponse, "model_key">,
): PricingResponse {
  return {
    effective_at: "2026-01-01T00:00:00Z",
    input_price_per_million: 0,
    output_price_per_million: 0,
    cache_read_price_per_million: null,
    cache_write_price_per_million: null,
    cache_write_1h_price_per_million: null,
    pricing_tiers: [],
    created_at: "2026-01-01T00:00:00Z",
    updated_at: "2026-01-01T00:00:00Z",
    ...overrides,
  }
}
