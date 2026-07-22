import { describe, expect, it } from "vitest";

import type { PricingResponse } from "@/api/types";
import { currentPricing, providerFromModelKey } from "@/lib/pricing";

function row(overrides: Partial<PricingResponse>): PricingResponse {
  return {
    model_key: "openai:gpt-4o",
    effective_at: "2026-01-01T00:00:00Z",
    input_price_per_million: 1,
    output_price_per_million: 2,
    cache_read_price_per_million: null,
    cache_write_price_per_million: null,
    created_at: "2026-01-01T00:00:00Z",
    updated_at: "2026-01-01T00:00:00Z",
    ...overrides,
  };
}

describe("providerFromModelKey", () => {
  it("returns the prefix before the first colon", () => {
    expect(providerFromModelKey("openai:gpt-4o")).toBe("openai");
    expect(providerFromModelKey("home_lab:llama:8b")).toBe("home_lab");
    expect(providerFromModelKey("bare-model")).toBe("—");
  });
});

describe("currentPricing", () => {
  const now = Date.parse("2026-06-01T00:00:00Z");

  it("keeps the newest past-effective row per model and sorts by key", () => {
    const result = currentPricing(
      [
        row({ model_key: "openai:gpt-4o", effective_at: "2026-01-01T00:00:00Z", input_price_per_million: 1 }),
        row({ model_key: "openai:gpt-4o", effective_at: "2026-03-01T00:00:00Z", input_price_per_million: 5 }),
        row({ model_key: "anthropic:claude", effective_at: "2026-02-01T00:00:00Z", input_price_per_million: 3 }),
      ],
      now,
    );

    expect(result.map((r) => r.model_key)).toEqual(["anthropic:claude", "openai:gpt-4o"]);
    expect(result.find((r) => r.model_key === "openai:gpt-4o")?.input_price_per_million).toBe(5);
  });

  it("ignores future-dated rows when an earlier price is active", () => {
    const result = currentPricing(
      [
        row({ effective_at: "2026-01-01T00:00:00Z", input_price_per_million: 1 }),
        row({ effective_at: "2026-12-01T00:00:00Z", input_price_per_million: 9 }),
      ],
      now,
    );

    expect(result).toHaveLength(1);
    expect(result[0].input_price_per_million).toBe(1);
  });
});
