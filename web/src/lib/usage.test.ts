import { describe, expect, it } from "vitest";

import type { UsageEntry } from "@/api/types";
import { summarizeUsage, usageByDay, usageByModel } from "@/lib/usage";

function entry(overrides: Partial<UsageEntry>): UsageEntry {
  return {
    id: Math.random().toString(36),
    user_id: "u1",
    api_key_id: "k1",
    timestamp: "2026-01-01T10:00:00Z",
    model: "gpt-4o",
    provider: "openai",
    endpoint: "/v1/chat/completions",
    prompt_tokens: 10,
    completion_tokens: 5,
    total_tokens: 15,
    cache_read_tokens: null,
    cache_write_tokens: null,
    cost: 0.01,
    status: "success",
    error_message: null,
    ...overrides,
  };
}

describe("summarizeUsage", () => {
  it("aggregates tokens and cost and computes the error rate", () => {
    const totals = summarizeUsage([
      entry({ total_tokens: 15, cost: 0.01 }),
      entry({ total_tokens: 25, cost: 0.02, status: "error" }),
      entry({ total_tokens: 0, cost: null, prompt_tokens: null, completion_tokens: null }),
    ]);

    expect(totals.requests).toBe(3);
    expect(totals.totalTokens).toBe(40);
    expect(totals.cost).toBeCloseTo(0.03, 6);
    expect(totals.errors).toBe(1);
    expect(totals.errorRate).toBeCloseTo(1 / 3, 6);
  });

  it("returns zeroed totals with no error rate for an empty list", () => {
    const totals = summarizeUsage([]);
    expect(totals.requests).toBe(0);
    expect(totals.errorRate).toBe(0);
  });
});

describe("usageByDay", () => {
  it("buckets entries by calendar day in chronological order", () => {
    const points = usageByDay([
      entry({ timestamp: "2026-01-02T08:00:00Z", cost: 0.02 }),
      entry({ timestamp: "2026-01-01T23:00:00Z", cost: 0.01 }),
      entry({ timestamp: "2026-01-02T09:00:00Z", cost: 0.03 }),
    ]);

    expect(points.map((p) => p.date)).toEqual(["2026-01-01", "2026-01-02"]);
    expect(points[1].requests).toBe(2);
    expect(points[1].cost).toBeCloseTo(0.05, 6);
  });
});

describe("usageByModel", () => {
  it("groups by model and sorts by request count descending", () => {
    const rows = usageByModel([
      entry({ model: "gpt-4o" }),
      entry({ model: "claude-opus-4-8" }),
      entry({ model: "claude-opus-4-8" }),
    ]);

    expect(rows[0]).toMatchObject({ model: "claude-opus-4-8", requests: 2 });
    expect(rows[1]).toMatchObject({ model: "gpt-4o", requests: 1 });
  });
});
