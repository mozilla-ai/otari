import { describe, expect, it } from "vitest";

import type { UsageEntry } from "@/api/types";
import { providersFromUsage, summarizeUsage, usageByModel } from "@/lib/usage";

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

describe("usageByModel", () => {
  it("groups by model with token breakdowns, sorted by request count descending", () => {
    const rows = usageByModel([
      entry({ model: "gpt-4o", provider: "openai", prompt_tokens: 10, completion_tokens: 5 }),
      entry({ model: "claude-opus-4-8", provider: "anthropic", prompt_tokens: 3, completion_tokens: 2 }),
      entry({ model: "claude-opus-4-8", provider: "anthropic", prompt_tokens: 7, completion_tokens: 1 }),
    ]);

    expect(rows[0]).toMatchObject({
      key: "anthropic:claude-opus-4-8",
      model: "claude-opus-4-8",
      provider: "anthropic",
      requests: 2,
      promptTokens: 10,
      completionTokens: 3,
    });
    expect(rows[1]).toMatchObject({ key: "openai:gpt-4o", model: "gpt-4o", requests: 1, promptTokens: 10 });
  });

  it("keeps the same model name from different providers in separate buckets", () => {
    const rows = usageByModel([
      entry({ model: "llama-3", provider: "openrouter", cost: 0.01 }),
      entry({ model: "llama-3", provider: "together", cost: 0.02 }),
    ]);

    expect(rows).toHaveLength(2);
    expect(rows.map((r) => r.key).sort()).toEqual(["openrouter:llama-3", "together:llama-3"]);
  });
});

describe("providersFromUsage", () => {
  it("returns the distinct, sorted set of providers", () => {
    const providers = providersFromUsage([
      entry({ provider: "openai" }),
      entry({ provider: "anthropic" }),
      entry({ provider: "openai" }),
      entry({ provider: null }),
    ]);

    expect(providers).toEqual(["anthropic", "openai"]);
  });
});
