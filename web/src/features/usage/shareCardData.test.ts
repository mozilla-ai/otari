import { describe, expect, it } from "vitest";

import type { UsageGroupRow, UsageSeriesPoint, UsageTotals } from "@/client";

import { availableStats, cardModels, collapseModelName, heroCandidates, resolveHero } from "./shareCardData";
import { seriesPoint } from "@/shared/test/fixtures";

function row(overrides: Partial<UsageGroupRow> & { key: string | null }): UsageGroupRow {
  return { label: null, cost: 0, tokens: 0, requests: 0, is_other: false, ...overrides };
}

// Deliberately leaves the newer fields absent. Several tests below assert what
// happens when a gateway does not send them, and the generated type marks them
// required, so filling them in would quietly delete the case under test.
function totals(overrides: Partial<UsageTotals> = {}): UsageTotals {
  return {
    cost: 0,
    prompt_tokens: 0,
    completion_tokens: 0,
    total_tokens: 0,
    cache_read_tokens: 0,
    cache_write_tokens: 0,
    request_count: 0,
    error_count: 0,
    avg_latency_ms: null,
    ...overrides,
  } as UsageTotals;
}

const series: UsageSeriesPoint[] = [];

describe("cardModels", () => {
  it("ranks by tokens, not cost, so a cheap high-volume model is not buried", () => {
    // The server returns rows biggest-spend-first, which is the wrong order for a
    // card about where the work actually went.
    const rows = [row({ key: "gpt-4o", cost: 42, tokens: 1_000 }), row({ key: "llama-3.3-70b", cost: 0, tokens: 9_000 })];
    expect(cardModels(rows).map((m) => m.key)).toEqual(["llama-3.3-70b", "gpt-4o"]);
  });

  it("collapses each row's model name to the final model type", () => {
    const rows = [row({ key: "otari.ai:fireworks/accounts/deepseek-v4-flash", tokens: 10 })];
    expect(cardModels(rows)[0].label).toBe("deepseek-v4-flash");
  });

  it("names the folded 'other' row and keeps it distinct from a real null key", () => {
    const [fold] = cardModels([row({ key: null, tokens: 5, is_other: true })]);
    expect(fold.label).toBe("other models");
    expect(fold.isOther).toBe(true);

    const [unknown] = cardModels([row({ key: null, tokens: 5 })]);
    expect(unknown.label).toBe("(unknown)");
    expect(unknown.isOther).toBe(false);
  });
});

describe("collapseModelName", () => {
  it("keeps only the final model type from a fully-qualified selector", () => {
    expect(collapseModelName("otari.ai:fireworks/accounts/deepseek-v4-flash")).toBe("deepseek-v4-flash");
  });

  it("leaves an Ollama tag alone, because its colon is part of the model name", () => {
    // Splitting on ":" as well would collapse this to "8b".
    expect(collapseModelName("llama3.1:8b")).toBe("llama3.1:8b");
  });

  it("leaves a bare model name alone", () => {
    expect(collapseModelName("gpt-4o")).toBe("gpt-4o");
  });
});

describe("availableStats", () => {
  it("omits a stat with no value rather than publishing an em dash", () => {
    const stats = availableStats({ totals: totals({ request_count: 5 }), series, hideDollars: false });
    expect(stats.map((s) => s.id)).not.toContain("latency");
    expect(stats.some((s) => s.value.includes("—"))).toBe(false);
  });

  it("omits a $0.00 cost, which is the normal state for a self-hosted gateway", () => {
    const stats = availableStats({ totals: totals({ cost: 0, request_count: 5 }), series, hideDollars: false });
    expect(stats.map((s) => s.id)).not.toContain("cost");
  });

  it("caveats cost when the gateway predates unpriced_requests", () => {
    // undefined means unknown, and an older gateway is *more* likely to be missing
    // prices. A truthy check would drop the caveat exactly there.
    const stats = availableStats({ totals: totals({ cost: 12 }), series, hideDollars: false });
    expect(stats.find((s) => s.id === "cost")?.caveated).toBe(true);
  });

  it("does not caveat cost when the gateway reports zero unpriced requests", () => {
    const stats = availableStats({ totals: totals({ cost: 12, unpriced_requests: 0 }), series, hideDollars: false });
    expect(stats.find((s) => s.id === "cost")?.caveated).toBe(false);
  });

  it("drops cost entirely when dollars are hidden", () => {
    const stats = availableStats({ totals: totals({ cost: 12, request_count: 3 }), series, hideDollars: true });
    expect(stats.map((s) => s.id)).not.toContain("cost");
  });

  it("formats latency in seconds past a second", () => {
    const stats = availableStats({ totals: totals({ avg_latency_ms: 2400 }), series, hideDollars: false });
    expect(stats.find((s) => s.id === "latency")?.value).toBe("2.40 s");
  });
});

describe("heroCandidates", () => {
  it("excludes cache hit rate, which can appear on the card but never carry it", () => {
    const stats = availableStats({
      totals: totals({ request_count: 7 }),
      series: [seriesPoint({ bucket_start: "2026-08-01T00:00:00Z", cost: 0, tokens: 0, requests: 1, input_tokens: 100, cache_read_tokens: 40 })],
      hideDollars: false,
    });
    expect(stats.map((s) => s.id)).toContain("cacheHitRate");
    expect(heroCandidates(stats).map((s) => s.id)).not.toContain("cacheHitRate");
  });
});

describe("resolveHero", () => {
  it("promotes the next available stat rather than leaving a hole", () => {
    // Cost is the preferred hero but dollars are hidden, so it is not in the list.
    const stats = availableStats({ totals: totals({ cost: 12, request_count: 7 }), series, hideDollars: true });
    expect(resolveHero(stats, "cost")?.id).toBe("requests");
  });

  it("returns undefined only when there is nothing at all to show", () => {
    expect(resolveHero([], "requests")).toBeUndefined();
  });

  it("never falls back to a stat that cannot lead", () => {
    const stats = availableStats({
      totals: totals(),
      series: [seriesPoint({ bucket_start: "2026-08-01T00:00:00Z", cost: 0, tokens: 0, requests: 1, input_tokens: 100, cache_read_tokens: 40 })],
      hideDollars: false,
    });
    expect(stats.map((s) => s.id)).toEqual(["cacheHitRate"]);
    expect(resolveHero(stats, "cost")).toBeUndefined();
  });
});
