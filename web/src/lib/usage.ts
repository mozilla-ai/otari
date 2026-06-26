import type { UsageEntry } from "@/api/types";

export interface UsageTotals {
  requests: number;
  promptTokens: number;
  completionTokens: number;
  totalTokens: number;
  cost: number;
  errors: number;
  errorRate: number;
}

export function summarizeUsage(entries: UsageEntry[]): UsageTotals {
  const totals: UsageTotals = {
    requests: entries.length,
    promptTokens: 0,
    completionTokens: 0,
    totalTokens: 0,
    cost: 0,
    errors: 0,
    errorRate: 0,
  };

  for (const entry of entries) {
    totals.promptTokens += entry.prompt_tokens ?? 0;
    totals.completionTokens += entry.completion_tokens ?? 0;
    totals.totalTokens += entry.total_tokens ?? 0;
    totals.cost += entry.cost ?? 0;
    if (entry.status !== "success") {
      totals.errors += 1;
    }
  }

  totals.errorRate = totals.requests === 0 ? 0 : totals.errors / totals.requests;
  return totals;
}

export interface DailyPoint {
  date: string;
  requests: number;
  cost: number;
}

// Buckets entries by calendar day (local time), returning chronological points.
export function usageByDay(entries: UsageEntry[]): DailyPoint[] {
  const buckets = new Map<string, DailyPoint>();

  for (const entry of entries) {
    const date = new Date(entry.timestamp);
    if (Number.isNaN(date.getTime())) {
      continue;
    }
    const key = date.toISOString().slice(0, 10);
    const point = buckets.get(key) ?? { date: key, requests: 0, cost: 0 };
    point.requests += 1;
    point.cost += entry.cost ?? 0;
    buckets.set(key, point);
  }

  return [...buckets.values()].sort((a, b) => a.date.localeCompare(b.date));
}

export interface ModelUsage {
  model: string;
  requests: number;
  totalTokens: number;
  cost: number;
}

export function usageByModel(entries: UsageEntry[]): ModelUsage[] {
  const buckets = new Map<string, ModelUsage>();

  for (const entry of entries) {
    const model = entry.model || "unknown";
    const row = buckets.get(model) ?? { model, requests: 0, totalTokens: 0, cost: 0 };
    row.requests += 1;
    row.totalTokens += entry.total_tokens ?? 0;
    row.cost += entry.cost ?? 0;
    buckets.set(model, row);
  }

  return [...buckets.values()].sort((a, b) => b.requests - a.requests);
}
