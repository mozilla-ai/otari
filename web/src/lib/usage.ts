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

export interface ModelUsage {
  // Stable "provider:model" identity (bare model when no provider). Matches the
  // /v1/models catalog id, so callers can join on it and use it as a React key.
  key: string;
  model: string;
  provider: string;
  requests: number;
  promptTokens: number;
  completionTokens: number;
  totalTokens: number;
  cost: number;
}

// Summarizes usage grouped by (provider, model), ordered by request count
// (busiest first). Keying on provider too keeps the same model name from two
// providers from collapsing into one misattributed bucket.
export function usageByModel(entries: UsageEntry[]): ModelUsage[] {
  const buckets = new Map<string, ModelUsage>();

  for (const entry of entries) {
    const model = entry.model || "unknown";
    const provider = entry.provider || "—";
    const key = entry.provider ? `${entry.provider}:${model}` : model;
    const row =
      buckets.get(key) ??
      ({
        key,
        model,
        provider,
        requests: 0,
        promptTokens: 0,
        completionTokens: 0,
        totalTokens: 0,
        cost: 0,
      } satisfies ModelUsage);
    row.requests += 1;
    row.promptTokens += entry.prompt_tokens ?? 0;
    row.completionTokens += entry.completion_tokens ?? 0;
    row.totalTokens += entry.total_tokens ?? 0;
    row.cost += entry.cost ?? 0;
    buckets.set(key, row);
  }

  return [...buckets.values()].sort((a, b) => b.requests - a.requests);
}

// Distinct "provider:model" keys seen across usage entries (bare model when no
// provider was recorded), matching how pricing keys are formed.
export function usedModelKeys(entries: UsageEntry[]): string[] {
  const set = new Set<string>();
  for (const entry of entries) {
    if (!entry.model) {
      continue;
    }
    set.add(entry.provider ? `${entry.provider}:${entry.model}` : entry.model);
  }
  return [...set].sort();
}

// Distinct provider names seen across usage entries.
export function providersFromUsage(entries: UsageEntry[]): string[] {
  const set = new Set<string>();
  for (const entry of entries) {
    if (entry.provider) {
      set.add(entry.provider);
    }
  }
  return [...set].sort();
}
