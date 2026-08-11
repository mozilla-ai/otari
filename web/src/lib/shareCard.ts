import type { UsageGroupRow, UsageSeriesPoint, UsageTotals } from "@/api/types";

import { formatNumber, formatPct, formatTokens, formatUsd } from "./format";
import { billedTokenTotal, cacheHitRate, costNeedsCaveat, formatLatency } from "./usageTotals";

// Derivations for the share card. Everything here is pure so the numbers the card
// publishes can be tested without a browser: the rasterizer cannot run in jsdom,
// but the claims the image makes are all decided in this file.

export interface CardModel {
  /** The model name, or undefined for the server's folded "other" row. */
  key: string | undefined;
  label: string;
  tokens: number;
  cost: number;
  requests: number;
  isOther: boolean;
}

/**
 * Rank the card's model rows by **tokens**.
 *
 * The server returns breakdowns biggest-spend-first (`order_by(cost_sum.desc())`),
 * which buries a high-volume model that happens to be cheap or self-hosted. The
 * card is about where the work went, so it ranks by volume and says so.
 */
/**
 * Collapse a stored model name to the part worth printing.
 *
 * `UsageLog.model` holds whatever selector resolved, which for a routed or
 * aggregator-served request is fully qualified:
 * `otari.ai:fireworks/accounts/deepseek-v4-flash`. On a card at feed scale the
 * routing path is noise; the model is the claim.
 *
 * Only the last path segment is taken. Splitting on ":" as well would be wrong:
 * an Ollama tag carries a colon inside the model name itself, so `llama3.1:8b`
 * would collapse to `8b`. A bare `provider:model` with no slash therefore keeps
 * its prefix, which is honest rather than mangled.
 */
export function collapseModelName(name: string): string {
  return name.slice(name.lastIndexOf("/") + 1);
}

export function cardModels(rows: UsageGroupRow[]): CardModel[] {
  return rows
    .map((row) => ({
      key: row.is_other ? undefined : (row.key ?? undefined),
      label: row.is_other ? "other models" : row.key === null ? "(unknown)" : collapseModelName(row.key),
      tokens: row.tokens,
      cost: row.cost,
      requests: row.requests,
      isOther: row.is_other,
    }))
    .sort((a, b) => b.tokens - a.tokens);
}

export type StatId = "cost" | "requests" | "tokens" | "cacheHitRate" | "latency";

export interface CardStat {
  id: StatId;
  label: string;
  value: string;
  /** True when the value carries a caveat the card must show (currently only unpriced cost). */
  caveated?: boolean;
}

export interface StatInputs {
  totals: UsageTotals | undefined;
  series: UsageSeriesPoint[];
  hideDollars: boolean;
}

/**
 * Build the stats the card *can* show, in descending claim strength.
 *
 * A stat with no value is omitted rather than rendered as an em dash: the page
 * uses "—" to keep table cells aligned, but a public artifact should not publish
 * a placeholder. Cost is omitted when dollars are hidden, and also when it is
 * exactly zero, since a proud "$0.00" is not a claim worth posting (and is the
 * normal state for a gateway serving only self-hosted models).
 */
export function availableStats(inputs: StatInputs): CardStat[] {
  const { totals, series, hideDollars } = inputs;
  const stats: CardStat[] = [];
  if (totals === undefined) {
    return stats;
  }

  if (!hideDollars && totals.cost > 0) {
    stats.push({ id: "cost", label: "Spend", value: formatUsd(totals.cost), caveated: costNeedsCaveat(totals) });
  }
  if (totals.request_count > 0) {
    stats.push({ id: "requests", label: "Requests", value: formatNumber(totals.request_count) });
  }
  const billed = billedTokenTotal(totals);
  if (billed !== undefined && billed > 0) {
    stats.push({ id: "tokens", label: "Tokens", value: formatTokens(billed) });
  }
  const hitRate = cacheHitRate(series);
  if (hitRate !== undefined) {
    stats.push({ id: "cacheHitRate", label: "Cache hits", value: formatPct(hitRate) });
  }
  const latency = formatLatency(totals.avg_latency_ms);
  if (latency !== undefined) {
    stats.push({ id: "latency", label: "Avg latency", value: latency });
  }
  return stats;
}

// Stats that can appear on the card but never carry it. Cache hit rate is a
// gateway-tuning number: it means little to a reader outside the deployment and
// says nothing about what the work was.
const NON_HERO: readonly StatId[] = ["cacheHitRate"];

/** The stats offered as the card's lead, in descending claim strength. */
export function heroCandidates(stats: CardStat[]): CardStat[] {
  return stats.filter((stat) => !NON_HERO.includes(stat.id));
}

/**
 * Resolve the hero slot.
 *
 * The hero can never be empty: if the preferred stat is unavailable (dollars
 * hidden while cost was the hero, or a metric the window has no value for) the
 * next available stat is promoted, so the card never renders a hole.
 * Undefined only when there is nothing at all to show, which is the empty state.
 */
export function resolveHero(stats: CardStat[], preferred: StatId): CardStat | undefined {
  const candidates = heroCandidates(stats);
  return candidates.find((stat) => stat.id === preferred) ?? candidates[0];
}
