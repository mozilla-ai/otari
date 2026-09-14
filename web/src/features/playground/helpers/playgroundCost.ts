// The per-turn stats line under an assistant reply, and the formatting it needs
// that `shared/helpers/format.ts` does not do.
//
// Money here is unlike money anywhere else in the dashboard. A usage rollup is
// dollars and the shared currency formatter is right for it; one Playground
// reply is often a fraction of a cent, where that formatter renders "$0.00" and
// says the request was free. So this scales its precision to the magnitude, and
// the smallest bucket says "below" rather than rounding to zero.

import { formatNumber } from "@/shared/helpers/format"

import type { TurnUsage } from "./playgroundTypes"

/** The floor below which a cost is reported as a bound rather than a figure. */
const SMALLEST_SHOWN_USD = 0.0001

/**
 * A USD cost with enough precision to be useful at a Playground reply's scale.
 *
 * Exactly zero is "$0", because a free model is a fact worth stating plainly. A
 * real but tiny cost is never rendered as "$0.0000", which reads as free.
 */
export function formatTurnCost(costUsd: number): string {
  if (costUsd === 0) return "$0"
  if (costUsd < SMALLEST_SHOWN_USD) return `<$${SMALLEST_SHOWN_USD}`
  if (costUsd < 0.01) return `$${costUsd.toFixed(4)}`
  if (costUsd < 1) return `$${costUsd.toFixed(3)}`
  return `$${costUsd.toFixed(2)}`
}

/**
 * "0.4s", "2.3s", "1m 5s": a compact duration from milliseconds.
 *
 * Under a tenth of a second it reports the bound rather than "0.0s", for the
 * reason `formatTurnCost` does the same below its floor: a cached reply is
 * genuinely that fast, and a stats line reading "0.0s ... 0.0s to first token"
 * says the request took no time at all.
 */
export function formatTurnDuration(ms: number): string {
  if (ms < 100) return "<0.1s"
  if (ms < 60_000) return `${(ms / 1000).toFixed(1)}s`
  // The total is rounded first and then split, rather than splitting and
  // rounding the remainder: the latter renders 119_600ms as "1m 60s", because
  // 59.6 rounds to 60 with the minute already taken.
  const totalSeconds = Math.round(ms / 1000)
  return `${Math.floor(totalSeconds / 60)}m ${totalSeconds % 60}s`
}

/**
 * Output tokens per second over the decode window, or undefined when there is
 * no window to measure.
 *
 * Deliberately not over the whole request: the time before the first token is
 * prompt processing and queueing, and including it reports a throughput the
 * model never ran at.
 */
export function computeTokensPerSecond(
  completionTokens: number,
  generationMs: number,
): number | undefined {
  if (completionTokens <= 0 || generationMs <= 0) return undefined
  return completionTokens / (generationMs / 1000)
}

/**
 * The stats line, e.g.
 * "30 in · 12 out · 16 cached · $0.003 · 2.3s · 48 tok/s · 0.4s to first token".
 *
 * Every part is dropped when it has nothing to say, so the line shortens rather
 * than carrying placeholders: a model that reported no cached tokens says
 * nothing about caching, and one this deployment prices nothing for shows its
 * tokens and its timing without a cost.
 */
export function formatTurnStats(usage: TurnUsage): string {
  return [
    `${formatNumber(usage.promptTokens)} in`,
    `${formatNumber(usage.completionTokens)} out`,
    usage.cachedTokens > 0
      ? `${formatNumber(usage.cachedTokens)} cached`
      : undefined,
    usage.costUsd !== undefined ? formatTurnCost(usage.costUsd) : undefined,
    formatTurnDuration(usage.totalMs),
    usage.tokensPerSecond !== undefined
      ? `${Math.round(usage.tokensPerSecond)} tok/s`
      : undefined,
    usage.ttftMs !== undefined
      ? `${formatTurnDuration(usage.ttftMs)} to first token`
      : undefined,
  ]
    .filter((part): part is string => part !== undefined)
    .join(" · ")
}
