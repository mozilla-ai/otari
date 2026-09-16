import { formatNumber } from "@/shared/helpers/format"
import {
  formatTurnCost,
  formatTurnDuration,
  formatTurnStats,
} from "./helpers/playgroundCost"
import type { TurnUsage } from "./helpers/playgroundTypes"

export function TurnReadout({ usage }: { usage: TurnUsage }) {
  return (
    <div>
      <span className="sr-only">{formatTurnStats(usage)}</span>
      <div
        aria-hidden="true"
        className="flex flex-wrap items-center gap-x-4 gap-y-2 text-mono-caption tabular-nums text-subtle"
      >
        <span className="flex items-baseline gap-1.5">
          <span className="font-medium text-muted">
            {formatNumber(usage.promptTokens)}
          </span>{" "}
          in{" "}
          <span className="ml-1 font-medium text-muted">
            {formatNumber(usage.completionTokens)}
          </span>{" "}
          out
          {usage.cachedTokens > 0 ? (
            <>
              <span className="ml-1 font-medium text-muted">
                {formatNumber(usage.cachedTokens)}
              </span>{" "}
              cached
            </>
          ) : null}
        </span>
        <span className="flex items-baseline gap-2 border-l border-border-strong pl-4">
          <span className="font-medium text-muted">
            {formatTurnDuration(usage.totalMs)}
          </span>
          {usage.ttftMs !== undefined ? (
            <span title="Time to first token">
              {formatTurnDuration(usage.ttftMs)} ttft
            </span>
          ) : null}
        </span>
        {usage.tokensPerSecond !== undefined ? (
          <span className="border-l border-border-strong pl-4">
            <span className="font-medium text-muted">
              {Math.round(usage.tokensPerSecond)}
            </span>{" "}
            tok/s
          </span>
        ) : null}
        {usage.costUsd !== undefined ? (
          <span className="border-l border-border-strong pl-4 font-medium text-foreground">
            {formatTurnCost(usage.costUsd)}
          </span>
        ) : null}
      </div>
    </div>
  )
}
