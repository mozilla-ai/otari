import { formatCost, formatNumber, formatPct } from "@/shared/helpers/format"

import { formatDuration } from "./format"
import type { SessionStats } from "./sessionStats"

function plural(count: number, word: string): string {
  return `${formatNumber(count)} ${word}${count === 1 ? "" : "s"}`
}

function Bar({
  label,
  value,
  share,
  hint,
}: {
  label: string
  value: string
  share: number
  hint?: string
}) {
  const width = `${Math.round(Math.min(1, Math.max(0, share)) * 100)}%`
  return (
    <li className="flex flex-col gap-1">
      <div className="flex items-baseline justify-between gap-3">
        <span className="text-body">
          {label}
          {hint ? (
            <span className="text-caption text-subtle"> {hint}</span>
          ) : null}
        </span>
        <span className="shrink-0 text-mono-caption">
          {value} <span className="text-subtle">{formatPct(share)}</span>
        </span>
      </div>
      <div aria-hidden className="h-1.5 rounded-full bg-surface-alt">
        <div className="h-full rounded-full bg-accent" style={{ width }} />
      </div>
    </li>
  )
}

function Time({ stats }: { stats: SessionStats }) {
  const { lengthMs, time } = stats
  const rows = [
    { label: "Model requests", ms: time.modelMs },
    { label: "Hooks", ms: time.hooksMs },
    { label: "Waiting on you", ms: time.waitingMs, hint: "to approve a call" },
    { label: "Running tools", ms: time.toolsMs },
  ]
    .filter((row) => row.ms > 0)
    .sort((a, b) => b.ms - a.ms)
  if (rows.length === 0 || lengthMs <= 0) return null
  return (
    <div className="flex flex-col gap-1.5">
      <h4 className="text-overline text-subtle">
        Where the {formatDuration(lengthMs)} went
      </h4>
      <ul className="flex flex-col gap-2">
        {rows.map((row) => (
          <Bar
            key={row.label}
            label={row.label}
            hint={row.hint}
            value={formatDuration(row.ms)}
            share={row.ms / lengthMs}
          />
        ))}
      </ul>
      {stats.inSubagents > 0 ? (
        <p className="text-caption text-subtle">
          Subagents run alongside the main agent, so these can add up to more
          than the session's length.
        </p>
      ) : null}
    </div>
  )
}

function Money({ stats }: { stats: SessionStats }) {
  const { cost, tokens } = stats
  if (!cost) return null
  const total = cost.input + cost.output + cost.cacheRead + cost.cacheWrite
  if (total <= 0) return null
  const rows = [
    {
      label: "Reading the cache",
      value: cost.cacheRead,
      tokens: tokens.cacheRead,
    },
    {
      label: "Writing the cache",
      value: cost.cacheWrite,
      tokens: tokens.cacheWrite,
    },
    { label: "Output", value: cost.output, tokens: tokens.output },
    { label: "Fresh input", value: cost.input, tokens: tokens.input },
  ].sort((a, b) => b.value - a.value)
  return (
    <div className="flex flex-col gap-1.5">
      <h4 className="text-overline text-subtle">Where the money went</h4>
      <ul className="flex flex-col gap-2">
        {rows.map((row) => (
          <Bar
            key={row.label}
            label={row.label}
            hint={plural(row.tokens, "token")}
            value={formatCost(row.value)}
            share={row.value / total}
          />
        ))}
      </ul>
      {stats.cacheSaved != null && stats.cacheSaved > 0 ? (
        <p className="text-caption text-subtle">
          Caching saved about {formatCost(stats.cacheSaved)}: what the cached
          tokens would have cost as fresh input, less what reading them did.
        </p>
      ) : null}
      {stats.pricedRequests < stats.requests ? (
        <p className="text-caption text-subtle">
          From the {plural(stats.pricedRequests, "request")} priced by the
          token, of {formatNumber(stats.requests)}.
        </p>
      ) : null}
    </div>
  )
}

function Judges({ stats }: { stats: SessionStats }) {
  const { judges } = stats
  if (judges.calls === 0) return null
  return (
    <p className="text-caption text-subtle">
      Judge gates called a model {plural(judges.calls, "time")}, on{" "}
      {judges.models.join(", ")}
      {judges.priced > 0
        ? `, for ${formatCost(judges.cost)} of the session's cost${
            judges.priced < judges.calls
              ? ` (${formatNumber(judges.priced)} priced)`
              : ""
          }`
        : ""}
      . The requests above leave them out.
    </p>
  )
}

function Tools({ stats }: { stats: SessionStats }) {
  if (stats.tools.length === 0) return null
  return (
    <div className="flex flex-col gap-1.5">
      <h4 className="text-overline text-subtle">Tool calls</h4>
      <ul className="flex flex-col border-t border-border">
        {stats.tools.map((tally) => (
          <li
            key={tally.tool}
            className="grid grid-cols-[1fr_auto_4.5rem] items-baseline gap-3 border-b border-border px-2 py-1.5"
          >
            <span className="min-w-0 break-words text-body">{tally.tool}</span>
            <span className="text-mono-caption">
              {plural(tally.calls, "call")}
            </span>
            <span className="text-right text-mono-caption text-subtle">
              {formatDuration(tally.runningMs)}
            </span>
          </li>
        ))}
      </ul>
    </div>
  )
}

// What a session did, where its time went and what its money paid for, before
// any span is picked.
export function SessionAtAGlance({ stats }: { stats: SessionStats }) {
  const subagents =
    stats.inSubagents > 0
      ? `, ${formatNumber(stats.inSubagents)} of them in subagents`
      : ""
  return (
    <section className="flex flex-col gap-3">
      <h3 className="text-overline">At a glance</h3>
      <p className="text-body">
        {plural(stats.turns, "turn")}, {plural(stats.requests, "model request")}{" "}
        and {plural(stats.toolCalls, "tool call")}
        {subagents}.
        {stats.gates.runs > 0
          ? ` Hooks ran ${plural(stats.gates.runs, "gate")}.`
          : null}
      </p>
      <Time stats={stats} />
      <Money stats={stats} />
      <Judges stats={stats} />
      <Tools stats={stats} />
    </section>
  )
}
