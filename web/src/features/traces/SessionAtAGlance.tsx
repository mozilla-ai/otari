import { type ReactNode, useId, useState } from "react"
import { FiChevronRight } from "react-icons/fi"

import { formatNumber, formatSpend } from "@/shared/helpers/format"

import { formatDuration, formatShare, plural } from "./format"
import {
  costParts,
  type Part,
  type PartKey,
  sum,
  type TimeBreakdown,
  type ToolRows,
  timeBreakdown,
  toolRows,
} from "./sessionBreakdown"
import type { SessionStats } from "./sessionStats"

// Written out whole so Tailwind sees every class. Waiting is hatched, because
// that time was the person's rather than the agent's; idle is outlined, the
// bar's own track drawn as a part.
const SWATCH: Partial<Record<PartKey, string>> = {
  model: "bg-chart-cat-1",
  tools: "bg-chart-cat-2",
  hooks: "bg-chart-cat-3",
  waiting:
    "border border-chart-cat-other bg-[repeating-linear-gradient(135deg,var(--color-chart-cat-other)_0_0.125rem,transparent_0.125rem_0.3125rem)]",
  idle: "border border-border-strong bg-surface-alt",
}
// Cost's parts are ordered rather than named, so they take the ramp by rank,
// the largest strongest.
const RAMP = [
  "bg-chart-ramp-1",
  "bg-chart-ramp-2",
  "bg-chart-ramp-3",
  "bg-chart-ramp-4",
]
const swatchOf = (part: Part, rank: number) =>
  SWATCH[part.key] ?? RAMP[rank] ?? "bg-chart-cat-other"
// An opened tools segment steps down a tone per tool, the slowest strongest.
const TONE = ["", "opacity-70", "opacity-50", "opacity-40"]

// The four lanes every row shares (swatch, name, value, share), so each figure
// lines up down the panel whichever list it is in.
const LANES =
  "grid grid-cols-[0.5rem_minmax(0,1fr)_auto_2.75rem] items-center gap-x-2"

function Line({
  swatch,
  name,
  value,
  share,
  isLead = false,
}: {
  swatch?: string
  name: ReactNode
  value: string
  share: string
  isLead?: boolean
}) {
  return (
    <>
      <span className={swatch ? `size-2 ${swatch}` : ""} />
      {name}
      <span className={`text-mono-caption ${isLead ? "" : "text-muted"}`}>
        {value}
      </span>
      <span className="text-right text-mono-caption text-subtle">{share}</span>
    </>
  )
}

function Name({
  label,
  hint,
  isLead = false,
  isHintLong = false,
}: {
  label: string
  hint?: string
  isLead?: boolean
  /** The hint, not the label, is the one that can run long, so it gives way first. */
  isHintLong?: boolean
}) {
  const [labelFit, hintFit] = isHintLong
    ? ["shrink-0", "min-w-0 truncate"]
    : ["truncate", "shrink-0"]
  return (
    <span className="flex min-w-0 items-baseline gap-1.5">
      <span className={`${labelFit} text-body ${isLead ? "" : "text-muted"}`}>
        {label}
      </span>
      {hint ? (
        <span className={`${hintFit} text-caption text-subtle`}>{hint}</span>
      ) : null}
    </span>
  )
}

function Head({ label, total }: { label: string; total?: string }) {
  return (
    <div className="flex items-baseline justify-between gap-3">
      <h4 className="text-overline text-subtle">{label}</h4>
      {total ? <span className="text-mono-caption">{total}</span> : null}
    </div>
  )
}

// How much of the bar a part takes. At least 2px (`min-w-0.5` beside it), so a
// part too short to see is still there for its row to name.
const basis = (part: Part) => ({ flex: `0 1 ${part.drawn * 100}%` })

// A part-to-whole bar. It says what the rows below it say, so it is hidden from
// assistive technology rather than read out twice.
function Bar({
  parts,
  lane,
  splits = {},
}: {
  parts: Part[]
  lane?: Part | null
  /** A part drawn as several, a tone each: an opened tools part, a tool per tone. */
  splits?: Partial<Record<PartKey, number[]>>
}) {
  const drawn = parts.filter((part) => part.drawn > 0)
  return (
    <div aria-hidden className="flex flex-col gap-0.5">
      <div className="flex h-2.5 gap-0.5 overflow-hidden">
        {drawn.map((part, rank) => (
          <Segment
            key={part.key}
            part={part}
            swatch={swatchOf(part, rank)}
            split={splits[part.key]}
          />
        ))}
      </div>
      {lane ? (
        // The same segments again, empty but the model's, so the lane sits
        // exactly under the request it ran inside.
        <div className="flex h-1 gap-0.5">
          {drawn.map((part) => (
            <span key={part.key} className="flex min-w-0.5" style={basis(part)}>
              {part.key === "model" ? (
                <Segment
                  part={lane}
                  swatch={swatchOf(lane, 0)}
                  split={splits.tools}
                />
              ) : null}
            </span>
          ))}
        </div>
      ) : null}
    </div>
  )
}

function Segment({
  part,
  swatch,
  split = [],
}: {
  part: Part
  swatch: string
  split?: number[]
}) {
  if (part.key === "idle" && part.drawn < part.share) {
    // Drawn shorter than it lasted, and marked so.
    return (
      <span
        className={`flex min-w-0.5 justify-center ${swatch}`}
        style={basis(part)}
      >
        <svg
          aria-hidden="true"
          viewBox="0 0 8 10"
          className="h-full w-2 text-subtle"
        >
          <path d="M1 10L4 0M4 10L7 0" stroke="currentColor" strokeWidth="1" />
        </svg>
      </span>
    )
  }
  if (split.length < 2) {
    return <span className={`min-w-0.5 ${swatch}`} style={basis(part)} />
  }
  const total = sum(split)
  return (
    <span className="flex min-w-0.5 gap-px" style={basis(part)}>
      {split.map((value, i) => (
        <span
          key={i}
          className={`min-w-px ${swatch} ${TONE[Math.min(i, TONE.length - 1)]}`}
          style={{ flex: `0 1 ${(value / total) * 100}%` }}
        />
      ))}
    </span>
  )
}

// Running tools, opening onto each tool it ran. The whole row is the button, as
// a `DisclosureRow` is: the chip alone would be a 20px target in a 32px row.
function ToolsRow({
  part,
  stats,
  rows,
  isLead,
  isOpen,
  onToggle,
}: {
  part: Part
  stats: SessionStats
  rows: ToolRows
  isLead: boolean
  isOpen: boolean
  onToggle: () => void
}) {
  const panelId = useId()
  const share = (ms: number) => formatShare(ms / stats.lengthMs)
  return (
    <li>
      <button
        type="button"
        aria-expanded={isOpen}
        aria-controls={panelId}
        onClick={onToggle}
        className={`${LANES} group -mx-2 min-h-11 w-[calc(100%+1rem)] px-2 text-left transition-colors duration-150 ease-out hover:bg-surface-alt focus-visible:otari-focus-ring motion-reduce:transition-none md:min-h-8`}
      >
        <Line
          swatch={swatchOf(part, 0)}
          isLead={isLead}
          name={
            <span className="flex min-w-0 items-center gap-2">
              <Name label={part.label} isLead={isLead} />
              <span className="flex shrink-0 items-center gap-0.5 bg-surface-alt py-0.5 pr-1.5 pl-1 text-caption text-subtle transition-colors duration-150 ease-out group-hover:bg-surface-subtle motion-reduce:transition-none">
                <FiChevronRight
                  aria-hidden
                  className={`size-3 transition-transform duration-150 ease-out motion-reduce:transition-none ${isOpen ? "rotate-90" : ""}`}
                />
                {plural(stats.tools.length, "tool")}
              </span>
            </span>
          }
          value={formatDuration(part.value)}
          share={formatShare(part.share)}
        />
      </button>
      {/* Rendered while closed too, so `aria-controls` always names an element. */}
      <div
        id={panelId}
        hidden={!isOpen}
        className="ml-[0.1875rem] border-l border-border pl-1.5"
      >
        <ul>
          {rows.shown.map((tool) => (
            <li key={tool.tool} className={`${LANES} min-h-5.5`}>
              <Line
                name={
                  <Name label={tool.tool} hint={plural(tool.calls, "call")} />
                }
                value={formatDuration(tool.runningMs)}
                share={share(tool.runningMs)}
              />
            </li>
          ))}
          {rows.rest ? (
            <li className={`${LANES} min-h-5.5`}>
              <Line
                name={
                  <Name
                    label={plural(rows.rest.tools, "more tool")}
                    hint={plural(rows.rest.calls, "call")}
                  />
                }
                value={formatDuration(rows.rest.runningMs)}
                share={share(rows.rest.runningMs)}
              />
            </li>
          ) : null}
        </ul>
        {rows.overlap ? (
          <p className="pb-1 pl-4 text-caption text-subtle">
            Some calls ran side by side, so these add up to more than{" "}
            {formatDuration(part.value)}.
          </p>
        ) : null}
      </div>
    </li>
  )
}

function Time({ stats, time }: { stats: SessionStats; time: TimeBreakdown }) {
  const [toolsOpen, setToolsOpen] = useState(false)
  const idle = time.parts.find((part) => part.key === "idle")
  const [onlyTool] = stats.tools
  const tools = toolRows(stats.tools, stats.time.toolsMs)
  const opened = toolsOpen
    ? {
        tools: [...tools.shown, ...(tools.rest ? [tools.rest] : [])].map(
          (tool) => tool.runningMs,
        ),
      }
    : {}
  return (
    <div className="flex flex-col gap-2.5">
      <div className="flex flex-col gap-0.5">
        <Head label="Time" total={formatDuration(stats.lengthMs)} />
        {time.idleShortened && idle ? (
          <p className="text-caption text-subtle">
            Idle is {formatShare(idle.share)} of it, so the bar draws it
            shorter, marked {"//"}.
          </p>
        ) : null}
      </div>
      <Bar parts={time.parts} lane={time.lane} splits={opened} />
      <ul>
        {time.parts.map((part, i) =>
          part.key === "tools" && stats.tools.length > 1 ? (
            <ToolsRow
              key={part.key}
              part={part}
              stats={stats}
              rows={tools}
              isLead={i === 0}
              isOpen={toolsOpen}
              onToggle={() => setToolsOpen((open) => !open)}
            />
          ) : (
            <li key={part.key} className={`${LANES} min-h-6`}>
              <Line
                swatch={swatchOf(part, 0)}
                isLead={i === 0}
                name={
                  <Name
                    label={part.label}
                    // One tool has nothing to open, so it is named in the row.
                    hint={
                      part.key === "tools" && onlyTool
                        ? `${onlyTool.tool}, ${plural(onlyTool.calls, "call")}`
                        : part.hint
                    }
                    isHintLong={part.key === "tools"}
                    isLead={i === 0}
                  />
                }
                value={formatDuration(part.value)}
                share={formatShare(part.share)}
              />
            </li>
          ),
        )}
      </ul>
      {time.lane ? (
        <p className="text-caption text-subtle">
          The tools ran while the model's requests were open, as the gateway's
          built-in tools do, so they get a lane of their own rather than a slice
          of the bar.
        </p>
      ) : null}
      {time.overlaps ? (
        <p className="text-caption text-subtle">
          Some of these ran at the same time, and that time counts in each of
          their rows.
        </p>
      ) : null}
    </div>
  )
}

function Cost({ stats, total }: { stats: SessionStats; total: number }) {
  const { by, parts } = costParts(stats)
  // Priced at any rate: a request priced at $0 was priced, not missing a price.
  const isPriced =
    total > 0 ||
    stats.pricedRequests > 0 ||
    stats.models.some((model) => model.cost != null)
  if (!isPriced && stats.requests === 0) return null
  const isSingle = parts.length === 1
  return (
    <div className="flex flex-col gap-2.5">
      <Head
        label="Cost"
        // The recorded total, which Spend and Cost by tool read too; the parts
        // only ever split what of it was priced.
        total={
          isPriced
            ? formatSpend(total > 0 ? total : sum(parts.map((p) => p.value)))
            : "No price"
        }
      />
      {parts.length > 1 ? <Bar parts={parts} /> : null}
      {parts.length > 0 ? (
        <ul>
          {parts.map((part, rank) => (
            <li key={part.key} className={`${LANES} min-h-6`}>
              <Line
                swatch={swatchOf(part, rank)}
                isLead={rank === 0}
                name={
                  <Name
                    label={part.label}
                    hint={part.hint}
                    isLead={rank === 0}
                  />
                }
                value={formatSpend(part.value)}
                // One part is all of it, which the head already says.
                share={isSingle ? "" : formatShare(part.share)}
              />
            </li>
          ))}
        </ul>
      ) : null}
      <CostNotes
        stats={stats}
        by={parts.length > 0 ? by : isPriced ? "whole" : "none"}
      />
    </div>
  )
}

function CostNotes({
  stats,
  by,
}: {
  stats: SessionStats
  /** What the split above is by, or that there is none: priced whole, or not priced. */
  by: "meter" | "model" | "whole" | "none"
}) {
  const { cacheSaved, judges, pricedRequests, requests, tokens } = stats
  const notes = [
    by === "none"
      ? `None of the ${plural(requests, "request")} ran on a model with a price, so there is nothing to split. Set one to see where the money went.`
      : null,
    by === "model"
      ? `${formatNumber(tokens.input + tokens.cacheRead + tokens.cacheWrite)} input and ${formatNumber(tokens.output)} output tokens. These requests were priced whole, so there is no split into cache, output and fresh input.`
      : null,
    by === "meter" && cacheSaved != null && cacheSaved > 0
      ? `Reading the cache saved about ${formatSpend(cacheSaved)} over sending the same tokens fresh.`
      : null,
    by === "meter" && pricedRequests < requests
      ? `Split from the ${plural(pricedRequests, "request")} priced by the token, of ${formatNumber(requests)}.`
      : null,
    judges.calls > 0
      ? judgeNote(judges, by !== "whole" && by !== "none")
      : null,
  ].filter((note) => note !== null)
  if (notes.length === 0) return null
  return (
    <div className="flex flex-col gap-0.5">
      {notes.map((note) => (
        <p key={note} className="text-caption text-subtle">
          {note}
        </p>
      ))}
    </div>
  )
}

// Judge gates call a model of their own, which the split leaves out, and the
// usage ledger may not have priced every call yet.
function judgeNote(judges: SessionStats["judges"], isSplit: boolean): string {
  const times =
    judges.calls === 1 ? "once" : `${formatNumber(judges.calls)} times`
  const called = `Judge gates called ${judges.models.join(", ")} ${times}`
  if (judges.priced === 0) return `${called}.`
  const left = isSplit ? " the split above leaves out" : ""
  const some =
    judges.priced < judges.calls
      ? `, from the ${plural(judges.priced, "call")} priced`
      : ""
  return `${called}, for ${formatSpend(judges.cost)}${left}${some}.`
}

// What a session did, where its time went and what its money paid for, before
// any span is picked.
export function SessionAtAGlance({
  stats,
  cost,
}: {
  stats: SessionStats
  /** The session's total as recorded; the parts split what of it was priced. */
  cost: number
}) {
  const time = timeBreakdown(stats)
  const subagents =
    stats.inSubagents > 0
      ? `, ${formatNumber(stats.inSubagents)} of them in subagents`
      : ""
  return (
    <section className="flex flex-col gap-6">
      <div className="flex flex-col gap-1">
        <h3 className="text-overline">At a glance</h3>
        <p className="text-body">
          {plural(stats.turns, "turn")},{" "}
          {plural(stats.requests, "model request")} and{" "}
          {plural(stats.toolCalls, "tool call")}
          {subagents}.
          {stats.gates.runs > 0
            ? ` Hooks ran ${plural(stats.gates.runs, "gate")}.`
            : null}
        </p>
      </div>
      {time ? <Time stats={stats} time={time} /> : null}
      <Cost stats={stats} total={cost} />
    </section>
  )
}
