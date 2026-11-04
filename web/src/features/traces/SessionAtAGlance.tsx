import { Fragment, type ReactNode, useId, useState } from "react"
import { FiChevronRight } from "react-icons/fi"

import { Dot } from "@/design-system/indicators/Dot"
import {
  formatNumber,
  formatSpend,
  formatTokensCompact,
} from "@/shared/helpers/format"

import { ContextMapView } from "./ContextMapView"
import {
  contextCounts,
  contextTotal,
  requestContextMap,
  type SessionRequests,
  sessionNote,
} from "./contextMap"
import { formatDuration, formatShare, plural } from "./format"
import type { GuardrailVerdict } from "./gateway"
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
// that time was the person's rather than the agent's.
const HATCH =
  "bg-[repeating-linear-gradient(135deg,var(--color-chart-cat-other)_0_0.125rem,transparent_0.125rem_0.3125rem)]"
const SWATCH: Partial<Record<PartKey, string>> = {
  model: "bg-chart-cat-1",
  tools: "bg-chart-cat-2",
  hooks: "bg-chart-cat-3",
  waiting: `border border-chart-cat-other ${HATCH}`,
  idle: "border border-border-strong bg-surface-subtle",
}
// On the bar idle is the track itself, nothing drawn on it, so neither part
// needs the outline a legend swatch takes to be seen.
const SEGMENT: Partial<Record<PartKey, string>> = {
  waiting: HATCH,
  idle: "bg-surface-subtle",
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
const segmentOf = (part: Part, rank: number) =>
  SEGMENT[part.key] ?? swatchOf(part, rank)
// An opened tools segment steps down a tone per tool, the slowest strongest.
const TONE = ["", "opacity-70", "opacity-50", "opacity-40"]

// Two columns once each can hold a name beside its figure, one below that,
// and never more than two however wide the panel grows. Dense, so the opened
// tools row taking a whole row leaves no hole beside the cell it opened from.
const LEGEND =
  "grid grid-flow-row-dense grid-cols-[repeat(auto-fill,minmax(max(12.5rem,calc((100%_-_1.5rem)/2)),1fr))] gap-x-6 gap-y-2 pt-1"
const CELL = "flex min-h-5 min-w-0 items-center gap-2"
const BAND = "flex flex-col gap-3 pt-5 pb-6"

// The gates' verdicts in the order the footer counts them, each with its mark.
const VERDICTS: { key: GuardrailVerdict; word: string; dot: string }[] = [
  { key: "passed", word: "passed", dot: "bg-success" },
  { key: "flagged", word: "flagged", dot: "bg-warning" },
  { key: "blocked", word: "blocked", dot: "bg-warning" },
  { key: "unchecked", word: "could not decide", dot: "bg-danger" },
  { key: "refused", word: "could not decide, blocked", dot: "bg-danger" },
]

function Cell({
  swatch,
  name,
  value,
}: {
  swatch: string
  name: ReactNode
  value: string
}) {
  return (
    <>
      <span className={`size-2 shrink-0 ${swatch}`} />
      {name}
      <span className="shrink-0 text-mono-caption">{value}</span>
    </>
  )
}

function Name({
  label,
  hint,
  yields = "label",
}: {
  label: string
  hint?: string
  /** Which of the two truncates first when the row runs out of room. */
  yields?: "label" | "hint"
}) {
  const [labelFit, hintFit] =
    hint && yields === "hint"
      ? ["shrink-0", "min-w-0 truncate"]
      : ["truncate", "shrink-0"]
  return (
    <span className="flex min-w-0 flex-1 items-baseline gap-1.5">
      <span className={`${labelFit} text-body text-muted`}>{label}</span>
      {hint ? (
        <span
          className={`${hintFit} text-caption text-subtle`}
          title={yields === "hint" ? hint : undefined}
        >
          {hint}
        </span>
      ) : null}
    </span>
  )
}

// A band's total said as a phrase, "Took 6m 12s", with what it counts across.
function Head({
  verb,
  total,
  counts,
  isMissing = false,
}: {
  verb: string
  total: string
  counts: Array<string | null>
  /** The total says there is none ("No price"), so it does not read as a figure. */
  isMissing?: boolean
}) {
  const said = counts.filter((count) => count !== null)
  return (
    <div className="flex flex-wrap items-baseline justify-between gap-x-3 gap-y-1">
      <h4 className="flex items-baseline gap-2 text-title">
        <span className="text-subtle">{verb}</span>{" "}
        <span className={`text-mono-title ${isMissing ? "text-subtle" : ""}`}>
          {total}
        </span>
      </h4>
      {said.length > 0 ? (
        <p className="text-body text-subtle">{said.join(" · ")}</p>
      ) : null}
    </div>
  )
}

function Notes({ notes }: { notes: Array<string | null> }) {
  const said = notes.filter((note) => note !== null)
  if (said.length === 0) return null
  return (
    <div className="flex flex-col gap-1 pt-1">
      {said.map((note) => (
        <p key={note} className="text-caption text-subtle">
          {note}
        </p>
      ))}
    </div>
  )
}

// How much of the bar a part takes. At least 2px (`min-w-0.5` beside it), so a
// part too short to see is still there for its cell to name.
const basis = (part: Part) => ({ flex: `0 1 ${part.drawn * 100}%` })

// A part-to-whole bar. It says what the legend below it says, so it is hidden
// from assistive technology rather than read out twice.
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
    <div aria-hidden className="flex flex-col gap-1">
      <div className="flex h-2 gap-1 overflow-hidden">
        {drawn.map((part, rank) => (
          <Segment
            key={part.key}
            part={part}
            swatch={segmentOf(part, rank)}
            split={splits[part.key]}
          />
        ))}
      </div>
      {lane ? (
        // The same segments again, empty but the model's, so the lane sits
        // exactly under the request it ran inside.
        <div className="flex h-1 gap-1">
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

// Running tools, opening onto each tool it ran. The whole cell is the button,
// as a `DisclosureRow` is: on a desk it reaches past the cell's 20px to a 32px
// target without pushing its row apart, and on touch its row grows to 44px.
function ToolsCell({
  part,
  tools,
  panelId,
  isOpen,
  onToggle,
}: {
  part: Part
  tools: number
  panelId: string
  isOpen: boolean
  onToggle: () => void
}) {
  return (
    <li className="flex min-w-0">
      <button
        type="button"
        aria-expanded={isOpen}
        aria-controls={panelId}
        onClick={onToggle}
        className="group -mx-2 flex min-h-11 w-[calc(100%+1rem)] min-w-0 items-center gap-2 px-2 text-left transition-colors duration-150 ease-out hover:bg-surface-alt focus-visible:otari-focus-ring motion-reduce:transition-none md:-my-1.5 md:min-h-5 md:py-1.5"
      >
        <Cell
          swatch={swatchOf(part, 0)}
          name={
            <span className="flex min-w-0 flex-1 items-center gap-1">
              <span className="truncate text-body text-muted transition-colors duration-150 ease-out group-hover:text-foreground group-aria-expanded:text-foreground motion-reduce:transition-none">
                {part.label}
              </span>
              <span className="sr-only">, {plural(tools, "tool")}</span>
              <FiChevronRight
                aria-hidden
                className={`size-3.5 shrink-0 text-subtle transition-transform duration-150 ease-out motion-reduce:transition-none ${isOpen ? "rotate-90" : ""}`}
              />
            </span>
          }
          value={formatDuration(part.value)}
        />
      </button>
    </li>
  )
}

function ToolsPanel({
  id,
  isOpen,
  part,
  rows,
}: {
  id: string
  isOpen: boolean
  part: Part
  rows: ToolRows
}) {
  return (
    // Rendered while closed too, so `aria-controls` always names an element.
    // A whole row of the legend, straight after the row its cell is in.
    <li
      id={id}
      hidden={!isOpen}
      className="col-span-full ml-[0.1875rem] border-l border-border-strong pl-3"
    >
      <ul>
        {rows.shown.map((tool) => (
          <li key={tool.tool} className="flex min-h-6 items-center gap-2">
            <Name label={tool.tool} hint={plural(tool.calls, "call")} />
            <span className="shrink-0 text-mono-caption text-muted">
              {formatDuration(tool.runningMs)}
            </span>
          </li>
        ))}
        {rows.rest ? (
          <li className="flex min-h-6 items-center gap-2">
            <Name
              label={plural(rows.rest.tools, "more tool")}
              hint={plural(rows.rest.calls, "call")}
            />
            <span className="shrink-0 text-mono-caption text-muted">
              {formatDuration(rows.rest.runningMs)}
            </span>
          </li>
        ) : null}
      </ul>
      {rows.overlap ? (
        <p className="pt-1 text-caption text-subtle">
          Some calls ran side by side, so these add up to more than{" "}
          {formatDuration(part.value)}.
        </p>
      ) : null}
    </li>
  )
}

function Time({ stats, time }: { stats: SessionStats; time: TimeBreakdown }) {
  const [toolsOpen, setToolsOpen] = useState(false)
  const panelId = useId()
  const idle = time.parts.find((part) => part.key === "idle")
  const [onlyTool] = stats.tools
  const opens = stats.tools.length > 1
  const tools = toolRows(stats.tools, stats.time.toolsMs)
  const opened = toolsOpen
    ? {
        tools: [...tools.shown, ...(tools.rest ? [tools.rest] : [])].map(
          (tool) => tool.runningMs,
        ),
      }
    : {}
  const subagents =
    stats.inSubagents > 0
      ? `, ${formatNumber(stats.inSubagents)} in subagents`
      : ""
  return (
    <div className={BAND}>
      <Head
        verb="Took"
        total={formatDuration(stats.lengthMs)}
        counts={[
          plural(stats.turns, "turn"),
          stats.toolCalls > 0
            ? `${plural(stats.toolCalls, "tool call")}${subagents}`
            : null,
        ]}
      />
      <Bar parts={time.parts} lane={time.lane} splits={opened} />
      <ul className={LEGEND}>
        {time.parts.map((part) =>
          part.key === "tools" && opens ? (
            <Fragment key={part.key}>
              <ToolsCell
                part={part}
                tools={stats.tools.length}
                panelId={panelId}
                isOpen={toolsOpen}
                onToggle={() => setToolsOpen((open) => !open)}
              />
              <ToolsPanel
                id={panelId}
                isOpen={toolsOpen}
                part={part}
                rows={tools}
              />
            </Fragment>
          ) : (
            <li key={part.key} className={CELL}>
              <Cell
                swatch={swatchOf(part, 0)}
                name={
                  <Name
                    label={part.label}
                    // One tool has nothing to open, so it is named in the cell.
                    hint={
                      part.key === "tools" && onlyTool
                        ? `${onlyTool.tool}, ${plural(onlyTool.calls, "call")}`
                        : part.hint
                    }
                    yields="hint"
                  />
                }
                value={formatDuration(part.value)}
              />
            </li>
          ),
        )}
      </ul>
      <Notes
        notes={[
          time.idleShortened && idle
            ? `Idle is ${formatShare(idle.share)} of the session, so the bar draws it shorter, marked //.`
            : null,
          time.lane
            ? "The tools ran while the model's requests were open, as the gateway's built-in tools do, so they get a lane of their own rather than a slice of the bar."
            : null,
          time.overlaps
            ? "Some of these ran at the same time, and that time counts in each of them."
            : null,
        ]}
      />
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
  // Split by what the tokens paid for; a split by model is not this panel's.
  const split = by === "meter" ? parts : []
  const tokens = sum(Object.values(stats.tokens))
  return (
    <div className={BAND}>
      <Head
        verb="Cost"
        // The recorded total, which Spend and Cost by tool read too; the parts
        // only ever split what of it was priced.
        total={
          isPriced
            ? formatSpend(total > 0 ? total : sum(parts.map((p) => p.value)))
            : "No price"
        }
        isMissing={!isPriced}
        counts={[
          plural(stats.requests, "request"),
          tokens > 0 ? `${formatTokensCompact(tokens)} tokens` : null,
        ]}
      />
      {split.length > 1 ? <Bar parts={split} /> : null}
      {split.length > 0 ? (
        <ul className={LEGEND}>
          {split.map((part, rank) => (
            <li key={part.key} className={CELL}>
              <Cell
                swatch={swatchOf(part, rank)}
                name={<Name label={part.label} />}
                value={formatSpend(part.value)}
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
  const { cacheSaved, judges, pricedRequests, requests } = stats
  return (
    <Notes
      notes={[
        by === "none"
          ? `None of the ${plural(requests, "request")} ran on a model with a price, so there is nothing to split. Set one to see where the money went.`
          : null,
        by === "model"
          ? "These requests were priced whole, so there is no split into cache, output and fresh input."
          : null,
        by === "meter" && cacheSaved != null && cacheSaved > 0
          ? `Reading the cache saved about ${formatSpend(cacheSaved)} over sending the same tokens fresh.`
          : null,
        by === "meter" && pricedRequests < requests
          ? `Split from the ${plural(pricedRequests, "request")} priced by the token, of ${formatNumber(requests)}.`
          : null,
        judges.calls > 0 ? judgeNote(judges, by === "meter") : null,
      ]}
    />
  )
}

// Judge gates call a model of their own, which the split leaves out, and the
// usage ledger may not have priced every call yet.
function judgeNote(judges: SessionStats["judges"], isSplit: boolean): string {
  const called = `Judge gates made ${plural(judges.calls, "model call")}`
  if (judges.priced === 0) return `${called}.`
  const left = isSplit ? " the split above leaves out" : ""
  const some =
    judges.priced < judges.calls
      ? `, from the ${plural(judges.priced, "call")} priced`
      : ""
  return `${called}, for ${formatSpend(judges.cost)}${left}${some}.`
}

function Gates({ gates }: { gates: SessionStats["gates"] }) {
  if (gates.runs === 0) return null
  return (
    <div className="flex flex-wrap items-center justify-between gap-x-4 gap-y-1 py-4">
      <p className="text-body">
        {plural(gates.runs, "gate")}{" "}
        <span className="text-subtle">ran in hooks</span>
      </p>
      <ul className="flex flex-wrap items-center gap-x-4 gap-y-1">
        {VERDICTS.filter(({ key }) => (gates.verdicts[key] ?? 0) > 0).map(
          ({ key, word, dot }) => (
            <li key={key} className="flex items-center gap-1.5 text-caption">
              <Dot className={dot} />
              {formatNumber(gates.verdicts[key] ?? 0)} {word}
            </li>
          ),
        )}
      </ul>
    </div>
  )
}

function Context({ requests }: { requests: SessionRequests }) {
  const map = requestContextMap(requests.latest)
  if (!map) return null
  return (
    <section aria-label="Context" className={BAND}>
      <Head
        verb="Context"
        total={contextTotal(map)}
        counts={contextCounts(map, requests)}
      />
      <ContextMapView map={map} lead={sessionNote(requests)} />
    </section>
  )
}

// What a session did, where its time went and what its money paid for, before
// any span is picked.
export function SessionAtAGlance({
  stats,
  cost,
  context,
}: {
  stats: SessionStats
  /** The session's total as recorded; the parts split what of it was priced. */
  cost: number
  context?: SessionRequests
}) {
  const time = timeBreakdown(stats)
  return (
    <section className="flex flex-col">
      <h3 className="text-overline">At a glance</h3>
      <div className="flex flex-col divide-y divide-border">
        {time ? <Time stats={stats} time={time} /> : null}
        <Cost stats={stats} total={cost} />
        {context ? <Context requests={context} /> : null}
        <Gates gates={stats.gates} />
      </div>
    </section>
  )
}
