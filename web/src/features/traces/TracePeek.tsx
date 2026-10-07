import { Link } from "@tanstack/react-router"
import { useCallback, useEffect, useRef, useState } from "react"
import {
  FiArrowUpRight,
  FiChevronDown,
  FiChevronUp,
  FiMaximize2,
  FiMinimize2,
  FiX,
} from "react-icons/fi"

import { CopyableValue } from "@/design-system/actions/CopyField"
import { IconButton } from "@/design-system/actions/IconButton"
import { EmptyMessage } from "@/design-system/feedback/EmptyMessage"
import { Kbd } from "@/design-system/indicators/Kbd"
import { Segmented } from "@/design-system/navigation/Segmented"
import { Tab, TabRow } from "@/design-system/navigation/TabRow"
import { Tooltip } from "@/design-system/overlays/Tooltip"
import {
  formatCost,
  formatNumber,
} from "@/shared/helpers/format"

import { formatDuration, formatTraceTimestamp } from "./format"
import { TypeMark } from "./observationTypes"
import { summarizeTrace, type Trace } from "./types"

type PeekTab = "preview" | "log"
type Format = "formatted" | "raw"

// Langfuse shows Attributes only while an observation is selected
// (TraceDetailView.tsx). P1 selects the trace root only, so it has two tabs.
const TABS: { id: PeekTab; label: string }[] = [
  { id: "preview", label: "Preview" },
  { id: "log", label: "Log View" },
]

const FORMAT_OPTIONS = [
  { value: "formatted", label: "Formatted" },
  { value: "raw", label: "Raw" },
]

const MIN_FRACTION = 0.4
const MAX_FRACTION = 0.9
const DEFAULT_FRACTION = 0.5
const DEFAULT_CAP_PX = 1400 // 87.5rem
const EXIT_MS = 250

function defaultWidth(): number {
  return Math.min(window.innerWidth * DEFAULT_FRACTION, DEFAULT_CAP_PX)
}

// A 4px bar in a 12px strip straddling an edge: transparent at rest, drawn on
// hover, focus and drag. Shared by the panel's outer edge and the inner divider.
function ResizeHandle({
  label,
  className,
  onDrag,
  onStep,
}: {
  label: string
  className: string
  onDrag: (clientX: number) => void
  onStep: (direction: -1 | 1) => void
}) {
  const [dragging, setDragging] = useState(false)
  return (
    // biome-ignore lint/a11y/useSemanticElements: an <hr> cannot take focus or a drag
    <div
      role="separator"
      aria-orientation="vertical"
      aria-label={label}
      tabIndex={0}
      className={`group absolute inset-y-0 z-10 w-3 cursor-ew-resize touch-none outline-none ${className}`}
      onPointerDown={(event) => {
        event.currentTarget.setPointerCapture(event.pointerId)
        setDragging(true)
      }}
      onPointerMove={(event) => {
        if (dragging) onDrag(event.clientX)
      }}
      onPointerUp={() => setDragging(false)}
      onPointerCancel={() => setDragging(false)}
      onKeyDown={(event) => {
        if (event.key === "ArrowLeft") onStep(-1)
        if (event.key === "ArrowRight") onStep(1)
      }}
    >
      <span
        className={`absolute inset-y-0 left-1/2 w-1 -translate-x-1/2 transition-colors group-hover:bg-control-border group-focus-visible:bg-control-border motion-reduce:transition-none ${
          dragging ? "bg-control-border" : "bg-transparent"
        }`}
      />
    </div>
  )
}

function ConversationBlock({
  label,
  value,
  format,
}: {
  label: string
  value: unknown
  format: Format
}) {
  const messages = Array.isArray(value) ? value : null
  return (
    <section className="flex flex-col gap-1.5">
      <h3 className="text-overline text-muted">{label}</h3>
      {value == null ? (
        <span className="text-subtle">—</span>
      ) : format === "formatted" && messages ? (
        <div className="flex flex-col gap-2">
          {messages.map((message, index) => (
            // biome-ignore lint/suspicious/noArrayIndexKey: static mock content
            <div key={index} className="flex flex-col gap-0.5">
              <span className="text-mono-caption text-muted">
                {String((message as { role?: string }).role ?? "message")}
              </span>
              <p className="text-body whitespace-pre-wrap">
                {String((message as { content?: string }).content ?? "")}
              </p>
            </div>
          ))}
        </div>
      ) : format === "formatted" &&
        typeof value === "object" &&
        "message" in (value as object) ? (
        <p className="text-body whitespace-pre-wrap">
          {String((value as { message: unknown }).message)}
        </p>
      ) : (
        <pre className="overflow-x-auto border border-border bg-surface-subtle p-3 text-mono-caption whitespace-pre-wrap">
          {JSON.stringify(value, null, 2)}
        </pre>
      )}
    </section>
  )
}

function Breakdown({ rows }: { rows: [string, string][] }) {
  return (
    <dl className="grid grid-cols-[auto_auto] gap-x-4 gap-y-0.5 text-mono-caption">
      {rows.map(([key, value]) => (
        <div key={key} className="contents">
          <dt>{key}</dt>
          <dd className="text-right">{value}</dd>
        </div>
      ))}
    </dl>
  )
}

const DOTTED =
  "underline decoration-dotted decoration-control-border underline-offset-4"

export function TracePeek({
  trace,
  onClose,
  onPrevious,
  onNext,
}: {
  trace: Trace
  onClose: () => void
  onPrevious?: () => void
  onNext?: () => void
}) {
  const [tab, setTab] = useState<PeekTab>("preview")
  const [format, setFormat] = useState<Format>("formatted")
  const [width, setWidth] = useState(defaultWidth)
  const [expanded, setExpanded] = useState(false)
  const [phase, setPhase] = useState<"entering" | "open" | "leaving">(
    "entering",
  )
  const [navFraction, setNavFraction] = useState(0.4)
  const bodyRef = useRef<HTMLDivElement>(null)
  const summary = summarizeTrace(trace)

  // setTimeout rather than requestAnimationFrame: a hidden tab never fires rAF.
  useEffect(() => {
    const timer = setTimeout(() => setPhase("open"), 20)
    return () => clearTimeout(timer)
  }, [])

  const close = useCallback(() => {
    setPhase("leaving")
    setTimeout(onClose, EXIT_MS)
  }, [onClose])

  useEffect(() => {
    function onKeyDown(event: KeyboardEvent) {
      const target = event.target as HTMLElement | null
      const typing =
        target?.tagName === "INPUT" ||
        target?.tagName === "TEXTAREA" ||
        target?.isContentEditable
      if (event.key === "Escape") close()
      if (typing || event.metaKey || event.ctrlKey || event.altKey) return
      if (event.key === "k") onPrevious?.()
      if (event.key === "j") onNext?.()
    }
    document.addEventListener("keydown", onKeyDown)
    return () => document.removeEventListener("keydown", onKeyDown)
  }, [close, onPrevious, onNext])

  const clampWidth = (px: number) =>
    Math.min(
      window.innerWidth * MAX_FRACTION,
      Math.max(window.innerWidth * MIN_FRACTION, px),
    )

  const shownWidth = expanded ? window.innerWidth * MAX_FRACTION : width
  const model = summary.models[0]

  return (
    <aside
      aria-label={`Trace ${trace.name}`}
      style={{ "--peek-width": `${shownWidth}px` } as React.CSSProperties}
      className={`fixed top-14 right-0 bottom-0 z-30 flex w-full flex-col border-l border-border-strong bg-surface transition-transform duration-250 ease-out-fluid motion-reduce:transition-none md:w-(--peek-width) ${
        phase === "open" ? "translate-x-0" : "translate-x-full"
      }`}
    >
      <ResizeHandle
        label="Resize trace panel"
        className="-left-1 max-md:hidden"
        onDrag={(clientX) => {
          setExpanded(false)
          setWidth(clampWidth(window.innerWidth - clientX))
        }}
        onStep={(direction) =>
          setWidth(clampWidth(shownWidth - direction * 48))
        }
      />

      <div className="flex h-12 shrink-0 items-center justify-between gap-2 border-b border-border px-4">
        <div className="flex min-w-0 items-center gap-2">
          <span className="text-overline text-muted">Trace</span>
          <CopyableValue
            value={trace.id}
            label="Copy trace ID"
            className="truncate text-mono-caption"
          />
        </div>
        <div className="otari-toolbar flex items-center gap-1">
          <Tooltip
            content={
              <span className="flex items-center gap-2">
                Previous trace <Kbd keys={["K"]} />
              </span>
            }
          >
            <IconButton
              label="Previous trace"
              size="sm"
              onPress={onPrevious}
              isDisabled={!onPrevious}
            >
              <FiChevronUp aria-hidden="true" className="size-4" />
            </IconButton>
          </Tooltip>
          <Tooltip
            content={
              <span className="flex items-center gap-2">
                Next trace <Kbd keys={["J"]} />
              </span>
            }
          >
            <IconButton
              label="Next trace"
              size="sm"
              onPress={onNext}
              isDisabled={!onNext}
            >
              <FiChevronDown aria-hidden="true" className="size-4" />
            </IconButton>
          </Tooltip>
          <span className="max-md:hidden">
            <Tooltip content={expanded ? "Collapse" : "Expand"}>
              <IconButton
                label={expanded ? "Collapse" : "Expand"}
                size="sm"
                onPress={() => setExpanded((value) => !value)}
              >
                {expanded ? (
                  <FiMinimize2 aria-hidden="true" className="size-4" />
                ) : (
                  <FiMaximize2 aria-hidden="true" className="size-4" />
                )}
              </IconButton>
            </Tooltip>
          </span>
          <Tooltip content="Close">
            <IconButton label="Close" size="sm" onPress={close}>
              <FiX aria-hidden="true" className="size-4" />
            </IconButton>
          </Tooltip>
        </div>
      </div>

      <div className="flex shrink-0 flex-col gap-2 px-4 py-3">
        <div className="flex items-center gap-2">
          <span className="grid size-6 shrink-0 place-items-center">
            <TypeMark type="TRACE" className="size-5" />
          </span>
          <h2 className="truncate text-title">{trace.name}</h2>
        </div>
        <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-mono-caption text-muted">
          <span>{formatTraceTimestamp(trace.timestamp, true)}</span>
          {formatDuration(summary.latencyMs) ? (
            <span>{formatDuration(summary.latencyMs)}</span>
          ) : null}
          {summary.totalCost > 0 ? (
            <Tooltip
              content={
                <Breakdown
                  rows={[
                    ["Input", formatCost(summary.inputCost)],
                    ["Output", formatCost(summary.outputCost)],
                  ]}
                />
              }
            >
              {(props) => (
                <button type="button" {...props} className={DOTTED}>
                  {formatCost(summary.totalCost)}
                </button>
              )}
            </Tooltip>
          ) : null}
          {summary.totalTokens > 0 ? (
            <Tooltip
              content={
                <Breakdown
                  rows={[
                    ["Input", formatNumber(summary.promptTokens)],
                    ["Output", formatNumber(summary.completionTokens)],
                  ]}
                />
              }
            >
              {(props) => (
                <button type="button" {...props} className={DOTTED}>
                  {formatNumber(summary.totalTokens)} tokens
                </button>
              )}
            </Tooltip>
          ) : null}
          {model ? (
            <Link
              to="/models"
              className="inline-flex items-center gap-0.5 text-link"
            >
              {model}
              <FiArrowUpRight aria-hidden="true" className="h-3.5 w-3.5" />
            </Link>
          ) : null}
        </div>
      </div>

      <div className="flex shrink-0 items-center justify-between gap-2 border-b border-border px-4 py-2">
        <TabRow>
          {TABS.map(({ id, label }) => (
            <Tab key={id} isActive={tab === id} onPress={() => setTab(id)}>
              {label}
            </Tab>
          ))}
        </TabRow>
        <Segmented
          size="sm"
          label="Format"
          options={FORMAT_OPTIONS}
          value={format}
          onChange={(next) => setFormat(next as Format)}
        />
      </div>

      <div ref={bodyRef} className="relative flex min-h-0 flex-1 overflow-auto">
        <>
          <div
            style={{ width: `${navFraction * 100}%` }}
            className="relative flex min-w-[260px] shrink-0 flex-col border-r border-border"
          >
            <div className="p-4">
              <EmptyMessage>
                {summary.observationCount} observations. The tree and timeline
                land here.
              </EmptyMessage>
            </div>
            <ResizeHandle
              label="Resize navigation pane"
              className="-right-1.5"
              onDrag={(clientX) => {
                const box = bodyRef.current?.getBoundingClientRect()
                if (!box) return
                setNavFraction(
                  Math.min(
                    0.7,
                    Math.max(0.2, (clientX - box.left) / box.width),
                  ),
                )
              }}
              onStep={(direction) =>
                setNavFraction((value) =>
                  Math.min(0.7, Math.max(0.2, value + direction * 0.04)),
                )
              }
            />
          </div>
          <div className="flex min-w-[360px] flex-1 flex-col gap-4 overflow-auto p-4">
            {tab === "preview" ? (
              <>
                <ConversationBlock
                  label="Input"
                  value={trace.input}
                  format={format}
                />
                <ConversationBlock
                  label="Output"
                  value={trace.output}
                  format={format}
                />
              </>
            ) : (
              <EmptyMessage>The log view is not built yet.</EmptyMessage>
            )}
          </div>
        </>
      </div>
    </aside>
  )
}
