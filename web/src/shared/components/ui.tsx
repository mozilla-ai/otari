import {
  Button,
  Card,
  Chip,
  ComboBox,
  Input,
  Label,
  ListBox,
  ListBoxItem,
  Select,
  Spinner,
  Tooltip,
} from "@heroui/react"
import type { LinkProps } from "@tanstack/react-router"
import { Link } from "@tanstack/react-router"
import type { KeyboardEvent as ReactKeyboardEvent, ReactNode } from "react"
import { useEffect, useId, useRef, useState } from "react"
import { Checkbox as AriaCheckbox } from "react-aria-components"
import { ApiError } from "@/shared/api/client"
import { Dot } from "@/shared/components/surface"
import { copyToClipboard } from "@/shared/helpers/clipboard"
import { formatRelative } from "@/shared/helpers/format"

// The box visual, split out so it can hold optimistic state: react-aria only
// reports the new `isSelected` after the whole collection re-renders (O(rows)
// per click, tens to hundreds of ms on big pages or slow machines), which made
// the checkmark feel laggy. On pointerdown the visual flips immediately; the
// authoritative state catches up and clears the override, and a timeout clears
// it as a backstop if the press never lands (e.g. drag-away).
export function CheckboxVisual({
  isSelected,
  isIndeterminate,
  isDisabled,
}: {
  isSelected: boolean
  isIndeterminate: boolean
  isDisabled: boolean
}) {
  const [flash, setFlash] = useState<boolean | null>(null)

  useEffect(() => {
    if (flash !== null && isSelected === flash) setFlash(null)
  }, [isSelected, flash])
  useEffect(() => {
    if (flash === null) return
    const timer = setTimeout(() => setFlash(null), 600)
    return () => clearTimeout(timer)
  }, [flash])

  const showChecked = flash ?? (isSelected || isIndeterminate)
  return (
    <span
      onPointerDown={() => {
        if (!isDisabled) setFlash(!isSelected)
      }}
      // Square, and outlined in the control edge rather than the divider
      // border: `--color-border` is a 0.06 alpha tuned to separate two surfaces
      // of nearly the same value, which leaves a 16px box on the page ground
      // almost invisible. Checked drops the border entirely so the accent fill
      // is the whole shape.
      className={`flex h-4 w-4 items-center justify-center transition-colors ${
        showChecked
          ? "bg-accent text-accent-foreground"
          : "border border-control-border bg-background"
      } group-data-[focus-visible]:otari-focus-ring`}
    >
      {isIndeterminate && flash === null ? (
        <svg
          viewBox="0 0 24 24"
          className="h-3 w-3"
          fill="none"
          stroke="currentColor"
          strokeWidth={3}
          aria-hidden="true"
        >
          <line x1="6" x2="18" y1="12" y2="12" strokeLinecap="round" />
        </svg>
      ) : showChecked ? (
        <svg
          viewBox="0 0 24 24"
          className="h-3 w-3"
          fill="none"
          stroke="currentColor"
          strokeWidth={3}
          aria-hidden="true"
        >
          <polyline
            points="5 12 10 17 19 7"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        </svg>
      ) : null}
    </span>
  )
}

/**
 * A checkbox on the design tokens.
 *
 * react-aria rather than HeroUI's own `Checkbox`, for the reason
 * `DataTable`'s selection box gives: HeroUI splits the control across
 * subcomponents and the two would not look alike. One visual serves both, so a
 * standalone checkbox and a table's selection box cannot drift apart.
 */
export function Checkbox({
  isSelected,
  onChange,
  isDisabled = false,
  children,
}: {
  isSelected: boolean
  onChange: (isSelected: boolean) => void
  isDisabled?: boolean
  children: ReactNode
}) {
  return (
    <AriaCheckbox
      isSelected={isSelected}
      onChange={onChange}
      isDisabled={isDisabled}
      className="group flex w-fit items-center gap-2 text-sm text-foreground"
    >
      {({ isSelected: selected, isDisabled: disabled }) => (
        <>
          <CheckboxVisual
            isSelected={selected}
            isIndeterminate={false}
            isDisabled={disabled}
          />
          {children}
        </>
      )}
    </AriaCheckbox>
  )
}

// A tile's attention status, on the foundation's three status roles. Color is
// never the only signal: a status tile also carries a word/icon via `statusLabel`.
export type StatStatus = "ok" | "warn" | "alert"

// The accent bar is `!`-important because the design foundation gives every
// HeroUI Card a 1px outline (`.card:not(.card--transparent)` in globals.css),
// and that rule is unlayered while a Tailwind utility sits in @layer utilities;
// unlayered always wins, whatever the specificity. Without the bang the
// shorthand `border:` resets this tile's left edge back to a hairline.
//
// The pill beside the value is HeroUI's own Chip, for the reason TrendChip is:
// `globals.css` aliases the status bases the chip's CSS reads (`--success`,
// `--warning`, `--danger`) onto our tokens, so naming a status here is all it
// takes for both themes to follow the foundation. This used to be a hand-rolled
// <span> carrying its own border/bg/text triple per status, which restated in
// three class strings what the library already derives, and left the status pill
// and the trend chip on the same tile as two different shapes.
const STAT_STATUS: Record<
  StatStatus,
  { accent: string; chip: "success" | "warning" | "danger" }
> = {
  ok: { accent: "border-l-success!", chip: "success" },
  warn: { accent: "border-l-warning!", chip: "warning" },
  alert: { accent: "border-l-danger!", chip: "danger" },
}

export function StatCard({
  label,
  value,
  hint,
  trend,
  status,
  statusLabel,
  chart,
  to,
}: {
  label: string
  value: ReactNode
  // Supporting context under the value: what the number is made of ("5.8%
  // errors", "311.2k read"), not how it moved. The movement is `trend`, and the
  // two share one row.
  hint?: ReactNode
  // Period-over-period change, as a <TrendChip>. It leads the aside row under
  // the value, ahead of `hint`: a pill sharing the value's baseline competes
  // with the number for the first glance, while a second row of its own spends
  // a line saying what belongs beside the hint anyway.
  trend?: ReactNode
  status?: StatStatus
  // A short word (and/or icon) shown as a pill beside the value. Required to be a
  // non-color signal so status is legible without hue (colorblind operators).
  statusLabel?: ReactNode
  // An optional trend visual (e.g. a <Sparkline>) rendered under the value/hint,
  // for KPI tiles that have a bucketed series on the wire.
  chart?: ReactNode
  // When set, the whole tile is a keyboard-focusable link to this route.
  to?: LinkProps["to"]
}) {
  const accent = status ? `border-l-4! ${STAT_STATUS[status].accent}` : ""
  const body = (
    // p-0 on the Card zeroes HeroUI's own card padding so it doesn't stack with
    // Card.Content's, which otherwise doubled the tile's height (most visible at
    // two-up on mobile). Content owns the padding: tighter on mobile, roomier up.
    <Card.Content className="flex flex-col gap-1 p-4 sm:p-5">
      <span className="text-overline">{label}</span>
      <span className="flex flex-wrap items-center gap-2">
        {/* text-xl (22px), deliberately a step *below* the page title's
            text-display (26px). It used to be text-xl rising to text-2xl at
            `sm`, which made a number inside a card the largest text on the
            page, 4px bigger than the name of the page itself. Fixing that by
            raising the title alone would have left the two agreeing by
            coincidence; both halves move so the ladder is correct by
            construction. `tabular-nums` so a column of these aligns. */}
        <span className="text-xl font-semibold tabular-nums text-foreground">
          {value}
        </span>
        {status && statusLabel ? (
          // `soft` and `sm` are TrendChip's defaults too, so a tile carrying
          // both (the error-rate tile carries a status word and a delta) draws
          // one shape at one weight rather than two.
          <Chip variant="soft" color={STAT_STATUS[status].chip} size="sm">
            <Chip.Label>{statusLabel}</Chip.Label>
          </Chip>
        ) : null}
      </span>
      {/* One row under the value carrying both the movement (the chip) and the
          composition (the hint): they are two halves of the same aside, and
          stacked they read as two, costing a line the tile does not have.
          `flex` also makes the chip hug its text, which it does not do as a
          direct child of this column: HeroUI's Chip is inline-flex, and a
          stretched child would run the pill the full width of the tile.

          `flex-wrap` because at five-up the pair does not always fit; the wrap
          is the fallback, not the layout. `items-center` aligns the hint's
          x-height to the middle of the pill rather than to its box.

          The wrap is reserved for so it does not misalign a row of tiles: one
          tile wrapping while its neighbors stay on one line would move its
          sparkline up relative to theirs. min-h-10.5 is 42px, which is what the
          wrap costs: a 20px chip line, `gap-y-1`, and one 18px hint line. The
          chip's line is 20px and not the 18px its `text-xs` implies, because
          HeroUI's `.chip` sets `--tw-leading` to `leading-5` on the element and
          `.chip--sm` does not reset it, so the size modifier's
          `line-height: var(--tw-leading, var(--text-xs--line-height))` resolves
          to the base 20px rather than to our token. #807 reserved 36px here for
          two lines of plain text, which that overruns. A hint long enough to
          wrap on its own still overruns this, so it is a floor and not a
          guarantee; the type is deliberately not sized to the longest string,
          which would be sizing it by accident. Reserved for a charted tile even
          with neither chip nor hint, since the chart is what makes the
          misalignment visible; a tile with no chart and nothing to say reserves
          nothing, so a lone tile carries no dead space. */}
      {trend || hint || chart ? (
        <span className="flex min-h-10.5 flex-wrap items-center gap-x-2 gap-y-1 text-xs tabular-nums text-muted">
          {trend}
          {/* Its own element, not a bare text node beside the chip: the two
              are separate statements, and a node keeps the hint addressable
              (by a test, and by a reader's selection) rather than merged into
              the chip's text. */}
          {hint ? <span>{hint}</span> : null}
        </span>
      ) : null}
      {chart ? <div className="mt-2">{chart}</div> : null}
    </Card.Content>
  )
  // min-w-0 (not a fixed min) so the tile fits its grid track: with
  // grid-cols-2's minmax(0,1fr) columns, a larger min-width would overflow the
  // track and overlap the neighboring tile on narrow (mobile) viewports.
  const cardClass = `flex-1 min-w-0 p-0 ${accent}`
  if (to) {
    return (
      <Card
        // Same reason as the accent above: the foundation's card outline is
        // unlayered, so the hover tint needs the bang to be seen at all.
        className={`${cardClass} transition-colors hover:border-accent!`}
      >
        <Link to={to} className="block rounded-[inherit]">
          {body}
        </Link>
      </Card>
    )
  }
  return <Card className={cardClass}>{body}</Card>
}

export function errorMessage(error: unknown): string {
  if (error instanceof ApiError) {
    return error.message
  }
  if (error instanceof Error) {
    return error.message
  }
  return "Something went wrong."
}

export function ErrorBanner({ error }: { error: unknown }) {
  if (!error) {
    return null
  }
  return (
    <div
      role="alert"
      className="rounded-lg border border-danger bg-danger-subtle px-4 py-3 text-sm text-danger"
    >
      {errorMessage(error)}
    </div>
  )
}

export function InfoBanner({
  tone = "info",
  children,
}: {
  tone?: "info" | "warning"
  children: ReactNode
}) {
  // A fact stated between rules, not a tinted box. An informational banner here
  // is almost always a ceiling ("this deployment has no sandbox", "an admin sets
  // this"), which is a fact about the deployment rather than a problem with it,
  // so it reads on the muted rung behind a subtle dot. A caution keeps the
  // danger dot and the same muted prose: the dot says "worth noticing" and the
  // words say what.
  return (
    <div className="flex items-start gap-3 border-y border-border py-3 text-sm text-muted">
      <Dot
        className={`mt-2 ${tone === "warning" ? "bg-danger" : "bg-surface-subtle"}`}
      />
      <div className="min-w-0">{children}</div>
    </div>
  )
}

export function PageHeader({
  title,
  description,
  action,
}: {
  title: string
  description?: string
  action?: ReactNode
}) {
  return (
    <div className="flex flex-col gap-3">
      <div>
        <h1 className="text-display">{title}</h1>
        {description ? (
          // max-w-prose because this ran to the full container width: 968px at
          // 14px is ~138 characters per line, roughly twice a comfortable
          // measure, and it is the same paragraph on every page.
          <p className="mt-1 max-w-prose text-sm text-muted">{description}</p>
        ) : null}
      </div>
      {/* The primary action sits on its own left-aligned row under the heading,
          so it stays near the sidebar the operator just came from rather than
          across the page at the top right. Wrapped so the button keeps its
          natural size instead of stretching in this flex column. */}
      {action ? <div className="flex flex-wrap gap-2">{action}</div> : null}
    </div>
  )
}

// A slowly ticking wall clock, only to keep a relative "updated Ns ago" label
// current between renders. This is a display timer, not the prohibited data
// polling (which belongs on a TanStack Query `refetchInterval`): it fetches
// nothing. Paused while the tab is hidden so a backgrounded dashboard is idle.
function useDisplayClock(intervalMs: number): number {
  const [now, setNow] = useState(() => Date.now())
  useEffect(() => {
    let timer: ReturnType<typeof setInterval> | undefined
    const start = () => {
      if (timer === undefined) {
        timer = setInterval(() => setNow(Date.now()), intervalMs)
      }
    }
    const stop = () => {
      if (timer !== undefined) {
        clearInterval(timer)
        timer = undefined
      }
    }
    const sync = () => {
      setNow(Date.now())
      if (document.visibilityState === "visible") start()
      else stop()
    }
    sync()
    document.addEventListener("visibilitychange", sync)
    return () => {
      stop()
      document.removeEventListener("visibilitychange", sync)
    }
  }, [intervalMs])
  return now
}

// A refresh control paired with a "last updated" timestamp, so an operator can
// tell stale numbers from fresh ones. The icon spins while a refetch is in
// flight. `updatedAt` is a TanStack Query `dataUpdatedAt` (ms epoch; 0 before
// the first successful load, which reads as "never" and is hidden).
export function RefreshButton({
  onRefresh,
  isFetching = false,
  updatedAt,
  label = "Refresh",
}: {
  onRefresh: () => void
  isFetching?: boolean
  updatedAt?: number
  label?: string
}) {
  const now = useDisplayClock(15_000)
  const freshness = updatedAt
    ? formatRelative(new Date(updatedAt).toISOString(), now)
    : null
  return (
    <span className="inline-flex items-center gap-2">
      {freshness ? (
        <span className="text-xs text-muted">Updated {freshness}</span>
      ) : null}
      <Button
        variant="outline"
        size="sm"
        isIconOnly
        isDisabled={isFetching}
        onPress={onRefresh}
        aria-label={label}
      >
        <svg
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          className={`h-4 w-4 ${isFetching ? "animate-spin" : ""}`}
          aria-hidden="true"
        >
          <path
            d="M20 11a8 8 0 1 0-.5 4"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
          <path d="M20 4v5h-5" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
      </Button>
    </span>
  )
}

// An identifier the operator needs verbatim (a model id, an alias target, a
// request id), rendered so it can be taken either way: highlighted with the mouse
// like ordinary text, or copied in one press.
//
// Highlighting is what needs the help. Inside a react-aria table the row is a
// press target, and a press on it both toggles the row's selection and sets
// `user-select: none` on the row for the duration; the re-render that selection
// causes lands mid-drag and discards the selection the browser had started, so
// dragging across an id used to select nothing at all (issue #478). Keeping the
// press from starting on the value itself is the fix: the pointer sequence stays
// with the browser, which selects text with it. `select-text` then beats the
// inherited `none` from any press elsewhere in the row (an own declaration
// outranks inheritance, so no `!important` is needed). The rest of the row keeps
// its behavior: DataTable still opens a drill-in for a plain click, including one
// on this value, and skips it for the click that ends a drag.
export function CopyableValue({
  value,
  label,
  className,
  children,
}: {
  /** The exact text a copy yields, which is not always what is rendered. */
  value: string
  label: string
  className?: string
  /** Defaults to `value`; pass children when the display form differs. */
  children?: ReactNode
}) {
  const keepPressFromRow = (event: { stopPropagation: () => void }) =>
    event.stopPropagation()
  return (
    <span className="inline-flex items-center gap-1">
      {/* biome-ignore lint/a11y/noStaticElementInteractions: the handlers only stop propagation so a text drag survives; there is no action to expose */}
      <span
        // Focusable, but not tabbable: pressing here focuses the value itself
        // instead of the react-aria table cell, whose focus bookkeeping re-renders
        // the row and (again) discards a drag that has only just begun. Without
        // this, the first drag in a freshly loaded table selected nothing and only
        // subsequent ones worked.
        tabIndex={-1}
        className={`select-text outline-none ${className ?? ""}`}
        onPointerDown={keepPressFromRow}
        onMouseDown={keepPressFromRow}
      >
        {children ?? value}
      </span>
      <CopyButton value={value} label={label} />
    </span>
  )
}

// A readonly, always-selectable field with a copy button: how a value an
// operator has to paste elsewhere is handed over. Shared by the Keys page's
// one-time reveal and the setup guide, which hand out the same key and the same
// snippets.
//
// The Clipboard API is undefined on the non-secure origins this dashboard is
// routinely served from, so the text is selected on click and Ctrl/Cmd-C always
// works even when the button cannot copy programmatically. "Copied" is only
// claimed when it truly copied.
//
// The label is a real `<label>` for the field, not a caption beside it: these
// values are handed over in pairs and threes (a key and two snippets), so
// "which field is this" has to be answerable by a screen reader and by a test
// that queries the way an operator reads.
export function CopyField({
  label,
  value,
  multiline = false,
  fieldRef,
}: {
  label: string
  value: string
  multiline?: boolean
  fieldRef?: React.RefObject<HTMLInputElement | HTMLTextAreaElement | null>
}) {
  const internalRef = useRef<HTMLInputElement | HTMLTextAreaElement | null>(
    null,
  )
  const ref = fieldRef ?? internalRef
  const fieldId = useId()
  const [copied, setCopied] = useState(false)
  const [selectHint, setSelectHint] = useState(false)
  // Same shape as CopyButton's below: the acknowledgement clears itself on a
  // timer, so the timer has to die with the component (and be replaced rather
  // than stacked when a second copy lands inside the window).
  const resetTimer = useRef<ReturnType<typeof setTimeout> | undefined>(
    undefined,
  )

  useEffect(() => () => clearTimeout(resetTimer.current), [])

  const copy = async () => {
    ref.current?.focus()
    ref.current?.select()
    try {
      if (navigator.clipboard?.writeText) {
        await navigator.clipboard.writeText(value)
        setCopied(true)
        setSelectHint(false)
        clearTimeout(resetTimer.current)
        resetTimer.current = setTimeout(() => setCopied(false), 2_000)
        return
      }
    } catch {
      // fall through to the manual path
    }
    // No Clipboard API (or it threw): the text is selected, so the operator can
    // press Ctrl/Cmd-C. Never claim it was copied.
    setSelectHint(true)
  }

  const shared =
    "w-full rounded-lg border border-border bg-surface-alt px-3 py-2 font-mono text-xs text-foreground"

  return (
    <div className="flex flex-col gap-1">
      <div className="flex items-center justify-between">
        <label htmlFor={fieldId} className="text-xs font-medium text-muted">
          {label}
        </label>
        <Button size="sm" variant="outline" onPress={copy}>
          {copied ? "Copied" : "Copy"}
        </Button>
      </div>
      {multiline ? (
        <textarea
          id={fieldId}
          ref={ref as React.RefObject<HTMLTextAreaElement>}
          readOnly
          rows={value.split("\n").length}
          value={value}
          onFocus={(e) => e.currentTarget.select()}
          className={`${shared} resize-none whitespace-pre`}
        />
      ) : (
        <input
          id={fieldId}
          ref={ref as React.RefObject<HTMLInputElement>}
          readOnly
          value={value}
          onFocus={(e) => e.currentTarget.select()}
          className={shared}
        />
      )}
      {/* Announce only the "Copied" event, never the secret itself. */}
      <span aria-live="polite" className="text-xs text-success">
        {copied ? "Copied to clipboard." : ""}
      </span>
      {selectHint ? (
        <span className="text-xs text-muted">
          Selected. Press Ctrl/Cmd-C to copy.
        </span>
      ) : null}
    </div>
  )
}

// A compact copy control for an identifier an operator has to paste elsewhere (a
// model id, an alias target). Table rows own click-drag for selection, so the
// text in a cell cannot be highlighted by hand (issue #478); this is how it gets
// out. copyToClipboard covers the plain-HTTP origins this dashboard is routinely
// served from, where the async Clipboard API does not exist; if even the legacy
// path fails, this says so rather than claiming a copy it did not make (the same
// rule as the Keys page's CopyField). Unlike that one, the value here is not a
// form field this can select for the operator, so the failure message asks them
// to select it rather than implying something already is.
// The acknowledgement is a tooltip over the icon that was pressed, so the answer
// appears where the operator is looking in a column of identical buttons. It is
// controlled (never hover-opened) because it reports an event, not a hint, and it
// renders in an overlay so it is not clipped by the table's scroll container and
// does not reflow the row it reports on.
export function CopyButton({ value, label }: { value: string; label: string }) {
  const [state, setState] = useState<"idle" | "copied" | "failed">("idle")
  const resetTimer = useRef<ReturnType<typeof setTimeout> | undefined>(
    undefined,
  )

  useEffect(() => () => clearTimeout(resetTimer.current), [])

  const copy = async () => {
    const copied = await copyToClipboard(value)
    setState(copied ? "copied" : "failed")
    clearTimeout(resetTimer.current)
    // A failure has something to read and act on, so it lingers longer.
    resetTimer.current = setTimeout(
      () => setState("idle"),
      copied ? 1_500 : 5_000,
    )
  }

  return (
    <Tooltip.Root isOpen={state !== "idle"}>
      <Button
        size="sm"
        variant="ghost"
        isIconOnly
        aria-label={`Copy ${label}`}
        onPress={copy}
      >
        <svg
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          className="h-3.5 w-3.5"
          aria-hidden="true"
        >
          <rect x="9" y="9" width="11" height="11" rx="2" />
          <path
            d="M5 15V5a2 2 0 0 1 2-2h8"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        </svg>
      </Button>
      <Tooltip.Content placement="top" showArrow>
        {state === "failed"
          ? "Copy blocked, select the value and press Ctrl/Cmd-C"
          : "Copied!"}
      </Tooltip.Content>
    </Tooltip.Root>
  )
}

// A first-run / empty-list panel: a Card with a heading, a sentence of context,
// and (optionally) a primary call to action. Pages that render a list share this
// so an empty Keys, Users, or Budgets page reads the same way instead of each
// hand-rolling the same Card. `children` slots richer content (e.g. a numbered
// getting-started list) between the copy and the action; omit the action for a
// purely informational empty state (e.g. "no usage yet").
export function EmptyState({
  title,
  description,
  actionLabel,
  onAction,
  isActionDisabled,
  children,
}: {
  title: string
  // A plain sentence, rendered in a <p>. Kept to a string (like PageHeader) so a
  // block element can't land inside that paragraph; richer/blockish content goes
  // through `children`, which renders as a sibling instead.
  description?: string
  actionLabel?: string
  onAction?: () => void
  isActionDisabled?: boolean
  children?: ReactNode
}) {
  return (
    <Card>
      <Card.Content className="flex flex-col gap-4 p-6">
        <div>
          <h2 className="text-heading">{title}</h2>
          {description ? (
            <p className="mt-1 max-w-prose text-sm text-muted">{description}</p>
          ) : null}
        </div>
        {children}
        {actionLabel && onAction ? (
          <div>
            <Button
              variant="primary"
              isDisabled={isActionDisabled}
              onPress={onAction}
            >
              {actionLabel}
            </Button>
          </div>
        ) : null}
      </Card.Content>
    </Card>
  )
}

// A full-width loading placeholder for a page (or section) whose content is
// gated on a first fetch. Without it, config pages that render nothing until
// `data` arrives (Settings, Tools & Guardrails, the Overview index) flash a bare
// header over blank space, which reads as broken. `role="status"` announces the
// wait (and its label) to assistive tech.
export function PageLoading({ label = "Loading…" }: { label?: string }) {
  return (
    <div
      role="status"
      className="flex items-center justify-center gap-2 px-4 py-10 text-sm text-muted"
    >
      {/* The spinner is its own role="status" live region as of HeroUI 3.2.4,
          which would nest one status inside another and announce a bare
          "Loading" alongside this label. Hide it; the label below carries the
          announcement for the region. */}
      <Spinner size="sm" aria-hidden="true" />
      <span>{label}</span>
    </div>
  )
}

/**
 * A destination this build declares but does not serve.
 *
 * The shell already answers a *gated-off* registered path with its own panel, so
 * in a standalone gateway this never renders: the surface is absent, the sidebar
 * drops the link, and the shell intercepts the route. It is what a deployment
 * that reports the surface but has not composed the overlay serving it would
 * see, which is the one case that would otherwise paint a blank page.
 */
export function UnavailableHere({ title }: { title: string }) {
  return (
    <EmptyState
      title={`${title} is not available here`}
      description="This deployment declares the page but does not serve it. Pick a destination from the sidebar."
    />
  )
}

// A destructive button that requires a second click to confirm, avoiding a
// modal dependency for revoke/delete actions.
export function ConfirmButton({
  children,
  confirmLabel,
  onConfirm,
  isPending,
}: {
  children: ReactNode
  confirmLabel: string
  onConfirm: () => void
  isPending?: boolean
}) {
  const [armed, setArmed] = useState(false)

  if (armed) {
    return (
      <span className="inline-flex items-center gap-1">
        <Button
          size="sm"
          variant="danger"
          isDisabled={isPending}
          onPress={onConfirm}
        >
          {confirmLabel}
        </Button>
        <Button
          size="sm"
          variant="ghost"
          isDisabled={isPending}
          onPress={() => setArmed(false)}
        >
          Cancel
        </Button>
      </span>
    )
  }

  return (
    <Button size="sm" variant="danger-soft" onPress={() => setArmed(true)}>
      {children}
    </Button>
  )
}

/** The bare text/password input the feature cards use where a HeroUI field is too much. */
export const INPUT_CLASS =
  "rounded-md border border-border bg-surface px-2 py-1 text-sm focus:border-accent focus:outline-none disabled:opacity-50"

/**
 * A small status pill for a row that has something to say about itself.
 *
 * Two tones and no more: `muted` states a fact about the row (which provider,
 * which scope, whether a credential is set), `warn` says the row is not doing
 * what its neighbors are. Lives here rather than in a feature folder because
 * the Tools cards each grew an identical copy.
 */
export function Badge({
  tone,
  children,
}: {
  tone: "muted" | "warn"
  children: string
}) {
  const className =
    tone === "warn"
      ? "border-warning bg-warning-subtle text-warning"
      : "border-border bg-surface text-muted"
  return (
    <span
      className={`rounded-full border px-2 py-0.5 text-xs font-medium ${className}`}
    >
      {children}
    </span>
  )
}

// react-aria reads an empty key as "nothing selected", and "" is a real filter
// value here ("All", "Any price"), so every option key carries this prefix and
// it is stripped back off on the way out. Both directions go through these two,
// so the prefix is written down once.
const OPTION_KEY_PREFIX = "v:"
const optionKey = (value: string) => `${OPTION_KEY_PREFIX}${value}`
const optionValue = (key: string) => key.slice(OPTION_KEY_PREFIX.length)

// Filter dropdown for page filter bars. On HeroUI's Select rather than a native
// <select> because a native one draws its menu *over* the control on macOS,
// covering the button that opened it; this one is a popover anchored under the
// trigger. Pass `label` for a visible label, or `ariaLabel` alone for a compact
// control, and `id` when an outside <label htmlFor> points at the trigger.
export function FilterSelect({
  id,
  label,
  ariaLabel,
  value,
  onChange,
  options,
  disabled,
}: {
  id?: string
  label?: string
  ariaLabel?: string
  value: string
  onChange: (value: string) => void
  options: { value: string; label: string }[]
  disabled?: boolean
}) {
  // A value no option carries is a URL naming something the list does not hold
  // (`/activity?status=bogus`) or a drill-down into a key with no rows in the
  // window. react-aria answers an unmatched key with its own "Select an item",
  // which would put library boilerplate where the applied filter belongs, so
  // the value is carried as its own option instead: the filter bar says what is
  // actually filtering, the same fallback the pages' own chips make with
  // `?? value`. A call site whose default is missing from its own options would
  // land here too, which is a bug at the call site rather than a shape to
  // design around; every list today carries its own.
  const items = options.some((option) => option.value === value)
    ? options
    : [{ value, label: value }, ...options]
  return (
    <Select.Root
      aria-label={label ? undefined : ariaLabel}
      isDisabled={disabled}
      selectedKey={optionKey(value)}
      // A null key is react-aria clearing the selection, which no filter here
      // asks for: reporting it would push the strip of a non-string ("ll") into
      // the filter, and on a URL-backed page into the query string with it.
      onSelectionChange={(key) => {
        if (key != null) onChange(optionValue(String(key)))
      }}
    >
      {label ? (
        <Label className="text-xs font-medium text-muted">{label}</Label>
      ) : null}
      <Select.Trigger id={id}>
        <Select.Value />
        <Select.Indicator />
      </Select.Trigger>
      <Select.Popover>
        <ListBox items={items} className="max-h-72 overflow-auto">
          {(option: { value: string; label: string }) => (
            <ListBoxItem id={optionKey(option.value)} textValue={option.label}>
              {option.label}
            </ListBoxItem>
          )}
        </ListBox>
      </Select.Popover>
    </Select.Root>
  )
}

// A type-to-filter combobox for page filter bars, accumulating a set of values.
// The option list is large (users, models), so a native <select> with thousands of
// <option>s is unusable and this narrows it as you type; and the question a usage
// view answers is usually a comparison ("these three models"), not a single choice.
// Picking an option adds it and clears the query, and the list stays open on the remaining
// options so a run of selections takes one gesture; picked options drop out of it.
// Removal lives with the page's filter chips (one per value) rather than a second
// chip row here, so the applied set is visible whether or not the picker is open.
// Dismiss the list (Escape, or a click outside) before reaching for the page
// behind it: it is an overlay, so it holds focus while open.
export function FilterMultiComboBox({
  label,
  values,
  onChange,
  options,
  placeholder,
  maxVisible = 50,
  maxValues = 50,
  allowsCustom = false,
}: {
  label: string
  values: string[]
  onChange: (values: string[]) => void
  options: { value: string; label: string }[]
  // Shown while nothing is picked (e.g. "All users"); once something is, the
  // input reports the size of the selection instead.
  placeholder?: string
  maxVisible?: number
  // Ceiling on the selection, matching what the analytics endpoints accept for
  // one repeatable filter. Stopping here keeps a 51st pick from failing every
  // query on the page with a 422 the operator cannot read.
  maxValues?: number
  // When true, Enter adds whatever was typed, so a filter whose value space is not
  // enumerable (any model name the log might hold, not just the ones a windowed
  // suggestion list knows) can still be filtered on. The options stay suggestions.
  allowsCustom?: boolean
}) {
  const [text, setText] = useState("")

  const atLimit = values.length >= maxValues
  const query = text.trim().toLowerCase()
  const visible = options
    .filter((o) => !values.includes(o.value))
    .filter(
      (o) =>
        !query ||
        o.value.toLowerCase().includes(query) ||
        o.label.toLowerCase().includes(query),
    )
    .slice(0, maxVisible)

  const add = (value: string) => {
    if (atLimit || values.includes(value)) return
    onChange([...values, value])
  }

  // Free text commits on Enter: react-aria fires no selection for a value that is
  // not in the list, so the key event is the only signal. Skipped while an option is
  // highlighted (aria-activedescendant), because that Enter belongs to the option
  // and committing the partial query beside it would add two values from one press.
  const onInputKeyDown = (event: ReactKeyboardEvent<HTMLInputElement>) => {
    if (!allowsCustom || event.key !== "Enter") return
    if (event.currentTarget.getAttribute("aria-activedescendant")) return
    const typed = text.trim()
    if (!typed) return
    add(typed)
    setText("")
  }

  return (
    <ComboBox.Root
      allowsEmptyCollection
      allowsCustomValue={allowsCustom}
      menuTrigger="focus"
      inputValue={text}
      onInputChange={setText}
      // Never a committed selection of its own: the picked values live in
      // `values`, so the input stays a search box.
      selectedKey={null}
      // At the ceiling the remaining options are offered but inert, so the list
      // reads as "full" rather than silently swallowing a click.
      disabledKeys={atLimit ? visible.map((o) => o.value) : []}
      onSelectionChange={(key) => {
        if (key == null) return
        add(String(key))
        setText("")
      }}
      className="flex flex-col gap-1"
    >
      <Label className="text-xs font-medium text-muted">{label}</Label>
      <ComboBox.InputGroup>
        <Input
          placeholder={
            values.length === 0
              ? placeholder
              : `${values.length} selected${atLimit ? " (max)" : ""}`
          }
          autoComplete="off"
          onKeyDown={onInputKeyDown}
        />
        <ComboBox.Trigger />
      </ComboBox.InputGroup>
      <ComboBox.Popover>
        <ListBox items={visible} className="max-h-72 overflow-auto">
          {(option: { value: string; label: string }) => (
            <ListBoxItem id={option.value} textValue={option.label}>
              {option.label}
            </ListBoxItem>
          )}
        </ListBox>
      </ComboBox.Popover>
    </ComboBox.Root>
  )
}
