import {
  Button,
  Card,
  ComboBox,
  Input,
  Label,
  ListBox,
  ListBoxItem,
  Spinner,
  Tooltip,
} from "@heroui/react"
import type { LinkProps } from "@tanstack/react-router"
import { Link } from "@tanstack/react-router"
import type { KeyboardEvent as ReactKeyboardEvent, ReactNode } from "react"
import { useEffect, useId, useRef, useState } from "react"

import { ApiError } from "@/shared/api/client"
import { copyToClipboard } from "@/shared/helpers/clipboard"
import { formatPct, formatRelative } from "@/shared/helpers/format"

// A tile's attention status, on the foundation's three status roles. Color is
// never the only signal: a status tile also carries a word/icon via `statusLabel`.
export type StatStatus = "ok" | "warn" | "alert"

// The accent bar is `!`-important because the design foundation gives every
// HeroUI Card a 1px outline (`.card:not(.card--transparent)` in globals.css),
// and that rule is unlayered while a Tailwind utility sits in @layer utilities —
// unlayered always wins, whatever the specificity. Without the bang the
// shorthand `border:` resets this tile's left edge back to a hairline.
const STAT_STATUS: Record<StatStatus, { accent: string; pill: string }> = {
  ok: {
    accent: "border-l-success!",
    pill: "border-success bg-success-subtle text-success",
  },
  warn: {
    accent: "border-l-warning!",
    pill: "border-warning bg-warning-subtle text-warning",
  },
  alert: {
    accent: "border-l-danger!",
    pill: "border-danger bg-danger-subtle text-danger",
  },
}

export function StatCard({
  label,
  value,
  hint,
  status,
  statusLabel,
  chart,
  to,
}: {
  label: string
  value: ReactNode
  hint?: ReactNode
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
      <span className="text-xs font-medium uppercase tracking-wide text-muted">
        {label}
      </span>
      <span className="flex flex-wrap items-center gap-2">
        <span className="text-xl font-semibold text-foreground sm:text-2xl">
          {value}
        </span>
        {status && statusLabel ? (
          <span
            className={`inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-xs font-medium ${STAT_STATUS[status].pill}`}
          >
            {statusLabel}
          </span>
        ) : null}
      </span>
      {hint ? <span className="text-xs text-muted">{hint}</span> : null}
      {chart ? <div className="mt-2">{chart}</div> : null}
    </Card.Content>
  )
  // min-w-0 (not a fixed min) so the tile fits its grid track: with
  // grid-cols-2's minmax(0,1fr) columns, a larger min-width would overflow the
  // track and overlap the neighbouring tile on narrow (mobile) viewports.
  const cardClass = `flex-1 min-w-0 p-0 ${accent}`
  if (to) {
    return (
      <Card
        // Same reason as the accent above: the foundation's card outline is
        // unlayered, so the hover tint needs the bang to be seen at all.
        className={`${cardClass} transition-colors hover:border-accent!`}
      >
        <Link
          to={to}
          className="block rounded-[inherit] focus:outline-none focus-visible:ring-2 focus-visible:ring-accent"
        >
          {body}
        </Link>
      </Card>
    )
  }
  return <Card className={cardClass}>{body}</Card>
}

// Period-over-period change hint. Pairs an arrow glyph with the number (never hue
// alone) so direction reads without color. null hides it (no comparable previous).
export function DeltaHint({ fraction }: { fraction: number | null }) {
  if (fraction === null) return null
  const arrow = fraction > 0 ? "▲" : fraction < 0 ? "▼" : "•"
  return (
    <span className="text-muted">
      {arrow} {formatPct(Math.abs(fraction))} vs prev
    </span>
  )
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
  const styles =
    tone === "warning"
      ? "border-warning bg-warning-subtle text-warning"
      : "border-accent bg-primary-subtle text-primary-subtle-foreground"
  return (
    <div className={`rounded-lg border px-4 py-3 text-sm ${styles}`}>
      {children}
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
        <h1 className="text-xl font-semibold text-foreground">{title}</h1>
        {description ? (
          <p className="mt-1 text-sm text-muted">{description}</p>
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
          <h2 className="text-lg font-semibold text-foreground">{title}</h2>
          {description ? (
            <p className="mt-1 text-sm text-muted">{description}</p>
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
      <Spinner size="sm" />
      <span>{label}</span>
    </div>
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

const FILTER_SELECT_CLASS =
  "rounded-lg border border-border bg-surface-alt px-3 py-2 text-sm text-foreground focus:border-accent focus:outline-none"

// Token-styled native select for page filter bars. Pass `label` (+ `id`) for a
// visible label, or `ariaLabel` alone for a compact control. Prefer `options`
// for static lists; use `children` when options are grouped or conditional.
export function FilterSelect({
  id,
  label,
  ariaLabel,
  value,
  onChange,
  options,
  children,
  disabled,
}: {
  id?: string
  label?: string
  ariaLabel?: string
  value: string
  onChange: (value: string) => void
  options?: { value: string; label: string }[]
  children?: ReactNode
  disabled?: boolean
}) {
  const fallbackId = useId()
  const selectId = id ?? (label ? fallbackId : undefined)
  const select = (
    <select
      id={selectId}
      aria-label={label ? undefined : ariaLabel}
      value={value}
      disabled={disabled}
      onChange={(event) => onChange(event.target.value)}
      className={FILTER_SELECT_CLASS}
    >
      {options
        ? options.map((option) => (
            <option key={option.value} value={option.value}>
              {option.label}
            </option>
          ))
        : children}
    </select>
  )

  if (label) {
    return (
      <div className="flex flex-col gap-1">
        <label htmlFor={selectId} className="text-xs font-medium text-muted">
          {label}
        </label>
        {select}
      </div>
    )
  }
  return select
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
