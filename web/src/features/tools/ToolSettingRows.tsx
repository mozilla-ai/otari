import { useId, useState } from "react"
import type { ToolSettingField } from "@/client"
import { Button } from "@/design-system/actions/Button"
import { errorMessage } from "@/design-system/feedback/errorMessage"
import { INPUT_CLASS } from "@/design-system/forms/inputClass"
import { SettingRow } from "@/design-system/layout/SettingRow"
import { FilterSelect } from "@/design-system/navigation/FilterSelect"
import { useTestService } from "@/shared/api/tools"
import { commitOnEnter, useAutosave } from "@/shared/hooks/useAutosave"

// Every field fills `SettingRow`'s lane rather than picking a width of its own,
// which is what keeps the column's edges straight whether a row holds a URL, a
// select or a two-digit number. `flex-1` rather than `w-full` because a field
// can share the lane with a trailing button, and `scroll-mt-16` so a jump from
// the tool panel lands it in the page rather than under the 56px top bar.
const TEXT_INPUT = `min-w-0 flex-1 scroll-mt-16 ${INPUT_CLASS}`
const MACHINE_INPUT = `otari-machine-field ${TEXT_INPUT}`
const NUMBER_INPUT = `otari-machine-field w-full scroll-mt-16 text-right tabular-nums ${INPUT_CLASS}`

/**
 * The mono caption beside a row's label, if the key is not already the label.
 *
 * A key with no copy entry falls back to using itself as the label, and showing
 * it twice would read "web_search_url web_search_url".
 */
function keyCaption(copy: FieldCopy, field: ToolSettingField) {
  return copy.label === field.key ? undefined : field.key
}

/**
 * The ids a row's control names itself by: the label, and the key caption when
 * there is one. "Backend URL" alone is not unique on a page configuring three
 * services, and an accessible name has to contain the visible label.
 */
function namedBy(labelId: string, caption: string | undefined) {
  return caption ? `${labelId} ${labelId}-key` : labelId
}

/** The field a "jump to this setting" link scrolls to and focuses. */
export function settingInputId(key: string) {
  return `tool-setting-${key}`
}

/** What a row writes, and what the page does with it. */
export type CommitField = (
  key: string,
  value: boolean | number | string | null,
) => Promise<unknown>

/** The label, help line and placeholder a key wears in the dashboard. */
export interface FieldCopy {
  label: string
  help: string
  /** A representative value, never the word "default": blank already means that. */
  placeholder: string
  /** Mono, for a value a machine reads. Off for a sentence a model reads. */
  machine?: boolean
}

function useDraft(committed: string) {
  const [draft, setDraft] = useState(committed)
  const [synced, setSynced] = useState(committed)
  // Re-hydrated from the server's answer, so a save or a clear lands in the
  // field instead of leaving a stale draft over it. In render rather than an
  // effect, which is the idiom `PolicyRow` uses: one rule for the job.
  if (committed !== synced) {
    setSynced(committed)
    setDraft(committed)
  }
  return [draft, setDraft] as const
}

function TextRow({
  field,
  copy,
  commit,
  disabled,
  trailing,
  note,
  onDraftChange,
}: {
  field: ToolSettingField
  copy: FieldCopy
  commit: CommitField
  disabled: boolean
  trailing?: React.ReactNode
  note?: React.ReactNode
  onDraftChange?: (value: string) => void
}) {
  const committed = typeof field.value === "string" ? field.value : ""
  const [draft, setDraft] = useDraft(committed)
  const save = useAutosave()
  const errorId = useId()
  const labelId = useId()
  const configKey = keyCaption(copy, field)

  return (
    <SettingRow
      label={copy.label}
      labelId={labelId}
      controlId={settingInputId(field.key)}
      configKey={configKey}
      help={copy.help}
      note={note}
      error={save.error}
      errorId={errorId}
      control={
        <div className="flex items-center gap-1.5">
          <input
            id={settingInputId(field.key)}
            type="text"
            inputMode={field.type === "url" ? "url" : "text"}
            aria-labelledby={namedBy(labelId, configKey)}
            aria-invalid={save.error ? true : undefined}
            aria-describedby={save.error ? errorId : undefined}
            value={draft}
            disabled={disabled || save.isSaving}
            placeholder={copy.placeholder}
            onChange={(event) => {
              setDraft(event.target.value)
              onDraftChange?.(event.target.value)
            }}
            onKeyDown={commitOnEnter}
            onBlur={() => {
              const next = draft.trim()
              if (next === committed) return
              void save.run(() => commit(field.key, next === "" ? null : next))
            }}
            className={`${copy.machine ? MACHINE_INPUT : TEXT_INPUT}`}
          />
          {trailing}
        </div>
      }
    />
  )
}

function NumberRow({
  field,
  copy,
  commit,
  disabled,
}: {
  field: ToolSettingField
  copy: FieldCopy
  commit: CommitField
  disabled: boolean
}) {
  const committed = typeof field.value === "number" ? String(field.value) : ""
  const [draft, setDraft] = useDraft(committed)
  const save = useAutosave()
  const errorId = useId()
  const labelId = useId()
  const configKey = keyCaption(copy, field)

  const trimmed = draft.trim()
  // Digits only. `Number` would read "0x10" as 16 and "1e1" as 10, and the
  // server's floor is 1, so both are refused here rather than sent.
  const parsed = /^\d+$/.test(trimmed) ? Number(trimmed) : Number.NaN
  const valid = trimmed === "" || (Number.isSafeInteger(parsed) && parsed >= 1)
  const message = save.error || (valid ? "" : "A whole number, 1 or more.")

  return (
    <SettingRow
      label={copy.label}
      labelId={labelId}
      controlId={settingInputId(field.key)}
      configKey={configKey}
      help={copy.help}
      error={message}
      errorId={errorId}
      control={
        <input
          id={settingInputId(field.key)}
          type="text"
          inputMode="numeric"
          aria-labelledby={namedBy(labelId, configKey)}
          aria-invalid={message ? true : undefined}
          aria-describedby={message ? errorId : undefined}
          value={draft}
          disabled={disabled || save.isSaving}
          placeholder={copy.placeholder}
          onChange={(event) => setDraft(event.target.value)}
          onKeyDown={commitOnEnter}
          onBlur={() => {
            if (!valid || trimmed === committed) return
            void save.run(() =>
              commit(field.key, trimmed === "" ? null : parsed),
            )
          }}
          className={`${NUMBER_INPUT}`}
        />
      }
    />
  )
}

// A nullable boolean has three meaningful states, so a select rather than a
// toggle: "Default" is what the backend decides, and a two-state control cannot
// say it. Discrete, so it saves on change with nothing typed to lose.
function BoolRow({
  field,
  copy,
  commit,
  disabled,
  defaultLabel,
}: {
  field: ToolSettingField
  copy: FieldCopy
  commit: CommitField
  disabled: boolean
  defaultLabel: string
}) {
  const save = useAutosave()
  const errorId = useId()
  const configKey = keyCaption(copy, field)
  const current =
    field.value === true ? "on" : field.value === false ? "off" : "default"

  return (
    <SettingRow
      label={copy.label}
      configKey={configKey}
      help={copy.help}
      error={save.error}
      errorId={errorId}
      control={
        <FilterSelect
          fullWidth
          ariaLabel={configKey ? `${copy.label} ${configKey}` : copy.label}
          value={current}
          onChange={(next) =>
            void save.run(() =>
              commit(field.key, next === "default" ? null : next === "on"),
            )
          }
          options={[
            { value: "default", label: defaultLabel },
            { value: "on", label: "On" },
            { value: "off", label: "Off" },
          ]}
          disabled={disabled || save.isSaving}
        />
      }
    />
  )
}

// The URL row carries Test, which probes the *typed* value so an operator can
// check an endpoint before leaving the field. The result is pinned to the URL
// it was asked about, so a late answer never lands beside a different one.
function UrlRow({
  field,
  copy,
  commit,
  disabled,
}: {
  field: ToolSettingField
  copy: FieldCopy
  commit: CommitField
  disabled: boolean
}) {
  const committed = typeof field.value === "string" ? field.value : ""
  const [typed, setTyped] = useState(committed)
  const [seen, setSeen] = useState(committed)
  const [testedUrl, setTestedUrl] = useState<string | null>(null)
  const test = useTestService()

  // A refetch can re-seed the field with no keystroke, which the `onChange`
  // reset below never sees. Following it here is what stops a result for the
  // old URL from sitting beside the new one.
  if (committed !== seen) {
    setSeen(committed)
    setTyped(committed)
  }

  const trimmed = typed.trim()
  // A result belongs to the URL it was asked about, so a late answer never
  // lands beside a different one.
  const settled = testedUrl === trimmed && !test.isPending

  return (
    <TextRow
      field={field}
      copy={copy}
      commit={commit}
      disabled={disabled}
      onDraftChange={(value) => {
        setTyped(value)
        // A result for the old URL must not sit beside a newly typed one.
        test.reset()
      }}
      note={
        // Always rendered so the outcome is announced when it arrives, not just
        // shown. Empty until there is one for the URL currently in the field.
        <p
          role="status"
          aria-live="polite"
          className={`text-caption ${test.data?.ok ? "text-success" : "text-danger"}`}
        >
          {settled &&
            (test.error ? errorMessage(test.error) : test.data?.reason)}
        </p>
      }
      trailing={
        <Button
          size="sm"
          // Wide enough for "Testing…" as well as "Test", which are 53px and
          // 84px apart: the button shares a fixed lane with the field now, so
          // letting it size to its label would narrow the URL under the cursor
          // for as long as a probe is in flight.
          className="min-w-[5.5rem] shrink-0"
          aria-label={`Test ${field.service}`}
          isDisabled={trimmed === "" || test.isPending}
          onPress={() => {
            setTestedUrl(trimmed)
            test.mutate({ service: field.service, url: trimmed })
          }}
        >
          {test.isPending ? "Testing…" : "Test"}
        </Button>
      }
    />
  )
}

/**
 * One editable tool setting, as the row grammar the whole page is built from.
 *
 * `readOnly` renders the value rather than a disabled control: a member does
 * not set this, and a dimmed field reads as a form that is briefly unavailable.
 */
export function ToolSettingRow({
  field,
  copy,
  commit,
  disabled,
  readOnly,
  defaultLabel = "Default",
}: {
  field: ToolSettingField
  copy: FieldCopy
  commit: CommitField
  disabled: boolean
  readOnly: boolean
  /** Names what the backend does when nothing is set ("Default (on)"). */
  defaultLabel?: string
}) {
  const configKey = keyCaption(copy, field)

  if (readOnly) {
    const shown =
      field.value === null || field.value === ""
        ? "Default"
        : field.value === true
          ? "On"
          : field.value === false
            ? "Off"
            : String(field.value)
    return (
      <SettingRow
        label={copy.label}
        configKey={configKey}
        help={copy.help}
        control={
          <span className="break-words text-caption text-foreground">
            {shown}
          </span>
        }
      />
    )
  }
  if (field.type === "url") {
    return (
      <UrlRow field={field} copy={copy} commit={commit} disabled={disabled} />
    )
  }
  if (field.type === "int") {
    return (
      <NumberRow
        field={field}
        copy={copy}
        commit={commit}
        disabled={disabled}
      />
    )
  }
  if (field.type === "bool") {
    return (
      <BoolRow
        field={field}
        copy={copy}
        commit={commit}
        disabled={disabled}
        defaultLabel={defaultLabel}
      />
    )
  }
  return (
    <TextRow field={field} copy={copy} commit={commit} disabled={disabled} />
  )
}

// The stored column is `input_price_per_million`, and `flat_request_cost` reads
// it as USD per *million* calls, so a cent a search is stored as 10000. Right
// for the wire, hostile at a keyboard: this row speaks dollars per call and
// does the conversion itself.
const PER_MILLION = 1_000_000

/** Per-call price for a tool Otari runs itself, in the same row grammar. */
export function ToolPriceRow({
  pricingKey,
  configured,
  commit,
  disabled,
  loadError,
}: {
  pricingKey: string
  configured: number | null
  commit: (perMillion: number) => Promise<unknown>
  disabled: boolean
  /** Set when the current rate could not be read, so nothing may overwrite it. */
  loadError?: string
}) {
  const committed = configured === null ? "" : String(configured / PER_MILLION)
  const [draft, setDraft] = useDraft(committed)
  const save = useAutosave()
  const errorId = useId()

  const trimmed = draft.trim()
  // Digits with an optional decimal part, spelled out rather than left to
  // `Number`, which reads "1e3" as 1000 and "0x10" as 16. The other numeric
  // rows refuse those; this one is money, so it admits a decimal point.
  const parsed = /^\d+(\.\d+)?$/.test(trimmed) ? Number(trimmed) : Number.NaN
  const invalid =
    trimmed !== "" && !(Number.isFinite(parsed) && parsed >= 0)
      ? "An amount in dollars, such as 0.01."
      : ""
  const message = loadError || save.error || invalid

  return (
    <SettingRow
      label="Price per call"
      configKey={pricingKey}
      help="Unpriced calls are recorded and billed nothing; refused when require_pricing is on."
      error={message}
      errorId={errorId}
      control={
        <div className="flex items-center gap-1.5">
          <span className="shrink-0 text-mono-overline text-subtle">USD</span>
          <input
            type="text"
            inputMode="decimal"
            aria-label={`Price per call for ${pricingKey}`}
            aria-invalid={message ? true : undefined}
            aria-describedby={message ? errorId : undefined}
            value={draft}
            disabled={disabled || save.isSaving}
            placeholder="0.00"
            onChange={(event) => setDraft(event.target.value)}
            onKeyDown={commitOnEnter}
            onBlur={() => {
              if (invalid || trimmed === committed) return
              // Blank cannot be sent: `/pricing` only writes a rate, so
              // there is no way to make a priced tool unpriced again from
              // here. Putting the stored value back says that without a
              // message that would nag on every pass through the field.
              if (trimmed === "") {
                setDraft(committed)
                return
              }
              // Rounded, because the wire value is per million: 0.07 * 1e6 is
              // 70000.00000000001 in binary floating point.
              void save.run(() => commit(Math.round(parsed * PER_MILLION)))
            }}
            className={`otari-machine-field min-w-0 flex-1 text-right tabular-nums ${INPUT_CLASS}`}
          />
        </div>
      }
    />
  )
}
