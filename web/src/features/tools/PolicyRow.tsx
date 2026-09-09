import { useId, useState } from "react"

import { INPUT_CLASS } from "@/shared/components/forms/inputClass"
import { SettingRow } from "@/shared/components/layout/SettingRow"
import { commitOnEnter, useAutosave } from "@/shared/hooks/useAutosave"

/** What a typed value has to become before it can be sent, or why it cannot. */
export type Parse<T> = (raw: string) => { value: T; error: string }

/**
 * One typed row of a workspace policy: it holds a draft and commits the whole
 * policy when the field is left.
 *
 * Both policy groups are a stance select over a handful of these, and the
 * shapes they need differ only in how the value is parsed and how wide the
 * lane is. `ToolSettingRows` is the other row family and stays separate: those
 * are keyed on a backend field descriptor and carry a config key, a
 * reachability note and a Test button, none of which a policy row has.
 */
export function PolicyRow<T>({
  label,
  help,
  placeholder,
  committed,
  parse,
  commit,
  disabled,
  machine = false,
  numeric = false,
}: {
  label: string
  help: string
  /** A representative value, never the word "default": blank already means that. */
  placeholder: string
  committed: string
  parse: Parse<T>
  commit: (value: T) => Promise<unknown>
  disabled: boolean
  /** Mono, for a value a machine reads. Off for a sentence a model reads. */
  machine?: boolean
  /** A short right-aligned lane, for a number rather than a list or a phrase. */
  numeric?: boolean
}) {
  const [draft, setDraft] = useState(committed)
  const [synced, setSynced] = useState(committed)
  const save = useAutosave()
  const errorId = useId()
  const controlId = useId()

  // Re-hydrated from the server's answer rather than from an effect: the row is
  // remounted by its key when the workspace changes, and a save is the only
  // other thing that moves the stored value.
  if (committed !== synced) {
    setSynced(committed)
    setDraft(committed)
  }

  const parsed = parse(draft)
  const message = save.error || parsed.error

  return (
    <SettingRow
      label={label}
      controlId={controlId}
      help={help}
      error={message}
      errorId={errorId}
      control={
        <input
          id={controlId}
          type="text"
          inputMode={numeric ? "numeric" : undefined}
          aria-label={label}
          aria-invalid={message ? true : undefined}
          aria-describedby={message ? errorId : undefined}
          value={draft}
          disabled={disabled || save.isSaving}
          placeholder={placeholder}
          onChange={(event) => setDraft(event.target.value)}
          onKeyDown={commitOnEnter}
          onBlur={() => {
            if (parsed.error || draft.trim() === committed) return
            void save.run(() => commit(parsed.value))
          }}
          className={`w-full ${machine || numeric ? "otari-machine-field" : ""} ${
            numeric
              ? "text-right tabular-nums md:w-[5.5rem]"
              : "md:w-[13.75rem]"
          } ${INPUT_CLASS}`}
        />
      }
    />
  )
}

/** A whole number the server will accept, or the reason it will not. */
export function ceilingParser(max: number, unit: string): Parse<number | null> {
  return (raw) => {
    const trimmed = raw.trim()
    if (trimmed === "") return { value: null, error: "" }
    // Digits only, so `0x10` and `1e1` are refused rather than silently read as
    // 16 and 10 by `Number`.
    const parsed = /^\d+$/.test(trimmed) ? Number(trimmed) : Number.NaN
    if (!Number.isSafeInteger(parsed) || parsed <= 0 || parsed > max) {
      return {
        value: null,
        error: `A whole number of ${unit} from 1 to ${max}.`,
      }
    }
    return { value: parsed, error: "" }
  }
}

/** A phrase, or nothing. Blank clears the stored one. */
export const parsePhrase: Parse<string | null> = (raw) => ({
  value: raw.trim() === "" ? null : raw.trim(),
  error: "",
})
