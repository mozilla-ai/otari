import { Description, Label, TextArea, TextField } from "@heroui/react"
import { type ReactNode, useId, useState } from "react"
import { FiPlus, FiX } from "react-icons/fi"

import type { GuardrailParameterSpec } from "@/client"
import { Button } from "@/design-system/actions/Button"
import { IconButton } from "@/design-system/actions/IconButton"
import { Checkbox } from "@/design-system/forms/Checkbox"
import { FieldMessages } from "@/design-system/forms/FieldMessages"
import { INPUT_CLASS } from "@/design-system/forms/inputClass"
import { Segmented } from "@/design-system/navigation/Segmented"
import {
  type FieldOption,
  type JsonFieldSpec,
  jsonFieldSpec,
} from "@/features/guardrails/guardrailFieldSuggestions"
import { parameterLabel } from "@/features/guardrails/guardrailParameters"

// One JSON argument, edited as controls with the JSON itself beside them.
//
// The value stays a JSON string throughout, which is what the form above and
// `buildCreateKwargs` below it already hold. Every control writes the string
// back, so the two views are one value rather than two that have to be kept in
// step, and switching to JSON mid-edit shows exactly what will be sent.
//
// A value the controls cannot represent is not one they get to destroy: the
// Fields view refuses to render it and says so, leaving JSON as the way in.
// That is the whole safety property, and it is what makes a suggestion list
// that will go stale safe to ship.

type View = "fields" | "json"

interface Parsed {
  value: unknown
  error?: string
}

/** The argument's current value, or why it cannot be read. */
function parse(text: string): Parsed {
  const trimmed = text.trim()
  if (trimmed === "") return { value: undefined }
  try {
    return { value: JSON.parse(trimmed) }
  } catch {
    return { value: undefined, error: "Not valid JSON." }
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value)
}

/** What a value's own shape says about how it is edited. */
function shapeOf(value: unknown): JsonFieldSpec["kind"] | undefined {
  if (Array.isArray(value)) {
    return value.every((entry) => typeof entry === "string")
      ? "list"
      : undefined
  }
  if (isRecord(value)) return "map"
  return undefined
}

/** A plain value as one line of text, and back. */
function asText(value: unknown): string {
  if (typeof value === "string") return value
  return JSON.stringify(value) ?? ""
}

function fromText(text: string): unknown {
  const trimmed = text.trim()
  if (trimmed === "true") return true
  if (trimmed === "false") return false
  if (trimmed !== "" && Number.isFinite(Number(trimmed))) return Number(trimmed)
  return text
}

function serialize(value: unknown): string {
  return JSON.stringify(value, null, 2)
}

/** A row of the map editor: a key the operator invents and one plain value. */
function MapRows({
  value,
  disabled,
  onChange,
}: {
  value: Record<string, unknown>
  disabled: boolean
  onChange: (next: Record<string, unknown>) => void
}) {
  const entries = Object.entries(value)
  const rename = (from: string, to: string) => {
    // Rebuilt in order rather than deleted and re-added, so renaming a key does
    // not move its row to the bottom under the operator's cursor.
    onChange(
      Object.fromEntries(
        entries.map(([key, held]) => (key === from ? [to, held] : [key, held])),
      ),
    )
  }
  return (
    <div className="flex flex-col gap-2">
      {entries.map(([key, held], index) => (
        // Keyed by position: the key itself is what the operator is editing, so
        // keying on it remounts the input on every keystroke and loses focus.
        <div key={index} className="flex items-center gap-2">
          <input
            aria-label={`Key ${index + 1}`}
            value={key}
            disabled={disabled}
            placeholder="name"
            onChange={(event) => rename(key, event.target.value)}
            className={`w-1/3 font-mono text-xs ${INPUT_CLASS}`}
          />
          <input
            aria-label={`Value for ${key || `key ${index + 1}`}`}
            value={asText(held)}
            disabled={disabled}
            placeholder="value"
            onChange={(event) =>
              onChange({ ...value, [key]: fromText(event.target.value) })
            }
            className={`min-w-0 flex-1 font-mono text-xs ${INPUT_CLASS}`}
          />
          <IconButton
            label={`Remove ${key || `key ${index + 1}`}`}
            isDisabled={disabled}
            onPress={() => {
              const { [key]: _removed, ...rest } = value
              onChange(rest)
            }}
          >
            <FiX aria-hidden="true" className="h-4 w-4" />
          </IconButton>
        </div>
      ))}
      <Button
        variant="ghost"
        isDisabled={disabled || Object.hasOwn(value, "")}
        onPress={() => onChange({ ...value, "": "" })}
      >
        <FiPlus aria-hidden="true" className="h-3.5 w-3.5" /> Add entry
      </Button>
    </div>
  )
}

/** An ordered list of plain strings, one per row. */
function ListRows({
  value,
  noun,
  disabled,
  onChange,
}: {
  value: string[]
  noun: string
  disabled: boolean
  onChange: (next: string[]) => void
}) {
  return (
    <div className="flex flex-col gap-2">
      {value.map((entry, index) => (
        // Keyed by position: these rows are ordered rather than identified, and
        // nothing here is stable enough to key on.
        <div key={index} className="flex items-center gap-2">
          <input
            aria-label={`${parameterLabel(noun)} ${index + 1}`}
            value={entry}
            disabled={disabled}
            onChange={(event) =>
              onChange(
                value.map((held, at) =>
                  at === index ? event.target.value : held,
                ),
              )
            }
            className={`min-w-0 flex-1 text-xs ${INPUT_CLASS}`}
          />
          <IconButton
            label={`Remove ${noun} ${index + 1}`}
            isDisabled={disabled}
            onPress={() => onChange(value.filter((_, at) => at !== index))}
          >
            <FiX aria-hidden="true" className="h-4 w-4" />
          </IconButton>
        </div>
      ))}
      <Button
        variant="ghost"
        isDisabled={disabled}
        onPress={() => onChange([...value, ""])}
      >
        <FiPlus aria-hidden="true" className="h-3.5 w-3.5" /> Add {noun}
      </Button>
    </div>
  )
}

/**
 * The named switches a guardrail documents, plus any key the operator names.
 *
 * The named ones are what this build could read out of any-guardrail; they are
 * not the set a vendor accepts. Alinia declares that it detects personal data
 * and documents no key that says so anywhere reachable, so a picker offering
 * Alinia for that job and no way to ask for it is a dead end the operator
 * cannot leave. Naming a key by hand is the way out, and it belongs here rather
 * than only in the JSON view.
 */
function FlagRows({
  options,
  value,
  noun,
  disabled,
  onChange,
}: {
  options: FieldOption[]
  value: Record<string, unknown>
  /** What one entry is called, for the control that adds one. */
  noun: string
  disabled: boolean
  onChange: (next: Record<string, unknown>) => void
}) {
  const known = new Set(options.map((option) => option.key))
  // A key this build has never heard of is shown rather than dropped, and shown
  // as its own editable name so a typo is fixable without leaving the view.
  const extra = Object.entries(value).filter(([key]) => !known.has(key))
  const rename = (from: string, to: string) =>
    onChange(
      Object.fromEntries(
        Object.entries(value).map(([key, held]) =>
          key === from ? [to, held] : [key, held],
        ),
      ),
    )
  return (
    <div className="flex flex-col gap-2.5">
      <div className="flex flex-col gap-2.5 sm:grid sm:grid-cols-2 sm:items-start">
        {options.map((option) => (
          <div key={option.key} className="flex flex-col gap-0.5">
            <Checkbox
              isSelected={Object.hasOwn(value, option.key)}
              isDisabled={disabled}
              onChange={(next) => {
                if (next) {
                  onChange({ ...value, [option.key]: option.value ?? true })
                  return
                }
                const { [option.key]: _removed, ...rest } = value
                onChange(rest)
              }}
            >
              {option.label}
            </Checkbox>
            {option.help ? (
              <span className="pl-6 text-caption text-subtle">
                {option.help}
              </span>
            ) : null}
          </div>
        ))}
      </div>
      {extra.map(([key], index) => (
        // Keyed by position: the key is what is being edited, so keying on it
        // remounts the input on every keystroke and loses focus.
        <div key={index} className="flex items-center gap-2">
          <input
            aria-label={`Other ${noun} ${index + 1}`}
            value={key}
            disabled={disabled}
            placeholder={`${noun} name`}
            onChange={(event) => rename(key, event.target.value)}
            className={`min-w-0 flex-1 font-mono text-xs ${INPUT_CLASS}`}
          />
          <IconButton
            label={`Remove ${key || `${noun} ${index + 1}`}`}
            isDisabled={disabled}
            onPress={() => {
              const { [key]: _removed, ...rest } = value
              onChange(rest)
            }}
          >
            <FiX aria-hidden="true" className="h-4 w-4" />
          </IconButton>
        </div>
      ))}
      <div>
        <Button
          variant="ghost"
          isDisabled={disabled || Object.hasOwn(value, "")}
          onPress={() => onChange({ ...value, "": true })}
        >
          <FiPlus aria-hidden="true" className="h-3.5 w-3.5" /> Add another{" "}
          {noun}
        </Button>
      </div>
    </div>
  )
}

/** A list built from named presets, each either in the list or not. */
function PresetRows({
  options,
  value,
  disabled,
  onChange,
}: {
  options: FieldOption[]
  value: unknown[]
  disabled: boolean
  onChange: (next: unknown[]) => void
}) {
  const has = (option: FieldOption) =>
    value.some(
      (entry) => JSON.stringify(entry) === JSON.stringify(option.value),
    )
  const known = new Set(options.map((option) => JSON.stringify(option.value)))
  const extra = value.filter((entry) => !known.has(JSON.stringify(entry)))
  return (
    <div className="flex flex-col gap-2.5 sm:grid sm:grid-cols-2 sm:items-start">
      {options.map((option) => (
        <div key={option.key} className="flex flex-col gap-0.5">
          <Checkbox
            isSelected={has(option)}
            isDisabled={disabled}
            onChange={(next) =>
              onChange(
                next
                  ? [...value, option.value]
                  : value.filter(
                      (entry) =>
                        JSON.stringify(entry) !== JSON.stringify(option.value),
                    ),
              )
            }
          >
            {option.label}
          </Checkbox>
          {option.help ? (
            <span className="pl-6 text-caption text-subtle">{option.help}</span>
          ) : null}
        </div>
      ))}
      {extra.length > 0 ? (
        <span className="text-caption text-subtle">
          {extra.length} more set in JSON.
        </span>
      ) : null}
    </div>
  )
}

export function GuardrailJsonField({
  spec,
  guardrailName,
  operation,
  value,
  error,
  disabled,
  description,
  onChange,
}: {
  spec: GuardrailParameterSpec
  /** The class the argument belongs to, which is how a suggestion is found. */
  guardrailName: string
  /** The category the guardrail was picked under, for the note below. */
  operation?: string
  value: string
  error: string | undefined
  disabled: boolean
  description: ReactNode
  onChange: (next: string) => void
}) {
  const label = parameterLabel(spec.name)
  const suggestion = jsonFieldSpec(guardrailName, spec)
  const [view, setView] = useState<View>(suggestion ? "fields" : "json")
  const jsonId = useId()

  // Only where the field has a vocabulary to be missing from, and only while
  // nothing is set: once a key is chosen the operator has answered it.
  const unmapped =
    suggestion?.byCategory !== undefined &&
    operation !== undefined &&
    operation !== "" &&
    suggestion.byCategory[operation] === undefined &&
    value.trim() === ""

  const parsed = parse(value)
  const kind = suggestion?.kind ?? shapeOf(parsed.value)
  const options = suggestion?.options ?? []
  const noun = suggestion?.itemNoun ?? "item"
  const write = (next: unknown) => onChange(serialize(next))

  // What the controls can hold. A value of the wrong shape for this field, or
  // text that is not JSON at all, is left to the JSON view rather than
  // rewritten: nothing here may lose a value it did not understand.
  const editable =
    parsed.error === undefined &&
    kind !== undefined &&
    (parsed.value === undefined ||
      (kind === "list" || kind === "presets"
        ? Array.isArray(parsed.value)
        : isRecord(parsed.value)))

  return (
    <div className="flex flex-col gap-2">
      <div className="flex flex-wrap items-center justify-between gap-x-4 gap-y-2">
        <span className="text-body" id={`${jsonId}-label`}>
          {label}
        </span>
        {kind === undefined ? null : (
          <Segmented
            label={`How to edit ${label}`}
            value={view}
            onChange={(next) => setView(next as View)}
            options={[
              { value: "fields", label: "Fields" },
              { value: "json", label: "JSON" },
            ]}
          />
        )}
      </div>

      {/* The guardrail was offered for a job this build cannot name a key for,
          which is a dead end unless it says so: the catalog has Alinia
          detecting personal data and no source names the detection that does
          it. Naming it by hand is the way out, and so is the vendor's own
          reference, which the picker links to. */}
      {view === "fields" && editable && unmapped ? (
        <p className="text-caption text-warning">
          {`This gateway cannot say which ${noun} covers what you chose. Add it by name below, or check the guardrail's reference.`}
        </p>
      ) : null}
      {view === "fields" && editable ? (
        <div className="flex flex-col gap-2.5">
          {kind === "flags" ? (
            <FlagRows
              options={options}
              value={isRecord(parsed.value) ? parsed.value : {}}
              noun={noun}
              disabled={disabled}
              onChange={write}
            />
          ) : kind === "presets" ? (
            <PresetRows
              options={options}
              value={Array.isArray(parsed.value) ? parsed.value : []}
              disabled={disabled}
              onChange={write}
            />
          ) : kind === "list" ? (
            <ListRows
              value={
                Array.isArray(parsed.value) ? (parsed.value as string[]) : []
              }
              noun={noun}
              disabled={disabled}
              onChange={write}
            />
          ) : (
            <MapRows
              value={isRecord(parsed.value) ? parsed.value : {}}
              disabled={disabled}
              onChange={write}
            />
          )}
        </div>
      ) : view === "fields" ? (
        <p className="py-1 text-caption text-warning">
          This value cannot be shown as fields. Edit it as JSON.
        </p>
      ) : (
        <TextField
          value={value}
          onChange={onChange}
          isDisabled={disabled}
          isInvalid={Boolean(error) || Boolean(parsed.error)}
          aria-labelledby={`${jsonId}-label`}
          className="flex flex-col gap-1"
        >
          <Label className="sr-only">{label} as JSON</Label>
          <TextArea
            rows={4}
            spellCheck={false}
            placeholder={kind === "list" || kind === "presets" ? "[]" : "{}"}
            className="font-mono text-xs"
          />
        </TextField>
      )}

      <FieldMessages>
        <Description className={error || parsed.error ? "text-danger" : ""}>
          {/* While the controls are up, the registry's line or nothing: the
              catalog's paragraph on which dict to pass is exactly what the
              checkboxes replaced. The JSON view shows that paragraph. */}
          {error ??
            parsed.error ??
            (view === "fields" && suggestion
              ? suggestion.help
              : view === "json" && spec.description
                ? spec.description
                : description)}
        </Description>
      </FieldMessages>
    </div>
  )
}
