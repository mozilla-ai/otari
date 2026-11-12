import { useState } from "react"
import type { ModelInputModality, ModelMetadataOverride } from "@/client"
import { FormDialog } from "@/design-system/feedback/FormDialog"
import { Field } from "@/design-system/forms/Field"
import { MultiSelect } from "@/design-system/forms/MultiSelect"
import { Select } from "@/design-system/forms/Select"
import { useDirtySnapshot } from "@/design-system/forms/useDirtySnapshot"

const MODALITIES: readonly { id: ModelInputModality; label: string }[] = [
  { id: "text", label: "Text" },
  { id: "image", label: "Image" },
  { id: "audio", label: "Audio" },
  { id: "video", label: "Video" },
  { id: "pdf", label: "PDF" },
]

// "auto" leaves the flag to the provider's listing and models.dev. Not "": the
// form Select reads that as nothing selected.
const FLAG_OPTIONS = [
  { value: "auto", label: "From the provider" },
  { value: "yes", label: "Yes" },
  { value: "no", label: "No" },
] as const

type Flag = (typeof FLAG_OPTIONS)[number]["value"]

const MAX_TOKENS = 2 ** 31 - 1

function flagOf(value: boolean | null | undefined): Flag {
  if (value == null) return "auto"
  return value ? "yes" : "no"
}

function flagValue(flag: string): boolean | null {
  if (flag === "auto") return null
  return flag === "yes"
}

/** A token count, `null` when blank, `NaN` when not a whole number in range. */
function parseTokens(value: string): number | null {
  const trimmed = value.trim()
  if (trimmed === "") return null
  const parsed = Number(trimmed)
  return Number.isInteger(parsed) && parsed > 0 && parsed <= MAX_TOKENS
    ? parsed
    : Number.NaN
}

export interface ModelDetailsDialogProps {
  isOpen: boolean
  onOpenChange: (open: boolean) => void
  /** The `provider:model` the details describe. */
  selector: string
  /** What is stored now; `undefined` when nothing is. */
  current: ModelMetadataOverride | undefined
  /**
   * Saves the override. Awaited, and a rejection is reported inside this
   * dialog, as `SetPriceDialog` does.
   */
  onSubmit: (body: ModelMetadataOverride) => Promise<unknown>
}

/**
 * The limits and capabilities of one offering, where its server and models.dev
 * say nothing or say it wrong. A blank field keeps their word.
 */
export function ModelDetailsDialog({
  isOpen,
  onOpenChange,
  selector,
  current,
  onSubmit,
}: ModelDetailsDialogProps) {
  // Seeded on mount only: the caller remounts this on each open.
  const [contextWindow, setContextWindow] = useState(
    current?.context_window?.toString() ?? "",
  )
  const [maxOutput, setMaxOutput] = useState(
    current?.max_output_tokens?.toString() ?? "",
  )
  const [toolCall, setToolCall] = useState<string>(flagOf(current?.tool_call))
  const [reasoning, setReasoning] = useState<string>(flagOf(current?.reasoning))
  const [structuredOutput, setStructuredOutput] = useState<string>(
    flagOf(current?.structured_output),
  )
  const [modalities, setModalities] = useState<string[]>(
    current?.input_modalities ?? [],
  )
  const { isDirty } = useDirtySnapshot({
    contextWindow,
    maxOutput,
    toolCall,
    reasoning,
    structuredOutput,
    modalities,
  })
  const [isSaving, setIsSaving] = useState(false)
  const [failure, setFailure] = useState<unknown>(undefined)

  const contextTokens = parseTokens(contextWindow)
  const output = parseTokens(maxOutput)
  const windowInvalid = Number.isNaN(contextTokens)
  const outputInvalid = Number.isNaN(output)

  const submit = () => {
    if (windowInvalid || outputInvalid || isSaving) return
    setFailure(undefined)
    setIsSaving(true)
    onSubmit({
      context_window: contextTokens,
      max_output_tokens: output,
      tool_call: flagValue(toolCall),
      reasoning: flagValue(reasoning),
      structured_output: flagValue(structuredOutput),
      input_modalities:
        modalities.length > 0 ? (modalities as ModelInputModality[]) : null,
    })
      .catch((error: unknown) => setFailure(error))
      .finally(() => setIsSaving(false))
  }

  return (
    <FormDialog
      isOpen={isOpen}
      onOpenChange={onOpenChange}
      size="lg"
      title="Edit details"
      description={`Leave a field blank to keep what the provider and models.dev say about ${selector}.`}
      submitLabel="Save details"
      onSubmit={submit}
      isPending={isSaving}
      isSubmitDisabled={windowInvalid || outputInvalid}
      isDirty={isDirty}
      error={failure}
    >
      <div className="grid gap-3 sm:grid-cols-2">
        <Field
          label="Context window"
          value={contextWindow}
          onChange={setContextWindow}
          inputMode="numeric"
          placeholder="131072"
          autoFocus
          isInvalid={windowInvalid}
          errorMessage="A whole number of tokens."
        />
        <Field
          label="Max output tokens"
          value={maxOutput}
          onChange={setMaxOutput}
          inputMode="numeric"
          placeholder="8192"
          isInvalid={outputInvalid}
          errorMessage="A whole number of tokens."
        />
        <Select
          label="Calls tools"
          value={toolCall}
          onChange={setToolCall}
          options={FLAG_OPTIONS}
        />
        <Select
          label="Reasons"
          value={reasoning}
          onChange={setReasoning}
          options={FLAG_OPTIONS}
        />
        <Select
          label="Structured output"
          value={structuredOutput}
          onChange={setStructuredOutput}
          options={FLAG_OPTIONS}
        />
      </div>
      <MultiSelect
        label="Reads"
        description="Leave empty to keep what the provider says."
        options={MODALITIES}
        value={modalities}
        onChange={setModalities}
        searchPlaceholder="From the provider"
      />
    </FormDialog>
  )
}
