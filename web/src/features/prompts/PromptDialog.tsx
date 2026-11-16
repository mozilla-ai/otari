import { useEffect, useState } from "react"
import type { StoredPrompt } from "@/client"
import { FormDialog } from "@/design-system/feedback/FormDialog"
import { Field } from "@/design-system/forms/Field"
import { TextArea } from "@/design-system/forms/TextArea"
import { useCreatePrompt, useUpdatePrompt } from "@/shared/api/prompts"

interface PromptDialogProps {
  isOpen: boolean
  onOpenChange: (isOpen: boolean) => void
  /** When set, the dialog edits this prompt; when null, it creates a new one. */
  editing: StoredPrompt | null
}

export function PromptDialog({
  isOpen,
  onOpenChange,
  editing,
}: PromptDialogProps) {
  const create = useCreatePrompt()
  const update = useUpdatePrompt()

  const [name, setName] = useState("")
  const [content, setContent] = useState("")

  // biome-ignore lint/correctness/useExhaustiveDependencies: runs on open; editing and reset refs are intentionally excluded
  useEffect(() => {
    if (isOpen) {
      setName(editing?.name ?? "")
      setContent(editing?.content ?? "")
      create.reset()
      update.reset()
    }
  }, [isOpen])

  const isPending = create.isPending || update.isPending
  const error = create.error ?? update.error
  const isDirty = editing
    ? name !== editing.name || content !== editing.content
    : name !== "" || content !== ""

  function handleSubmit() {
    if (editing) {
      update.mutate(
        { id: editing.id, body: { name: name.trim(), content } },
        { onSuccess: () => onOpenChange(false) },
      )
    } else {
      create.mutate(
        { name: name.trim(), content },
        { onSuccess: () => onOpenChange(false) },
      )
    }
  }

  const title = editing ? "Edit prompt" : "New prompt"
  const submitLabel = editing ? "Save" : "Create prompt"

  return (
    <FormDialog
      isOpen={isOpen}
      onOpenChange={onOpenChange}
      title={title}
      submitLabel={submitLabel}
      onSubmit={handleSubmit}
      isPending={isPending}
      error={error}
      isDirty={isDirty}
      size="lg"
    >
      <div className="flex flex-col gap-4">
        <Field
          label="Name"
          value={name}
          onChange={setName}
          placeholder="e.g. system-prompt-v2"
          description="A unique identifier callers use to reference this prompt."
          isDisabled={isPending}
        />
        <TextArea
          label="Content"
          value={content}
          onChange={setContent}
          placeholder="Enter the prompt text…"
          rows={6}
          isDisabled={isPending}
        />
      </div>
    </FormDialog>
  )
}
