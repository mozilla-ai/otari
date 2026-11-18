import { useState } from "react"

import type { AgentRecommender } from "@/client"
import { FormDialog } from "@/design-system/feedback/FormDialog"
import { Field } from "@/design-system/forms/Field"
import { SecretField } from "@/design-system/forms/SecretField"
import { Select } from "@/design-system/forms/Select"
import { useDirtySnapshot } from "@/design-system/forms/useDirtySnapshot"
import {
  useAddAgentRecommender,
  useUpdateAgentRecommender,
} from "@/shared/api/agentRecommenders"

import {
  checkDraft,
  createBody,
  DECISION_PROVIDERS,
  type DraftErrors,
  draftFrom,
  EMPTY_DRAFT,
  isSelfHosted,
  keyPlaceholder,
  patchBody,
  type RecommenderDraft,
} from "./recommenderDraft"

// Add or edit a router, the decision model that picks a subagent's model: where
// to reach it, its key, and what a recommendation costs, its tokens at the
// rates set here.
//
// The stored key is never shown: on edit the key box names its last four
// characters and keeps it when left blank, as long as the provider and the
// address stay the same.

export function AgentRecommenderForm({
  recommender,
  isOpen,
  onClose,
}: {
  /** The router to edit; absent adds a new one. */
  recommender?: AgentRecommender
  isOpen: boolean
  onClose: () => void
}) {
  const add = useAddAgentRecommender()
  const update = useUpdateAgentRecommender()
  const write = recommender ? update : add
  const [draft, setDraft] = useState<RecommenderDraft>(() =>
    recommender ? draftFrom(recommender) : EMPTY_DRAFT,
  )
  const [errors, setErrors] = useState<DraftErrors>({})
  const { isDirty } = useDirtySnapshot(draft)

  const edit = (change: Partial<RecommenderDraft>) => {
    setDraft((current) => ({ ...current, ...change }))
    setErrors({})
    if (write.isError) write.reset()
  }
  const selfHosted = isSelfHosted(draft.provider)

  const submit = () => {
    const found = checkDraft(draft, recommender)
    setErrors(found)
    if (Object.keys(found).length > 0) return
    if (recommender) {
      update.mutate(
        { id: recommender.id, body: patchBody(draft) },
        { onSuccess: onClose },
      )
    } else {
      add.mutate(createBody(draft), { onSuccess: onClose })
    }
  }

  return (
    <FormDialog
      isOpen={isOpen}
      onOpenChange={(open) => {
        if (!open) onClose()
      }}
      title={recommender ? "Edit router" : "New router"}
      description={
        recommender
          ? undefined
          : "The first router added is switched on. Switch on a later one in the table."
      }
      submitLabel={recommender ? "Save" : "Add router"}
      onSubmit={submit}
      isPending={write.isPending}
      isDirty={isDirty}
      error={write.error}
    >
      <Select
        label="Provider"
        value={draft.provider}
        onChange={(provider) => edit({ provider, apiKey: "" })}
        options={DECISION_PROVIDERS}
        shouldReserveMessage={false}
      />
      <Field
        label="Model"
        value={draft.model}
        onChange={(model) => edit({ model })}
        placeholder="jev-latest"
        description="The decision model, as the provider names it."
        isInvalid={errors.model !== undefined}
        errorMessage={errors.model}
      />
      <Field
        label="API base"
        value={draft.apiBase}
        onChange={(apiBase) => edit({ apiBase, apiKey: "" })}
        placeholder={selfHosted ? "http://decisions.internal:8080" : ""}
        description={
          selfHosted
            ? "The server's address."
            : "Optional. Leave blank to use the provider's own address."
        }
        isInvalid={errors.apiBase !== undefined}
        errorMessage={errors.apiBase}
      />
      <SecretField
        label="API key"
        value={draft.apiKey}
        onChange={(apiKey) => edit({ apiKey })}
        placeholder={keyPlaceholder(draft, recommender)}
        description={
          selfHosted
            ? "Optional for a server that takes no key."
            : "Stored encrypted and never shown again."
        }
        isInvalid={errors.apiKey !== undefined}
        errorMessage={errors.apiKey}
      />
      <Field
        label="Input, per 1M tokens (USD)"
        value={draft.inputPrice}
        onChange={(inputPrice) => edit({ inputPrice })}
        placeholder="0"
        inputMode="decimal"
        description="Charged for the tokens the decision model reads."
        isInvalid={errors.inputPrice !== undefined}
        errorMessage={errors.inputPrice}
      />
      <Field
        label="Output, per 1M tokens (USD)"
        value={draft.outputPrice}
        onChange={(outputPrice) => edit({ outputPrice })}
        placeholder="0"
        inputMode="decimal"
        description="Charged for the tokens the decision model writes. Leave both prices at 0 for a free router."
        isInvalid={errors.outputPrice !== undefined}
        errorMessage={errors.outputPrice}
      />
    </FormDialog>
  )
}
