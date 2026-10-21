/**
 * Add or edit a subagent router, the decision model that picks a subagent's
 * model: where to reach it, its key, and what a recommendation costs, its
 * tokens at the rates set here.
 *
 * The stored key is never shown: on edit the key box names its last four
 * characters and keeps it when left blank, as long as the provider and the
 * address stay the same.
 *
 * Saving is three steps, so what an edition adds through
 * `overlayRecommenderMargin` lands before the router answers: the router is
 * saved switched off, the edition saves its part, and the router is switched
 * on if it is meant to be. A failure after the first step leaves the router
 * off, and saving again picks up the router already added rather than adding
 * a second.
 */

import { useState } from "react"

import type { AgentRecommender, DecisionProviderOption } from "@/client"
import { FormDialog } from "@/design-system/feedback/FormDialog"
import { Field } from "@/design-system/forms/Field"
import { SecretField } from "@/design-system/forms/SecretField"
import { Select } from "@/design-system/forms/Select"
import { useDirtySnapshot } from "@/design-system/forms/useDirtySnapshot"
import {
  RecommenderMarginField,
  useRecommenderMargin,
} from "@/features/routing/overlayRecommenderMargin"
import {
  useAddAgentRecommender,
  useUpdateAgentRecommender,
} from "@/shared/api/agentRecommenders"

import {
  checkDraft,
  createBody,
  type DraftErrors,
  draftFrom,
  EMPTY_DRAFT,
  isSelfHosted,
  keyPlaceholder,
  patchBody,
  providerOptions,
  type RecommenderDraft,
} from "./recommenderDraft"

export function SubagentRouterForm({
  recommender,
  providers,
  switchOn,
  isOpen,
  onClose,
}: {
  /** The router to edit; absent adds a new one. */
  recommender?: AgentRecommender
  /** The decision providers the server lists. */
  providers: readonly DecisionProviderOption[]
  /**
   * Whether the router answers once saved: a router being edited that is on,
   * or the first one added. Read when the dialog mounts, because the first
   * step switches the router off and the list refetches under it.
   */
  switchOn: boolean
  isOpen: boolean
  onClose: () => void
}) {
  const add = useAddAgentRecommender()
  const update = useUpdateAgentRecommender()
  const margin = useRecommenderMargin()
  const [isMeantOn] = useState(switchOn)
  const [draft, setDraft] = useState<RecommenderDraft>(() =>
    recommender ? draftFrom(recommender) : EMPTY_DRAFT,
  )
  const [marginValue, setMarginValue] = useState(() =>
    recommender ? margin.initialValue(recommender.id) : "",
  )
  const [errors, setErrors] = useState<DraftErrors & { margin?: string }>({})
  // A router this dialog added on an attempt that failed later, so a retry
  // finishes it rather than adding a second.
  const [added, setAdded] = useState<AgentRecommender>()
  const [isSaving, setIsSaving] = useState(false)
  const [failure, setFailure] = useState<unknown>()
  const { isDirty } = useDirtySnapshot({ draft, marginValue })

  const stored = recommender ?? added
  const edit = (change: Partial<RecommenderDraft>) =>
    setDraft((current) => ({ ...current, ...change }))
  const selfHosted = isSelfHosted(draft.provider, providers)

  const submit = async () => {
    const found = {
      ...checkDraft(draft, stored, providers),
      ...margin.validate(marginValue),
    }
    setErrors(found)
    if (Object.keys(found).length > 0) return
    setFailure(undefined)
    setIsSaving(true)
    try {
      const saved = stored
        ? await update.mutateAsync({ id: stored.id, patch: patchBody(draft) })
        : await add.mutateAsync(createBody(draft))
      if (!recommender) setAdded(saved)
      await margin.save(saved.id, marginValue)
      if (isMeantOn) {
        await update.mutateAsync({ id: saved.id, patch: { enabled: true } })
      }
      onClose()
    } catch (error) {
      setFailure(error)
    } finally {
      setIsSaving(false)
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
      onSubmit={() => void submit()}
      isPending={isSaving}
      isDirty={isDirty}
      error={failure}
    >
      <Select
        label="Provider"
        value={draft.provider}
        onChange={(provider) => edit({ provider })}
        options={providerOptions(providers)}
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
        onChange={(apiBase) => edit({ apiBase })}
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
        placeholder={keyPlaceholder(draft, stored)}
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
        description="Charged for the tokens the decision model reads."
        isInvalid={errors.inputPrice !== undefined}
        errorMessage={errors.inputPrice}
      />
      <Field
        label="Output, per 1M tokens (USD)"
        value={draft.outputPrice}
        onChange={(outputPrice) => edit({ outputPrice })}
        placeholder="0"
        description="Charged for the tokens the decision model writes."
        isInvalid={errors.outputPrice !== undefined}
        errorMessage={errors.outputPrice}
      />
      <RecommenderMarginField
        recommenderId={stored?.id ?? ""}
        value={marginValue}
        onChange={setMarginValue}
        error={errors.margin ?? ""}
      />
    </FormDialog>
  )
}
