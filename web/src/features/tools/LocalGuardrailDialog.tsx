import { useState } from "react"

import type {
  BuiltInGuardrailSpec,
  StoredGuardrail,
  UpdateGuardrailRequest,
} from "@/client"
import { FormDialog } from "@/design-system/feedback/FormDialog"
import { InfoBanner } from "@/design-system/feedback/InfoBanner"
import { Field } from "@/design-system/forms/Field"
import { useDirtySnapshot } from "@/design-system/forms/useDirtySnapshot"
import { GuardrailParameterFields } from "@/features/tools/GuardrailParameterFields"
import { GuardrailParametersSection } from "@/features/tools/GuardrailParametersSection"
import { GuardrailPicker } from "@/features/tools/GuardrailPicker"
import { buildCreateKwargs } from "@/features/tools/guardrailParameters"
import { suggestName, taskLabel } from "@/features/tools/guardrailTasks"
import { useGuardrailParameterForm } from "@/features/tools/useGuardrailParameterForm"
import {
  useOrganizationContext,
  useProviderKeyEncryption,
} from "@/shared/api/organizations"
import { useCreateGuardrail, useUpdateGuardrail } from "@/shared/api/tools"

// Defining one guardrail, in three stages: the task, the guardrail that does it,
// then whatever that guardrail asks for.
//
// The mutations live here rather than on the card, which is the rule in
// feedback.md and not a preference: the card remounts this component on every
// open, and a mutation held above that key keeps its error, so a failed create's
// banner would greet the next open of a blank form.
//
// The third stage is hidden rather than disabled until a guardrail is chosen.
// The two controls above it exist and are about to be usable; these fields do
// not exist at all until a guardrail names them, and there is nothing honest to
// draw disabled.

/** The task to open an existing row on: the guardrail's own headline category. */
function taskOf(spec: BuiltInGuardrailSpec | undefined): string {
  return spec?.primary_category ?? ""
}

export function LocalGuardrailDialog({
  isOpen,
  onClose,
  guardrails,
  takenNames,
  editing,
}: {
  isOpen: boolean
  onClose: () => void
  guardrails: readonly BuiltInGuardrailSpec[]
  /** Names already in use, so a suggestion does not collide with one. */
  takenNames: readonly string[]
  /** The row being edited, or undefined when defining a new guardrail. */
  editing?: StoredGuardrail
}) {
  const create = useCreateGuardrail()
  const update = useUpdateGuardrail()
  const context = useOrganizationContext()
  const encryptionAvailable = useProviderKeyEncryption()

  const editingSpec = guardrails.find(
    (spec) => spec.guardrail_name === editing?.guardrail_name,
  )
  const [task, setTask] = useState(() => taskOf(editingSpec))
  const [guardrailName, setGuardrailName] = useState(
    () => editing?.guardrail_name ?? "",
  )
  // Null means "still the suggestion". An empty string is a cleared field, which
  // is a decision and must not re-suggest under the operator's cursor.
  const [nameDraft, setNameDraft] = useState<string | null>(
    editing ? editing.name : null,
  )

  const spec = guardrails.find(
    (entry) => entry.guardrail_name === guardrailName,
  )
  const createSpecs = spec?.create_parameters ?? []
  const validateSpecs = spec?.validate_parameters ?? []
  // The form holds only what can be stored. An unstorable argument is still
  // rendered below, disabled and saying why, but seeding or validating it would
  // block a submit over a field nobody can fill.
  const storedSecrets = Object.keys(editing?.create_secrets ?? {})
  const storableSpecs = createSpecs
    .filter((entry) => entry.storable !== false)
    // A required secret the row already holds is not required *of the operator*:
    // `SecretField` is never prefilled, so leaving it blank means "keep it", and
    // a validator that still called it missing refused every edit that changed
    // anything else. `required` on the wire means "the guardrail cannot run
    // without one", which a stored value satisfies.
    .map((entry) =>
      entry.secret && entry.required && storedSecrets.includes(entry.name)
        ? { ...entry, required: false }
        : entry,
    )

  const name = nameDraft ?? suggestName(task, takenNames)
  const setupForm = useGuardrailParameterForm(
    storableSpecs,
    editing?.create_kwargs,
    guardrailName,
  )
  const perCall = useGuardrailParameterForm(
    validateSpecs,
    editing?.validate_kwargs,
    guardrailName,
  )

  // A guardrail that takes a credential cannot be stored at all without the key
  // that encrypts it. Asked of the chosen guardrail rather than of the Add
  // button, so a local guardrail that needs no credential stays addable on a
  // deployment with no OTARI_SECRET_KEY set.
  const needsSecret = storableSpecs.some((entry) => entry.secret)
  // `context.data &&` so a failed context read does not claim the key is unset.
  const secretBlocked =
    Boolean(context.data) && needsSecret && !encryptionAvailable

  const { isDirty } = useDirtySnapshot({
    task,
    guardrailName,
    nameDraft,
    values: setupForm.values,
    perCallValues: perCall.values,
    extraJson: perCall.extraJson,
  })

  const pending = create.isPending || update.isPending
  const ready = name.trim() !== "" && guardrailName !== "" && !secretBlocked

  const submit = () => {
    if (pending || !ready) return
    // Both, not a short-circuit: each reports its own messages and the operator
    // should see every field that needs them, not the first half of them.
    const setupOk = setupForm.check()
    const perCallOk = perCall.check()
    if (!setupOk || !perCallOk) return

    const create_kwargs = buildCreateKwargs(
      storableSpecs,
      setupForm.values,
      editing ? storedSecrets : undefined,
    )
    const validate_kwargs = perCall.build() ?? {}

    if (editing) {
      // `guardrail_name` is deliberately absent: every stored argument belongs
      // to the class, so changing it here would orphan the lot.
      const body: UpdateGuardrailRequest = {
        create_kwargs,
        validate_kwargs,
        expected_updated_at: editing.updated_at,
      }
      update.mutate({ name: editing.name, body }, { onSuccess: onClose })
      return
    }
    create.mutate(
      {
        name: name.trim(),
        guardrail_name: guardrailName,
        create_kwargs,
        validate_kwargs,
        enabled: true,
      },
      { onSuccess: onClose },
    )
  }

  return (
    <FormDialog
      isOpen={isOpen}
      onOpenChange={(open) => {
        if (!open) onClose()
      }}
      size="lg"
      title={editing ? `Edit ${editing.name}` : "New guardrail"}
      submitLabel={editing ? "Save guardrail" : "Add guardrail"}
      onSubmit={submit}
      isPending={pending}
      isSubmitDisabled={!ready}
      isDirty={isDirty}
      error={create.error ?? update.error}
    >
      {editing ? (
        <div className="flex flex-col gap-1">
          <span className="text-body">
            {editingSpec?.display_name ?? editing.guardrail_name}
            {task === "" ? null : (
              <span className="text-subtle"> · {taskLabel(task)}</span>
            )}
          </span>
          <span className="text-caption">
            Which guardrail runs cannot be changed here: every argument below
            belongs to this one. Remove this definition and add another instead.
          </span>
        </div>
      ) : (
        <GuardrailPicker
          guardrails={guardrails}
          task={task}
          guardrailName={guardrailName}
          disabled={pending}
          onChange={(nextTask, nextGuardrail) => {
            setTask(nextTask)
            setGuardrailName(nextGuardrail)
          }}
        />
      )}

      {spec ? (
        <>
          <Field
            label="Name"
            value={name}
            onChange={setNameDraft}
            isRequired
            // The name is the primary key of the row, so the API offers no way
            // to change it. Renaming is a remove and an add.
            isDisabled={Boolean(editing) || pending}
            placeholder="prompt-injection"
            description="What a caller sends as its profile. Suggested from the task; type your own if you prefer."
            reserveMessage
          />
          {secretBlocked ? (
            <InfoBanner tone="warning">
              {spec.display_name} needs a credential, and this gateway has no
              OTARI_SECRET_KEY set to encrypt one with. Set it and restart, then
              add this guardrail.
            </InfoBanner>
          ) : null}
          {storableSpecs.length > 0 || createSpecs.length > 0 ? (
            <GuardrailParameterFields
              specs={createSpecs}
              scopeName={name || spec.display_name}
              values={setupForm.values}
              errors={setupForm.issues}
              disabled={pending || secretBlocked}
              storedSecrets={editing ? storedSecrets : undefined}
              onChange={setupForm.setValue}
            />
          ) : null}
          <GuardrailParametersSection
            key={`${guardrailName}:${validateSpecs.length}`}
            specs={validateSpecs}
            scopeName={name || spec.display_name}
            values={perCall.values}
            errors={perCall.issues}
            extraJson={perCall.extraJson}
            extraJsonError={perCall.rawError}
            described
            disabled={pending || secretBlocked}
            onChange={perCall.setValue}
            onExtraJsonChange={perCall.setExtraJson}
          />
        </>
      ) : null}
    </FormDialog>
  )
}
