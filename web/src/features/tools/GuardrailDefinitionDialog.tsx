import { useState } from "react"

import type {
  BuiltInGuardrailSpec,
  StoredGuardrail,
  UpdateGuardrailRequest,
} from "@/client"
import { FormDialog } from "@/design-system/feedback/FormDialog"
import { InfoBanner } from "@/design-system/feedback/InfoBanner"
import { Field } from "@/design-system/forms/Field"
import { Select } from "@/design-system/forms/Select"
import { useDirtySnapshot } from "@/design-system/forms/useDirtySnapshot"
import { Disclosure } from "@/design-system/navigation/Disclosure"
import { DocsLink } from "@/design-system/navigation/DocsLink"
import { FormSectionRule } from "@/features/tools/FormSectionRule"
import {
  enforcementFields,
  type GuardrailEnforcement,
  GuardrailEnforcementFields,
} from "@/features/tools/GuardrailEnforcementFields"
import { GuardrailExtraJsonField } from "@/features/tools/GuardrailExtraJsonField"
import { GuardrailParameterFields } from "@/features/tools/GuardrailParameterFields"
import { splitParameters } from "@/features/tools/guardrailFieldSplit"
import { suggestedCreateKwargs } from "@/features/tools/guardrailFieldSuggestions"
import {
  findGuardrail,
  guardrailDocsHref,
  guardrailOptions,
  operationLabel,
  operationOptions,
  suggestName,
} from "@/features/tools/guardrailOperations"
import { buildCreateKwargs } from "@/features/tools/guardrailParameters"
import { useGuardrailParameterForm } from "@/features/tools/useGuardrailParameterForm"
import {
  useOrganizationContext,
  useProviderKeyEncryption,
} from "@/shared/api/organizations"
import {
  useCreateGuardrailDefinition,
  useUpdateGuardrailDefinition,
} from "@/shared/api/tools"
import { useWorkspaces } from "@/shared/api/workspaces"

// Defining one guardrail, in the order the decision is made: what should be
// checked, which guardrail does it, how the deployment wants it enforced, then
// whatever that guardrail asks for.
//
// The second control is disabled rather than absent before the first is
// answered: it exists and is about to be usable, which is a different thing from
// the fields below it, which do not exist at all until a guardrail names them
// and so are hidden rather than drawn empty.
//
// The mutations live here rather than on the card, which is the rule in
// feedback.md and not a preference: the card remounts this component on every
// open, and a mutation held above that key keeps its error, so a failed create's
// banner would greet the next open of a blank form.

/** The operation to open an existing row on: the guardrail's own headline one. */
function operationOf(spec: BuiltInGuardrailSpec | undefined): string {
  return spec?.primary_category ?? ""
}

export function GuardrailDefinitionDialog({
  isOpen,
  onClose,
  onCreated,
  guardrails,
  takenNames,
  editing,
}: {
  isOpen: boolean
  onClose: () => void
  /** A create landed. The card opens its drill-in, where the new row lives. */
  onCreated?: () => void
  guardrails: readonly BuiltInGuardrailSpec[]
  /** Names already in use, so a suggestion does not collide with one. */
  takenNames: readonly string[]
  /** The row being edited, or undefined when defining a new guardrail. */
  editing?: StoredGuardrail
}) {
  const create = useCreateGuardrailDefinition()
  const update = useUpdateGuardrailDefinition()
  const context = useOrganizationContext()
  const encryptionAvailable = useProviderKeyEncryption()

  const editingSpec = findGuardrail(guardrails, editing?.guardrail_name ?? "")
  const [operation, setOperation] = useState(() => operationOf(editingSpec))
  const [guardrailName, setGuardrailName] = useState(
    () => editing?.guardrail_name ?? "",
  )
  // Null means "still the suggestion". An empty string is a cleared field, which
  // is a decision and must not re-suggest under the operator's cursor.
  const [nameDraft, setNameDraft] = useState<string | null>(
    editing ? editing.name : null,
  )
  const [enforcement, setEnforcement] = useState<GuardrailEnforcement>(() => ({
    mode: editing?.mode ?? "block",
    onUnavailable: editing?.on_unavailable ?? "block",
    everywhere: editing?.applies_to_all_workspaces ?? true,
    workspaceIds: editing?.workspace_ids ?? [],
  }))
  const workspaces = useWorkspaces()

  const spec = findGuardrail(guardrails, guardrailName)
  const createSpecs = spec?.create_parameters ?? []
  const validateSpecs = spec?.validate_parameters ?? []
  const storedSecrets = Object.keys(editing?.create_secrets ?? {})
  // The form holds only what can be stored. An unstorable argument is still
  // rendered below, disabled and saying why, but seeding or validating it would
  // block a submit over a field nobody can fill.
  const storableSpecs = createSpecs
    .filter((entry) => entry.storable !== false)
    // A required secret the row already holds is not required *of the operator*:
    // `SecretField` is never prefilled, so leaving it blank means "keep it", and
    // a validator that still called it missing would refuse every edit that
    // changed anything else. `required` on the wire means "the guardrail cannot
    // run without one", which a stored value satisfies.
    .map((entry) =>
      entry.secret && entry.required && storedSecrets.includes(entry.name)
        ? { ...entry, required: false }
        : entry,
    )

  // Split from the catalog entry rather than from `storableSpecs`, which demotes
  // a stored secret so an edit is not refused over a field it need not retype.
  // Splitting on the demoted copy would move a credential into the accordion the
  // moment it was set.
  const createSplit = splitParameters(spec, createSpecs)
  const validateSplit = splitParameters(spec, validateSpecs)
  // Values come from the form's own list, so a demoted secret keeps its relaxed
  // `required` while staying where the split put it.
  const shown = (entries: readonly { name: string }[]) => {
    const names = new Set(entries.map((entry) => entry.name))
    return storableSpecs.filter((entry) => names.has(entry.name))
  }
  const createDecisions = shown(createSplit.decisions)

  const name = nameDraft ?? suggestName(operation, takenNames)
  // A new definition opens with the detection its operation names already
  // switched on, where the vendor documents which one that is. An existing row
  // opens on what it stored, whatever this would have suggested.
  const seeded =
    editing?.create_kwargs ??
    suggestedCreateKwargs(storableSpecs, guardrailName, operation)
  const setup = useGuardrailParameterForm(storableSpecs, seeded, guardrailName)
  const perCall = useGuardrailParameterForm(
    validateSpecs,
    editing?.validate_kwargs,
    guardrailName,
  )

  // A guardrail that takes a credential cannot be stored at all without the key
  // that encrypts it. Asked of the chosen guardrail rather than of the Add
  // button, so one that needs no credential stays addable on a deployment with
  // no OTARI_SECRET_KEY set.
  const needsSecret = storableSpecs.some((entry) => entry.secret)
  // `context.data &&` so a failed context read does not claim the key is unset.
  const secretBlocked =
    Boolean(context.data) && needsSecret && !encryptionAvailable

  const { isDirty } = useDirtySnapshot({
    operation,
    guardrailName,
    nameDraft,
    enforcement,
    values: setup.values,
    perCallValues: perCall.values,
    extraJson: perCall.extraJson,
  })

  // A submit refused over a field inside a collapsed panel is a form that says
  // no and shows nothing, so the panel opens itself the moment one of the
  // fields it hides has something to say.
  const extrasHaveError =
    createSplit.extras.some((entry) => Boolean(setup.issues[entry.name])) ||
    validateSplit.extras.some((entry) => Boolean(perCall.issues[entry.name])) ||
    perCall.rawError !== undefined

  const pending = create.isPending || update.isPending
  // A chosen scope with nothing in it would store a definition that checks
  // nothing, which is what the "Every workspace" option is for.
  const scoped = enforcement.everywhere || enforcement.workspaceIds.length > 0
  const ready =
    name.trim() !== "" && guardrailName !== "" && !secretBlocked && scoped

  const submit = () => {
    if (pending || !ready) return
    // Both, not a short-circuit: each reports its own messages and the operator
    // should see every field that needs them, not the first half of them.
    const setupOk = setup.check()
    const perCallOk = perCall.check()
    if (!setupOk || !perCallOk) return

    const create_kwargs = buildCreateKwargs(
      storableSpecs,
      setup.values,
      editing ? storedSecrets : undefined,
    )
    const validate_kwargs = perCall.build() ?? {}

    if (editing) {
      // `guardrail_name` is deliberately absent: every stored argument belongs
      // to the class, so changing it here would orphan the lot.
      const body: UpdateGuardrailRequest = {
        create_kwargs,
        validate_kwargs,
        ...enforcementFields(enforcement),
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
        ...enforcementFields(enforcement),
      },
      { onSuccess: onCreated ?? onClose },
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
            {operation === "" ? null : (
              <span className="text-subtle">
                {" "}
                · {operationLabel(operation)}
              </span>
            )}
            {editingSpec ? (
              <>
                {" "}
                <DocsLink href={guardrailDocsHref(editingSpec)} />
              </>
            ) : null}
          </span>
          <span className="text-caption">
            Which guardrail runs cannot be changed here: every argument below
            belongs to this one. Remove this definition and add another instead.
          </span>
        </div>
      ) : (
        <>
          <Select
            label="What do you want checked?"
            value={operation}
            // The fields belong to the guardrail and the guardrail belongs to
            // the operation, so a change up here drops what is below it. A
            // guardrail left selected under an operation that no longer offers
            // it is one the operator can no longer see to change.
            onChange={(next) => {
              setOperation(next)
              setGuardrailName("")
            }}
            options={operationOptions(guardrails)}
            isRequired
            isDisabled={pending}
            autoFocus
            placeholder="Choose what to check"
            // Nothing to say under it, so it holds nothing: the reserved line
            // exists for an error to replace a description.
            reserveMessage={false}
          />
          <Select
            label="Which guardrail?"
            value={guardrailName}
            onChange={setGuardrailName}
            options={guardrailOptions(guardrails, operation)}
            isRequired
            isDisabled={pending || operation === ""}
            placeholder="Choose a guardrail"
            // The vendor's own reference, beside the one-line summary rather
            // than instead of it: what the arguments below mean is a question
            // this form cannot answer and that page can.
            description={
              operation === "" ? (
                "Choose what you want checked first."
              ) : spec ? (
                <>
                  {spec.description}{" "}
                  <DocsLink href={guardrailDocsHref(spec)}>
                    {`${spec.display_name} reference`}
                  </DocsLink>
                </>
              ) : (
                "What runs the check."
              )
            }
            reserveMessage
          />
        </>
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
            description="What a caller sends as its profile. Suggested from what you chose; type your own if you prefer."
            reserveMessage
          />
          {secretBlocked ? (
            <InfoBanner tone="warning">
              {spec.display_name} needs a credential, and this gateway has no
              OTARI_SECRET_KEY set to encrypt one with. Set it and restart, then
              add this guardrail.
            </InfoBanner>
          ) : null}
          <GuardrailEnforcementFields
            value={enforcement}
            onChange={setEnforcement}
            workspaces={workspaces.data ?? []}
            isLoadingWorkspaces={workspaces.isPending && !workspaces.data}
            workspacesError={workspaces.data ? undefined : workspaces.error}
            isDisabled={pending}
          />
          {/* Four parts, in the order the form asks them: what this checks,
              what it is called, how it is enforced, and how the guardrail
              itself is set up. The rules are what stop each reading as more of
              the one above it. */}
          {createSpecs.length > 0 ? (
            <FormSectionRule label={`${spec.display_name} settings`} />
          ) : null}
          {createDecisions.length > 0 ? (
            <GuardrailParameterFields
              specs={createDecisions}
              guardrailName={guardrailName}
              operation={operation}
              scopeName={name || spec.display_name}
              values={setup.values}
              errors={setup.issues}
              disabled={pending || secretBlocked}
              storedSecrets={editing ? storedSecrets : undefined}
              onChange={setup.setValue}
            />
          ) : null}
          {/* A per-call argument with no default is still a decision: AnyLlm's
              whole configuration is one, and folding it away would leave that
              guardrail's form empty. */}
          {validateSplit.decisions.length > 0 ? (
            <GuardrailParameterFields
              specs={validateSplit.decisions}
              guardrailName={guardrailName}
              scopeName={name || spec.display_name}
              values={perCall.values}
              errors={perCall.issues}
              disabled={pending || secretBlocked}
              onChange={perCall.setValue}
            />
          ) : null}
          {/* Last in the dialog, and one of them: what the vendor already chose,
              what only some deployments set, and the escape hatch. A form that
              asks for all of it at once reads as a dozen decisions when there
              are two. */}
          <Disclosure
            key={extrasHaveError ? "open" : "closed"}
            heading="Advanced settings"
            isDefaultExpanded={extrasHaveError}
          >
            <div className="flex flex-col gap-4">
              {createSplit.extras.length > 0 ? (
                <GuardrailParameterFields
                  specs={createSplit.extras}
                  guardrailName={guardrailName}
                  operation={operation}
                  scopeName={name || spec.display_name}
                  values={setup.values}
                  errors={setup.issues}
                  disabled={pending || secretBlocked}
                  storedSecrets={editing ? storedSecrets : undefined}
                  onChange={setup.setValue}
                />
              ) : null}
              {validateSplit.extras.length > 0 ? (
                <GuardrailParameterFields
                  specs={validateSplit.extras}
                  guardrailName={guardrailName}
                  scopeName={name || spec.display_name}
                  values={perCall.values}
                  errors={perCall.issues}
                  disabled={pending || secretBlocked}
                  onChange={perCall.setValue}
                />
              ) : null}
              <GuardrailExtraJsonField
                value={perCall.extraJson}
                error={perCall.rawError}
                disabled={pending || secretBlocked}
                onChange={perCall.setExtraJson}
              />
            </div>
          </Disclosure>
        </>
      ) : null}
    </FormDialog>
  )
}
