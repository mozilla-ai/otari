import { useState } from "react"

import type {
  BuiltInGuardrailCatalog,
  OrganizationGuardrailDefinition,
} from "@/client"
import { FormDialog } from "@/design-system/feedback/FormDialog"
import { InfoBanner } from "@/design-system/feedback/InfoBanner"
import { Field } from "@/design-system/forms/Field"
import { Select } from "@/design-system/forms/Select"
import { useDirtySnapshot } from "@/design-system/forms/useDirtySnapshot"
import { Disclosure } from "@/design-system/navigation/Disclosure"
import {
  checkLabel,
  createFieldLayout,
  definableGuardrails,
  guardrailChecks,
  guardrailsForCheck,
  suggestDefinitionName,
} from "@/features/guardrails/builtInCatalog"
import {
  buildCreateKwargs,
  definitionFieldSpecs,
  heldSecrets,
  seedableArguments,
} from "@/features/guardrails/definitionForm"
import { GuardrailParameterFields } from "@/features/guardrails/GuardrailParameterFields"
import { suggestedCreateKwargs } from "@/features/guardrails/guardrailFieldSuggestions"
import {
  parameterLabel,
  seedParameters,
} from "@/features/guardrails/guardrailParameters"
import { useGuardrailParameterForm } from "@/features/guardrails/useGuardrailParameterForm"
import {
  useCreateOrganizationGuardrailDefinition,
  useUpdateOrganizationGuardrailDefinition,
} from "@/shared/api/guardrails"

/** A guardrail as the picker names it: the product, then who makes it. */
function guardrailOptionLabel(spec: { display_name: string; vendor?: string }) {
  return spec.vendor
    ? `${spec.display_name} · ${spec.vendor}`
    : spec.display_name
}

/**
 * Set up a guardrail Otari builds and runs itself, or edit one.
 *
 * No question here about who runs the check: that is the mandate's question,
 * and a definition is only ever "Otari runs it". The check and the guardrail
 * are create-only, because a definition's arguments belong to its guardrail and
 * a changed guardrail would re-split them under another class.
 *
 * Mounted only while open and keyed by its caller per open, so every open
 * starts from the stored row or from blank.
 */
export function DefinitionDialog({
  isOpen,
  onClose,
  definition,
  catalog,
  isCatalogPending,
  takenNames,
  onSaved,
}: {
  isOpen: boolean
  onClose: () => void
  /** The definition being edited; absent to set one up. */
  definition?: OrganizationGuardrailDefinition
  catalog: BuiltInGuardrailCatalog | undefined
  isCatalogPending: boolean
  /** Names the organization already uses, so the suggestion avoids them. */
  takenNames: readonly string[]
  onSaved: (saved: OrganizationGuardrailDefinition) => void
}) {
  const isEdit = definition !== undefined
  const create = useCreateOrganizationGuardrailDefinition()
  const update = useUpdateOrganizationGuardrailDefinition()
  const [guardrailName, setGuardrailName] = useState(
    definition?.guardrail_name ?? "",
  )
  const [check, setCheck] = useState("")
  const [name, setName] = useState(definition?.name ?? "")
  // A suggestion follows the guardrail until the user types a name of their own.
  const [isNameTouched, setNameTouched] = useState(isEdit)

  const spec = definableGuardrails(catalog).find(
    (candidate) => candidate.guardrail_name === guardrailName,
  )
  const layout = createFieldLayout(spec)
  // Unreadable secrets are no secrets to keep: the form asks for them again.
  const held =
    definition?.secrets_decryptable === false
      ? new Set<string>()
      : heldSecrets(definition)
  const fields = definitionFieldSpecs(layout.fields, held)
  const advanced = definitionFieldSpecs(layout.advanced, held)
  const specs = [...fields, ...advanced]
  // A new definition starts with the check just picked already ticked, where
  // the vendor documents which of its own keys means that check.
  const seed = definition
    ? seedableArguments(specs, definition)
    : suggestedCreateKwargs(specs, guardrailName, check)
  const parameters = useGuardrailParameterForm(specs, seed, guardrailName)
  const { isDirty } = useDirtySnapshot({
    check,
    guardrailName,
    name,
    values: parameters.values,
  })

  const scopeName = name.trim() || spec?.display_name || "the new guardrail"
  const isPending = create.isPending || update.isPending

  const chooseGuardrail = (next: string) => {
    setGuardrailName(next)
    if (!isNameTouched) {
      setName(next === "" ? "" : suggestDefinitionName(next, takenNames))
    }
  }

  const submit = () => {
    if (!parameters.check()) return
    const kwargs = buildCreateKwargs(specs, parameters.values, held)
    const trimmed = name.trim()
    const done = {
      onSuccess: (saved: OrganizationGuardrailDefinition) => {
        onSaved(saved)
        onClose()
      },
    }
    if (!isEdit) {
      create.mutate(
        { name: trimmed, guardrail_name: guardrailName, create_kwargs: kwargs },
        done,
      )
      return
    }
    // The arguments go only when they changed. Left out, the gateway neither
    // rewrites nor reads the stored credentials, which is what lets a rename
    // succeed on a deployment whose secret key has moved.
    const untouched = buildCreateKwargs(
      specs,
      seedParameters(specs, seed).values,
      held,
    )
    const argumentsChanged =
      definition.secrets_decryptable === false ||
      JSON.stringify(kwargs) !== JSON.stringify(untouched)
    update.mutate(
      {
        definitionId: definition.id,
        body: {
          ...(trimmed === definition.name ? {} : { name: trimmed }),
          ...(argumentsChanged ? { create_kwargs: kwargs } : {}),
        },
      },
      done,
    )
  }

  const checks = guardrailChecks(catalog)
  const choices = guardrailsForCheck(catalog, check)

  return (
    <FormDialog
      isOpen={isOpen}
      onOpenChange={(open) => {
        if (!open) onClose()
      }}
      // A variable-length list of typed fields, which the default frame has no
      // room for.
      size="lg"
      title={isEdit ? "Guardrail" : "New guardrail"}
      description="Otari builds and runs this check itself. It changes no request until you choose where it runs."
      submitLabel={isEdit ? "Save guardrail" : "Set up guardrail"}
      onSubmit={submit}
      isPending={isPending}
      isSubmitDisabled={guardrailName === "" || name.trim() === ""}
      isDirty={isDirty}
      error={create.error ?? update.error}
    >
      {definition?.secrets_decryptable === false ? (
        <InfoBanner tone="warning">
          This gateway cannot read the stored credentials, so the guardrail is
          not running. Type them again, or restore the OTARI_SECRET_KEY they
          were saved under.
        </InfoBanner>
      ) : null}
      {isEdit ? (
        <p className="text-body">
          <span className="text-muted">Guardrail </span>
          {spec ? guardrailOptionLabel(spec) : definition.guardrail_name}
        </p>
      ) : (
        <>
          <Select
            label="What do you want checked?"
            value={check}
            onChange={(next) => {
              setCheck(next)
              chooseGuardrail("")
            }}
            options={checks.map((value) => ({
              value,
              label: checkLabel(value),
            }))}
            placeholder={
              isCatalogPending ? "Loading checks…" : "Choose a check"
            }
            isDisabled={isCatalogPending}
            shouldReserveMessage={false}
          />
          <Select
            label="Which guardrail?"
            value={guardrailName}
            onChange={chooseGuardrail}
            options={choices.map((choice) => ({
              value: choice.guardrail_name,
              label: guardrailOptionLabel(choice),
            }))}
            placeholder="Choose a guardrail"
            isDisabled={check === ""}
            description={spec?.description}
            shouldReserveMessage={false}
          />
        </>
      )}
      {spec || isEdit ? (
        <Field
          label="Name"
          value={name}
          onChange={(next) => {
            setName(next)
            setNameTouched(true)
          }}
          description="Your own label. A mandate picks the guardrail by this name."
          shouldReserveMessage
        />
      ) : null}
      {fields.length > 0 ? (
        <GuardrailParameterFields
          specs={fields}
          scopeName={scopeName}
          values={parameters.values}
          errors={parameters.issues}
          disabled={isPending}
          guardrailName={guardrailName}
          operation={check}
          onChange={parameters.setValue}
        />
      ) : null}
      {/* Only where it holds a field. An argument the form cannot store is
          named inside it, and alone that is not worth a section to open. */}
      {advanced.length > 0 ? (
        <Disclosure heading="Advanced">
          <div className="flex flex-col gap-3">
            <GuardrailParameterFields
              specs={advanced}
              scopeName={scopeName}
              values={parameters.values}
              errors={parameters.issues}
              disabled={isPending}
              guardrailName={guardrailName}
              operation={check}
              onChange={parameters.setValue}
            />
            {layout.unstorable.length > 0 ? (
              <p className="text-caption">
                {`Not offered here: ${layout.unstorable
                  .map(parameterLabel)
                  .join(", ")}. `}
                It takes a live object rather than a setting, so a stored
                guardrail cannot hold it.
              </p>
            ) : null}
          </div>
        </Disclosure>
      ) : null}
    </FormDialog>
  )
}
