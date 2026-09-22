import { useState } from "react"
import type {
  BuiltInGuardrailCatalog,
  GuardrailCatalog,
  OrganizationGuardrail,
  OrganizationGuardrailDefinition,
  Workspace,
} from "@/client"
import { Button } from "@/design-system/actions/Button"
import { FormDialog } from "@/design-system/feedback/FormDialog"
import { Field } from "@/design-system/forms/Field"
import { SecretField } from "@/design-system/forms/SecretField"
import { Select } from "@/design-system/forms/Select"
import { useDirtySnapshot } from "@/design-system/forms/useDirtySnapshot"
import { GuardrailParametersSection } from "@/features/guardrails/GuardrailParametersSection"
import { GuardrailProfileField } from "@/features/guardrails/GuardrailProfileField"
import {
  findProfile,
  parameterSpecs,
  profileIdentity,
} from "@/features/guardrails/guardrailParameters"
import { useGuardrailParameterForm } from "@/features/guardrails/useGuardrailParameterForm"
import { WorkspaceScope } from "@/features/guardrails/WorkspaceScope"
import {
  useCreateOrganizationGuardrail,
  useUpdateOrganizationGuardrail,
} from "@/shared/api/tools"

type Mode = "block" | "monitor"
/** Who runs the check: Otari, from a definition, or a guardrails service. */
export type MandateShape = "definition" | "remote"

const MODE_OPTIONS = [
  { value: "monitor", label: "Monitor" },
  { value: "block", label: "Block" },
]

const UNAVAILABLE_OPTIONS = [
  { value: "block", label: "Refuse the request" },
  { value: "monitor", label: "Serve it unchecked" },
]

/** The same two wire values, said the way each shape actually fails. */
const UNAVAILABLE_WORDS: Record<
  MandateShape,
  { label: string; description: string }
> = {
  definition: {
    label: "If it can't run",
    description:
      "When the guardrail could not be built, is switched off, or the vendor refuses the call. Only a blocking guardrail can refuse; a monitoring one always serves.",
  },
  remote: {
    label: "If unreachable",
    description:
      "When the guardrails service cannot be reached. Only a blocking guardrail can refuse; a monitoring one always serves.",
  },
}

export function mandateShape(mandate: OrganizationGuardrail): MandateShape {
  return mandate.definition_id ? "definition" : "remote"
}

/**
 * Say where a check runs, for which workspaces, and how hard it bites; or edit
 * that for one mandate.
 *
 * The shape comes first because the two answers read different sources: a
 * definition from the organization's own table, or a profile from the remote
 * service's list. Nothing that cannot apply to the chosen shape is drawn, and
 * the shape is create-only, as a provider key's provider is.
 *
 * Mounted only while open and keyed by its caller per open.
 */
export function MandateDialog({
  isOpen,
  onClose,
  mandate,
  definitions,
  isDefinitionsSettled,
  builtInCatalog,
  remoteCatalog,
  isRemoteCatalogPending,
  workspaces,
  onSetUpDefinition,
  onSaved,
}: {
  isOpen: boolean
  onClose: () => void
  /** The mandate being edited; absent to create one. */
  mandate?: OrganizationGuardrail
  definitions: readonly OrganizationGuardrailDefinition[]
  /** Whether the definitions read has answered, so the shape can be offered. */
  isDefinitionsSettled: boolean
  builtInCatalog: BuiltInGuardrailCatalog | undefined
  remoteCatalog: GuardrailCatalog | undefined
  isRemoteCatalogPending: boolean
  workspaces: readonly Workspace[]
  /** Opens the definition dialog, for an organization that has none yet. */
  onSetUpDefinition: () => void
  onSaved: (message: string) => void
}) {
  const isEdit = mandate !== undefined
  const create = useCreateOrganizationGuardrail()
  const update = useUpdateOrganizationGuardrail()
  const hasDefinitions = definitions.length > 0

  // Empty until the user picks one, and derived until then: the default waits
  // on the definitions read, and state seeded before it answered would be
  // wrong for the rest of the open.
  const [shapeChoice, setShapeChoice] = useState<MandateShape | "">(
    mandate ? mandateShape(mandate) : "",
  )
  const shape: MandateShape =
    shapeChoice || (hasDefinitions ? "definition" : "remote")

  const [definitionId, setDefinitionId] = useState(mandate?.definition_id ?? "")
  const [profile, setProfile] = useState(mandate?.profile ?? "")
  const [isProfileTouched, setProfileTouched] = useState(isEdit)
  const [mode, setMode] = useState<Mode>((mandate?.mode as Mode) ?? "monitor")
  const [onUnavailable, setOnUnavailable] = useState<Mode>(
    (mandate?.on_unavailable as Mode) ?? "block",
  )
  const [url, setUrl] = useState(mandate?.url ?? "")
  // Blank means "keep the stored credential": the field is write-only.
  const [credential, setCredential] = useState("")
  const [everywhere, setEverywhere] = useState(
    mandate?.applies_to_all_workspaces ?? false,
  )
  const [scope, setScope] = useState<string[]>([
    ...(mandate?.workspace_ids ?? []),
  ])

  const definition = definitions.find((row) => row.id === definitionId)
  const builtIn = builtInCatalog?.guardrails?.find(
    (spec) => spec.guardrail_name === definition?.guardrail_name,
  )
  // The per-call arguments, typed from whichever source describes them.
  const specs =
    shape === "definition"
      ? (builtIn?.validate_parameters ?? [])
      : parameterSpecs(remoteCatalog, profile)
  const isDescribed =
    shape === "definition"
      ? builtIn !== undefined || definitionId === ""
      : profile === "" || findProfile(remoteCatalog, profile) !== undefined
  const parameters = useGuardrailParameterForm(
    specs,
    mandate?.validate_kwargs,
    shape === "definition"
      ? `definition:${definitionId}`
      : profileIdentity(remoteCatalog, profile),
  )
  const { isDirty } = useDirtySnapshot({
    shape,
    definitionId,
    profile,
    mode,
    onUnavailable,
    url,
    credential,
    everywhere,
    scope,
    values: parameters.values,
    extraJson: parameters.extraJson,
  })

  const chooseDefinition = (next: string) => {
    setDefinitionId(next)
    if (!isProfileTouched) {
      setProfile(definitions.find((row) => row.id === next)?.name ?? "")
    }
  }

  const named = profile.trim()
  const isPending = create.isPending || update.isPending
  const isIncomplete =
    named === "" || (shape === "definition" && definitionId === "")

  const submit = () => {
    if (!parameters.check()) return
    const scopeBody = {
      applies_to_all_workspaces: everywhere,
      workspace_ids: everywhere ? [] : scope,
    }
    const done = {
      onSuccess: () => {
        onSaved(isEdit ? `${named} saved` : `${named} added`)
        onClose()
      },
    }
    if (!isEdit) {
      create.mutate(
        {
          profile: named,
          mode,
          on_unavailable: onUnavailable,
          validate_kwargs: parameters.build(),
          ...scopeBody,
          ...(shape === "definition"
            ? { definition_id: definitionId }
            : {
                url: url.trim() === "" ? null : url.trim(),
                credential: credential === "" ? null : credential,
              }),
        },
        done,
      )
      return
    }
    update.mutate(
      {
        guardrailId: mandate.id,
        body: {
          mode,
          on_unavailable: onUnavailable,
          applies_to_all_workspaces: everywhere,
          // The server refuses a list beside "every workspace".
          ...(everywhere ? {} : { workspace_ids: scope }),
          // Sent whole: the form renders all of it, so an untouched form sends
          // back what is stored, and a masked value keeps its secret.
          validate_kwargs: parameters.build(),
          ...(named === mandate.profile ? {} : { profile: named }),
          ...(shape === "definition"
            ? definitionId === mandate.definition_id
              ? {}
              : { definition_id: definitionId }
            : {
                // Omitted leaves the endpoint, "" clears it, a value replaces
                // it; only sent when it differs from what is stored.
                ...(url.trim() === (mandate.url ?? "")
                  ? {}
                  : { url: url.trim() }),
                // Omitted when blank, so an edit never clears a credential.
                ...(credential === "" ? {} : { credential }),
              }),
        },
      },
      done,
    )
  }

  const words = UNAVAILABLE_WORDS[shape]

  return (
    <FormDialog
      isOpen={isOpen}
      onOpenChange={(open) => {
        if (!open) onClose()
      }}
      // The per-call parameters are a variable-length list plus a raw editor.
      size="lg"
      title="Mandated guardrail"
      description="Runs on every request from the chosen workspaces, whether the caller asked for it or not."
      submitLabel={isEdit ? "Save mandate" : "Mandate a guardrail"}
      onSubmit={submit}
      isPending={isPending}
      isSubmitDisabled={isIncomplete || (!isEdit && !isDefinitionsSettled)}
      isDirty={isDirty}
      error={create.error ?? update.error}
    >
      {isEdit ? (
        <p className="text-body">
          <span className="text-muted">Runs on </span>
          {shape === "definition"
            ? "a guardrail you set up"
            : "your own service"}
        </p>
      ) : (
        <div className="flex flex-col gap-2">
          <Select
            label="Runs on"
            value={shape}
            onChange={(next) => setShapeChoice(next as MandateShape)}
            // Held until the definitions read answers, so the default does
            // not move under the cursor.
            isDisabled={!isDefinitionsSettled}
            options={[
              {
                value: "definition",
                label: "A guardrail you set up",
                // Disabled with its reason rather than hidden, so the choice
                // stays findable.
                isDisabled: !hasDefinitions,
              },
              { value: "remote", label: "Your own guardrails service" },
            ]}
            description={
              !isDefinitionsSettled
                ? "Reading the guardrails you have set up…"
                : !hasDefinitions
                  ? "You have not set up a guardrail yet, so only your own service is available."
                  : shape === "definition"
                    ? "Otari builds and runs the check itself."
                    : "Otari posts each check to a service you run."
            }
          />
          {isDefinitionsSettled && !hasDefinitions ? (
            <div>
              <Button size="sm" variant="ghost" onPress={onSetUpDefinition}>
                Set up a guardrail
              </Button>
            </div>
          ) : null}
        </div>
      )}

      {shape === "definition" ? (
        <Select
          label="Guardrail"
          value={definitionId}
          onChange={chooseDefinition}
          options={definitions.map((row) => ({
            value: row.id,
            label: row.name,
          }))}
          placeholder="Choose a guardrail"
          shouldReserveMessage={false}
        />
      ) : null}

      {shape === "definition" ? (
        <Field
          label="Profile a caller sends"
          value={profile}
          onChange={(next) => {
            setProfile(next)
            setProfileTouched(true)
          }}
          description="The name this check goes by in a request. Unique in the organization."
          shouldReserveMessage
        />
      ) : (
        <GuardrailProfileField
          catalog={remoteCatalog}
          isPending={isRemoteCatalogPending}
          value={profile}
          onChange={setProfile}
        />
      )}

      {shape === "remote" ? (
        <>
          <Field
            label="Endpoint"
            value={url}
            onChange={setUrl}
            placeholder="blank uses the deployment guardrails URL"
            shouldReserveMessage={false}
          />
          <SecretField
            label="Credential"
            value={credential}
            onChange={setCredential}
            placeholder={
              mandate?.has_credential ? "stored; blank keeps it" : undefined
            }
            description="Needs an https endpoint of its own, since the deployment URL may be a plain-http sidecar, and OTARI_SECRET_KEY set on the gateway."
          />
        </>
      ) : null}

      <Select
        label="Mode"
        value={mode}
        onChange={(next) => setMode(next as Mode)}
        options={MODE_OPTIONS}
        description="A caller can tighten a mandated guardrail but never weaken it."
      />
      <Select
        label={words.label}
        value={onUnavailable}
        onChange={(next) => setOnUnavailable(next as Mode)}
        options={UNAVAILABLE_OPTIONS}
        description={words.description}
      />
      <WorkspaceScope
        scopeName={named || "New guardrail"}
        variant="form"
        appliesEverywhere={everywhere}
        selected={scope}
        workspaces={workspaces}
        onEverywhere={setEverywhere}
        onToggle={(workspaceId) =>
          setScope((current) =>
            current.includes(workspaceId)
              ? current.filter((id) => id !== workspaceId)
              : [...current, workspaceId],
          )
        }
      />
      <GuardrailParametersSection
        // Remounted when the panel's shape changes, which is what recomputes
        // whether it starts open.
        key={`${shape}:${isDescribed}:${specs.length}`}
        specs={specs}
        scopeName={named === "" ? "the new guardrail" : named}
        values={parameters.values}
        errors={parameters.issues}
        extraJson={parameters.extraJson}
        extraJsonError={parameters.rawError}
        isDescribed={isDescribed}
        extraJsonDescription={
          shape === "definition"
            ? "Handed to the guardrail with each check, under whatever the fields above set."
            : undefined
        }
        onChange={parameters.setValue}
        onExtraJsonChange={parameters.setExtraJson}
      />
    </FormDialog>
  )
}
