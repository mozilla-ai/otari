import { useEffect, useState } from "react"
import type {
  GuardrailCatalog,
  GuardrailParameterSpec,
  OrganizationGuardrail,
  Workspace,
} from "@/client"
import { Button } from "@/design-system/actions/Button"
import { ConfirmDialog } from "@/design-system/feedback/ConfirmDialog"
import { ErrorBanner } from "@/design-system/feedback/ErrorBanner"
import { errorMessage } from "@/design-system/feedback/errorMessage"
import { FormDialog } from "@/design-system/feedback/FormDialog"
import { InfoBanner } from "@/design-system/feedback/InfoBanner"
import { Checkbox } from "@/design-system/forms/Checkbox"
import { Field } from "@/design-system/forms/Field"
import { INPUT_CLASS } from "@/design-system/forms/inputClass"
import { SecretField } from "@/design-system/forms/SecretField"
import { Select } from "@/design-system/forms/Select"
import { useDirtySnapshot } from "@/design-system/forms/useDirtySnapshot"
import { Badge } from "@/design-system/indicators/Badge"
import { SettingsGroup } from "@/design-system/layout/SettingsGroup"
import { FilterSelect } from "@/design-system/navigation/FilterSelect"
import { canManage } from "@/features/organization/roles"
import { GuardrailParametersSection } from "@/features/tools/GuardrailParametersSection"
import { GuardrailProfileField } from "@/features/tools/GuardrailProfileField"
import {
  buildValidateKwargs,
  findProfile,
  type ParameterErrors,
  type ParameterValues,
  parameterErrors,
  parameterSpecs,
  parseExtraJson,
  profileIdentity,
  type SeededParameters,
  seedParameters,
} from "@/features/tools/guardrailParameters"
import { useOrganizationContext } from "@/shared/api/organizations"
import {
  useCreateOrganizationGuardrail,
  useDeleteOrganizationGuardrail,
  useGuardrailProfiles,
  useOrganizationGuardrails,
  useUpdateOrganizationGuardrail,
} from "@/shared/api/tools"
import { useWorkspaces } from "@/shared/api/workspaces"

// The layer above the deployment-wide guardrail settings this card sits under.
// The settings above say where guardrails run; these say which ones run whether
// the caller asked for them or not, and in which workspaces.
//
// This is the one tool plane that adds rather than narrows, and the reason is
// worth keeping in view while editing: a guardrail is a restriction the
// organization accepts, not a capability it acquires, so an entry here can only
// ever make fewer requests succeed. That is why an entry may name an endpoint
// and a credential of its own where the workspace code-execution policy below
// may not. See `src/gateway/AGENTS.md`.

type Mode = "block" | "monitor"

const MODE_OPTIONS = [
  { value: "monitor", label: "Monitor" },
  { value: "block", label: "Block" },
]

const UNAVAILABLE_OPTIONS = [
  { value: "block", label: "Refuse the request" },
  { value: "monitor", label: "Serve it unchecked" },
]

/** What an entry's scope reads as, without making the reader count rows. */
function scopeLabel(
  guardrail: OrganizationGuardrail,
  workspaces: readonly Workspace[],
): string {
  if (guardrail.applies_to_all_workspaces) {
    return "Every workspace, including new ones"
  }
  const count = guardrail.workspace_ids.length
  if (count === 0) return "No workspaces yet"
  const named = guardrail.workspace_ids
    .map((id) => workspaces.find((workspace) => workspace.id === id)?.name)
    .filter((name): name is string => name !== undefined)
  return named.length === count ? named.join(", ") : `${count} workspaces`
}

/** The scope picker's two positions, which are the two the server stores. */
function WorkspaceScope({
  // Names which entry's scope this is, because the card renders one of these
  // per row plus one in the add form and the workspace names repeat in all of
  // them. Without it every checkbox on the card is labelled "Alpha".
  scopeName,
  everywhere,
  selected,
  workspaces,
  disabled,
  variant = "filter",
  onEverywhere,
  onToggle,
}: {
  /** Names the workspace group, so a box reads as "Beta" inside "prompt-injection". */
  scopeName: string
  everywhere: boolean
  selected: readonly string[]
  workspaces: readonly Workspace[]
  disabled?: boolean
  /**
   * Which half of the pair the picker is. A row on this card is a dense line
   * of toolbar controls; the dialog is a form, where a caption label beside
   * the control would be the one thing on it reading differently.
   */
  variant?: "filter" | "form"
  onEverywhere: (value: boolean) => void
  onToggle: (workspaceId: string) => void
}) {
  const scopeOptions = [
    { value: "all", label: "Every workspace" },
    { value: "chosen", label: "Chosen workspaces" },
  ]
  return (
    <div className="flex flex-col gap-2">
      {variant === "form" ? (
        <Select
          label="Runs in"
          value={everywhere ? "all" : "chosen"}
          onChange={(next) => onEverywhere(next === "all")}
          options={scopeOptions}
          isDisabled={disabled}
          reserveMessage={false}
        />
      ) : (
        <FilterSelect
          label="Runs in"
          value={everywhere ? "all" : "chosen"}
          onChange={(next) => onEverywhere(next === "all")}
          options={scopeOptions}
          disabled={disabled}
        />
      )}
      {everywhere ? null : (
        // A named group rather than a per-box aria-label. Each box is labelled
        // by the workspace name a reader can see, and the group says which
        // guardrail those names belong to; an aria-label on the box would have
        // replaced the visible text for assistive tech instead of qualifying it.
        <fieldset aria-label={scopeName} className="flex flex-wrap gap-3">
          {workspaces.map((workspace) => (
            <span key={workspace.id} className="text-sm text-muted">
              <Checkbox
                isSelected={selected.includes(workspace.id)}
                isDisabled={disabled}
                onChange={() => onToggle(workspace.id)}
              >
                {workspace.name}
              </Checkbox>
            </span>
          ))}
          {workspaces.length === 0 ? (
            <span className="text-caption">
              No workspaces to choose from yet.
            </span>
          ) : null}
        </fieldset>
      )}
    </div>
  )
}

/**
 * The `validate_kwargs` half of one entry's form: the typed values, the raw
 * editor beside them, and the messages a submit produced.
 *
 * A hook rather than five `useState` calls at each of the two call sites, which
 * is what keeps the seeding rule in one place: the row and the add form seed
 * from different sources but must both re-seed when the profile's schema
 * arrives, and a catalog that loads a moment after the card does is the ordinary
 * case rather than the edge one.
 */
function useParameterForm(
  specs: GuardrailParameterSpec[],
  stored: Record<string, unknown> | null | undefined,
  /** From `profileIdentity`, which says what counts as a different profile. */
  identity: string,
) {
  const [state, setState] = useState<SeededParameters>(() =>
    seedParameters(specs, stored),
  )
  const [issues, setIssues] = useState<ParameterErrors>({})
  const [rawError, setRawError] = useState<string | undefined>(undefined)

  // Two of the three dependencies are serialized, for the reason the workspace
  // scope below is: each is a fresh object on every fetch and on every catalog
  // read, so depending on them by reference would wipe a half-typed parameter
  // whenever any row on the card saved. Parsed back inside the effect so
  // nothing it touches is missing from the dependency list.
  //
  // The identity is the third, because the two above cannot separate two
  // profiles that declare the same parameters, which is the ordinary shape of a
  // pair differing only in the model it pins. Nothing inside the effect reads
  // it.
  const specsJson = JSON.stringify(specs)
  const storedJson = JSON.stringify(stored ?? {})
  // biome-ignore lint/correctness/useExhaustiveDependencies: identity is a re-seed trigger, not an input
  useEffect(() => {
    setState(
      seedParameters(
        JSON.parse(specsJson) as GuardrailParameterSpec[],
        JSON.parse(storedJson) as Record<string, unknown>,
      ),
    )
    setIssues({})
    setRawError(undefined)
  }, [identity, specsJson, storedJson])

  return {
    values: state.values,
    extraJson: state.extraJson,
    issues,
    rawError,
    setValue: (name: string, next: ParameterValues[string]) =>
      setState((current) => ({
        ...current,
        values: { ...current.values, [name]: next },
      })),
    setExtraJson: (next: string) =>
      setState((current) => ({ ...current, extraJson: next })),
    /**
     * Validate on submit and report whether the entry may be sent. Messages
     * appear here rather than on the first keystroke, which is what the forms
     * guide asks for.
     */
    check: (): boolean => {
      const found = parameterErrors(specs, state.values)
      const raw = parseExtraJson(state.extraJson).error
      setIssues(found)
      setRawError(raw)
      return raw === undefined && Object.keys(found).length === 0
    },
    build: () => buildValidateKwargs(specs, state.values, state.extraJson),
  }
}

function GuardrailRow({
  guardrail,
  catalog,
  workspaces,
  onSaved,
}: {
  guardrail: OrganizationGuardrail
  catalog: GuardrailCatalog | undefined
  workspaces: readonly Workspace[]
  onSaved: (message: string) => void
}) {
  const update = useUpdateOrganizationGuardrail()
  const remove = useDeleteOrganizationGuardrail()
  const [mode, setMode] = useState<Mode>(guardrail.mode as Mode)
  const [onUnavailable, setOnUnavailable] = useState<Mode>(
    guardrail.on_unavailable as Mode,
  )
  const [enabled, setEnabled] = useState(guardrail.enabled)
  const [everywhere, setEverywhere] = useState(
    guardrail.applies_to_all_workspaces,
  )
  const [scope, setScope] = useState<string[]>([...guardrail.workspace_ids])
  const [url, setUrl] = useState(guardrail.url ?? "")
  // Blank means "keep the stored credential". The field is write-only, so it
  // never shows what is stored, only whether something is.
  const [credential, setCredential] = useState("")
  const [error, setError] = useState("")
  const [isDeleteOpen, setDeleteOpen] = useState(false)
  const specs = parameterSpecs(catalog, guardrail.profile)
  const describedProfile = findProfile(catalog, guardrail.profile) !== undefined
  const parameters = useParameterForm(
    specs,
    guardrail.validate_kwargs,
    profileIdentity(catalog, guardrail.profile),
  )

  // Rehydrate from whatever the server last said, so the row never drifts from
  // the stored entry after a save.
  //
  // Keyed on the stored values and not on the `guardrail` object, which is what
  // `SearchToolsCard` does. Every row mutation invalidates the whole list, so a
  // refetch re-renders every row; TanStack Query's structural sharing is what
  // keeps an untouched row's object identity stable through that, and depending
  // on the values rather than the reference means a half-typed edit in one row
  // does not hang on that behavior staying true.
  const storedScope = guardrail.workspace_ids.join(",")
  useEffect(() => {
    setMode(guardrail.mode as Mode)
    setOnUnavailable(guardrail.on_unavailable as Mode)
    setEnabled(guardrail.enabled)
    setEverywhere(guardrail.applies_to_all_workspaces)
    // Rebuilt from the joined form rather than read off the row, so the effect
    // depends on the scope's *value*: the array is rebuilt by every fetch, and
    // depending on it would put object identity back in the dependency list.
    setScope(storedScope === "" ? [] : storedScope.split(","))
    setUrl(guardrail.url ?? "")
  }, [
    guardrail.mode,
    guardrail.on_unavailable,
    guardrail.enabled,
    guardrail.applies_to_all_workspaces,
    guardrail.url,
    storedScope,
  ])

  const busy = update.isPending || remove.isPending

  const save = () => {
    setError("")
    if (!parameters.check()) return
    update.mutate(
      {
        guardrailId: guardrail.id,
        body: {
          mode,
          on_unavailable: onUnavailable,
          enabled,
          applies_to_all_workspaces: everywhere,
          // An empty box clears the stored endpoint, which is the server's own
          // three-state rule for this field: omitted leaves it, "" clears it, a
          // value replaces it. Only sent when it differs from what is stored,
          // so saving a mode never rewrites the endpoint.
          ...(url.trim() === (guardrail.url ?? "") ? {} : { url: url.trim() }),
          // Sent only when the entry has a scope to carry: the server refuses a
          // list alongside "every workspace" rather than storing one that
          // decides nothing.
          ...(everywhere ? {} : { workspace_ids: scope }),
          // Omitted entirely when blank, so saving a mode never clears the
          // credential.
          ...(credential === "" ? {} : { credential }),
          // Sent whole every time, unlike the two fields above. Those are
          // write-only or nullable and need their omitted/cleared/replaced
          // states; this one is fully rendered by the form, so what the form
          // holds *is* the stored value and sending it back is a no-op when
          // nothing was touched. `null` is how the API clears it.
          validate_kwargs: parameters.build(),
        },
      },
      {
        onSuccess: () => {
          setCredential("")
          onSaved(`${guardrail.profile} saved`)
        },
        onError: (err) => setError(errorMessage(err)),
      },
    )
  }

  return (
    <div className="flex flex-col gap-2 py-4">
      <div className="flex flex-wrap items-center gap-2">
        <code className="font-mono text-body">{guardrail.profile}</code>
        <Badge tone="muted">{scopeLabel(guardrail, workspaces)}</Badge>
        {guardrail.url ? <Badge tone="muted">own endpoint</Badge> : null}
        {guardrail.has_credential ? (
          <Badge tone="muted">credential set</Badge>
        ) : null}
        {guardrail.enabled ? null : <Badge tone="warn">Paused</Badge>}
      </div>
      <div className="flex flex-wrap items-end gap-2">
        <FilterSelect
          label="Mode"
          value={mode}
          onChange={(next) => setMode(next as Mode)}
          options={MODE_OPTIONS}
          disabled={busy}
        />
        <FilterSelect
          label="If unreachable"
          value={onUnavailable}
          onChange={(next) => setOnUnavailable(next as Mode)}
          options={UNAVAILABLE_OPTIONS}
          disabled={busy}
        />
        <FilterSelect
          label="Status"
          value={enabled ? "on" : "off"}
          onChange={(next) => setEnabled(next === "on")}
          options={[
            { value: "on", label: "Active" },
            { value: "off", label: "Paused" },
          ]}
          disabled={busy}
        />
        <input
          type="text"
          inputMode="url"
          aria-label={`Endpoint for ${guardrail.profile}`}
          value={url}
          disabled={busy}
          placeholder="blank uses the URL above"
          onChange={(event) => setUrl(event.target.value)}
          className={`w-full sm:w-72 ${INPUT_CLASS}`}
        />
        <input
          type="password"
          autoComplete="new-password"
          aria-label={`New credential for ${guardrail.profile}`}
          value={credential}
          disabled={busy}
          placeholder={
            guardrail.has_credential ? "replace credential" : "add credential"
          }
          onChange={(event) => setCredential(event.target.value)}
          className={`w-full sm:w-52 ${INPUT_CLASS}`}
        />
      </div>
      <WorkspaceScope
        scopeName={guardrail.profile}
        everywhere={everywhere}
        selected={scope}
        workspaces={workspaces}
        disabled={busy}
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
        // whether it starts open: the section derives that once, and a catalog
        // that lands after the card did would otherwise leave a raw value
        // hidden behind a collapsed panel.
        key={`${describedProfile}:${specs.length}`}
        specs={specs}
        scopeName={guardrail.profile}
        values={parameters.values}
        errors={parameters.issues}
        extraJson={parameters.extraJson}
        extraJsonError={parameters.rawError}
        described={describedProfile}
        disabled={busy}
        onChange={parameters.setValue}
        onExtraJsonChange={parameters.setExtraJson}
      />
      <div className="flex flex-wrap items-center gap-2">
        <Button
          size="sm"
          variant="primary"
          aria-label={`Save ${guardrail.profile}`}
          isDisabled={busy}
          onPress={save}
        >
          {update.isPending ? "Saving…" : "Save"}
        </Button>
        <Button
          size="sm"
          variant="ghost"
          // Named per row, as the Save beside it is: the card is a list of
          // profiles, so a bare "Remove guardrail" is the same name N times.
          aria-label={`Remove ${guardrail.profile}`}
          isDisabled={busy}
          onPress={() => setDeleteOpen(true)}
        >
          Remove guardrail
        </Button>
      </div>
      {error ? (
        <span className="break-words text-caption text-danger">{error}</span>
      ) : null}

      <ConfirmDialog
        isOpen={isDeleteOpen}
        // Cleared on the way out: a refusal otherwise sits on the mutation
        // and greets the next open as if it had just happened.
        onOpenChange={(open) => {
          setDeleteOpen(open)
          if (!open) remove.reset()
        }}
        heading="Remove guardrail"
        // The stored mode, not the row's unsaved `mode`: this describes what is
        // in force, and a monitoring guardrail never blocked anything.
        body={
          guardrail.mode === "block"
            ? `${guardrail.profile} stops running on every request it covers, and its stored credential is removed with it. Requests it would have blocked are served.`
            : `${guardrail.profile} stops running on every request it covers, and its stored credential is removed with it. Requests it would have recorded go unchecked.`
        }
        confirmLabel="Remove permanently"
        isPending={remove.isPending}
        error={remove.error}
        onConfirm={() => {
          setError("")
          remove.mutate(guardrail.id, {
            onSuccess: () => {
              setDeleteOpen(false)
              onSaved(`${guardrail.profile} removed`)
            },
          })
        }}
      />
    </div>
  )
}

function AddGuardrailDialog({
  isOpen,
  onClose,
  catalog,
  catalogPending,
  workspaces,
  onSaved,
}: {
  isOpen: boolean
  onClose: () => void
  catalog: GuardrailCatalog | undefined
  catalogPending: boolean
  workspaces: readonly Workspace[]
  onSaved: (message: string) => void
}) {
  const create = useCreateOrganizationGuardrail()
  const [profile, setProfile] = useState("")
  const [mode, setMode] = useState<Mode>("monitor")
  const [url, setUrl] = useState("")
  const [credential, setCredential] = useState("")
  const [everywhere, setEverywhere] = useState(false)
  const [scope, setScope] = useState<string[]>([])
  const specs = parameterSpecs(catalog, profile)
  // An empty picker describes nothing, but its panel should not open on that
  // account: there is no profile yet for a raw parameter to belong to.
  const describedProfile =
    profile === "" || findProfile(catalog, profile) !== undefined
  // Nothing stored yet, so the fields start blank and re-seed whenever the
  // picker moves to another profile the catalog describes, whether or not that
  // profile's schema differs from the one left behind.
  const parameters = useParameterForm(
    specs,
    undefined,
    profileIdentity(catalog, profile),
  )

  // Everything the operator can change, in one snapshot: a field added to this
  // form would otherwise have to be remembered in a second place, and the
  // parameters are the half most easily forgotten.
  const { isDirty } = useDirtySnapshot({
    profile,
    mode,
    url,
    credential,
    everywhere,
    scope,
    values: parameters.values,
    extraJson: parameters.extraJson,
  })

  const submit = () => {
    const named = profile.trim()
    if (!parameters.check()) return
    create.mutate(
      {
        profile: named,
        mode,
        url: url.trim() === "" ? null : url.trim(),
        credential: credential === "" ? null : credential,
        validate_kwargs: parameters.build(),
        applies_to_all_workspaces: everywhere,
        workspace_ids: everywhere ? [] : scope,
      },
      {
        onSuccess: () => {
          onSaved(`${named} added`)
          onClose()
        },
      },
    )
  }

  return (
    <FormDialog
      isOpen={isOpen}
      onOpenChange={(open) => {
        if (!open) onClose()
      }}
      // `lg`, unlike the search-tool dialog beside it: the parameters section
      // is a variable-length list of controls plus a raw-JSON escape hatch,
      // which the small frame has no room for.
      size="lg"
      title="Mandated guardrail"
      submitLabel="Mandate a guardrail"
      onSubmit={submit}
      isPending={create.isPending}
      isSubmitDisabled={profile.trim() === ""}
      isDirty={isDirty}
      error={create.error}
    >
      <GuardrailProfileField
        catalog={catalog}
        pending={catalogPending}
        value={profile}
        onChange={setProfile}
      />
      <Select
        label="Mode"
        value={mode}
        onChange={(next) => setMode(next as Mode)}
        options={MODE_OPTIONS}
        description="A caller can tighten a mandated guardrail but never weaken it."
      />
      <Field
        label="Endpoint"
        value={url}
        onChange={setUrl}
        placeholder="blank uses the guardrails URL above"
        reserveMessage={false}
      />
      <SecretField
        label="Credential"
        value={credential}
        onChange={setCredential}
        description="Needs an https endpoint of its own, since the URL above may be a plain-http sidecar, and OTARI_SECRET_KEY set on the gateway."
      />
      <WorkspaceScope
        scopeName={profile || "New guardrail"}
        variant="form"
        everywhere={everywhere}
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
        // See the row above: the picker moving to a profile the catalog cannot
        // describe has to open the editor that is then the only place its
        // parameters can go.
        key={`${describedProfile}:${specs.length}`}
        specs={specs}
        scopeName={profile === "" ? "the new guardrail" : profile}
        values={parameters.values}
        errors={parameters.issues}
        extraJson={parameters.extraJson}
        extraJsonError={parameters.rawError}
        described={describedProfile}
        onChange={parameters.setValue}
        onExtraJsonChange={parameters.setExtraJson}
      />
    </FormDialog>
  )
}

export function OrganizationGuardrailsCard({
  onSaved,
}: {
  onSaved: (message: string) => void
}) {
  const context = useOrganizationContext()
  // The client half of the gate the service enforces, and it gates the *read*
  // too: these rows name the endpoints this gateway connects to and say which
  // carry a credential, so a member who cannot manage the organization cannot
  // see them either, and asking would earn a 403 over a form they cannot use.
  const manages = canManage(context.data)
  const guardrails = useOrganizationGuardrails(manages)
  // Behind the same gate for the same reason the entries are: nothing here is
  // asked for over a form the caller cannot use.
  const catalog = useGuardrailProfiles(manages)
  const workspaces = useWorkspaces()
  const [adding, setAdding] = useState(false)
  // Bumped on every open and used as the dialog's key, so the draft is cleared
  // on the way in rather than on the way out.
  const [openCount, setOpenCount] = useState(0)
  const openAdd = () => {
    setOpenCount((count) => count + 1)
    setAdding(true)
  }
  const entries = guardrails.data ?? []
  const known = workspaces.data ?? []

  return (
    <>
      {/* Outside the group, not inside it: `FormDialog` renders its trigger
          slot as a real element, and a group's rows are a `divide-y` container
          where one more child changes which row is last.

          Keyed on the open count, so each open remounts a blank form. */}
      {manages ? (
        <AddGuardrailDialog
          key={openCount}
          isOpen={adding}
          onClose={() => setAdding(false)}
          catalog={catalog.data}
          // `isFetched` rather than `isPending`: an errored query returns to
          // pending when its observers remount, which would leave the picker
          // stuck reading a service that already answered.
          catalogPending={!catalog.isFetched}
          workspaces={known}
          onSaved={onSaved}
        />
      ) : null}
      <SettingsGroup
        bounded
        title="Organization guardrails"
        description="Guardrails that run on every request from the workspaces below, whether the caller asked for them or not. They compose with the deployment settings above rather than replacing them: an entry with no endpoint of its own is sent to the guardrails URL set there, and an organization that mandates nothing leaves every request checked exactly as it is today."
        action={
          manages ? (
            <Button variant="primary" onPress={openAdd}>
              Mandate a guardrail
            </Button>
          ) : null
        }
      >
        {manages ? null : (
          <InfoBanner>
            Organization guardrails are set by an owner or admin of the
            organization.
          </InfoBanner>
        )}
        {manages ? (
          <>
            <ErrorBanner error={guardrails.error ?? workspaces.error} />
            {entries.map((guardrail) => (
              <GuardrailRow
                key={guardrail.id}
                guardrail={guardrail}
                catalog={catalog.data}
                workspaces={known}
                onSaved={onSaved}
              />
            ))}
            {entries.length === 0 && !guardrails.isLoading ? (
              <p className="py-4 text-sm text-muted">
                No organization guardrails, so only the guardrails a caller asks
                for run.
              </p>
            ) : null}
          </>
        ) : null}
      </SettingsGroup>
    </>
  )
}
