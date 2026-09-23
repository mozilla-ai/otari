import { useEffect, useState } from "react"
import type {
  GuardrailCatalog,
  OrganizationGuardrail,
  Workspace,
} from "@/client"
import { Button } from "@/design-system/actions/Button"
import { ConfirmDialog } from "@/design-system/feedback/ConfirmDialog"
import { ErrorBanner } from "@/design-system/feedback/ErrorBanner"
import { errorMessage } from "@/design-system/feedback/errorMessage"
import { InfoBanner } from "@/design-system/feedback/InfoBanner"
import { INPUT_CLASS } from "@/design-system/forms/inputClass"
import { Badge } from "@/design-system/indicators/Badge"
import { SettingsGroup } from "@/design-system/layout/SettingsGroup"
import { FilterSelect } from "@/design-system/navigation/FilterSelect"
import { DefinitionDialog } from "@/features/guardrails/DefinitionDialog"
import { GuardrailParametersSection } from "@/features/guardrails/GuardrailParametersSection"
import {
  findProfile,
  parameterSpecs,
  profileIdentity,
} from "@/features/guardrails/guardrailParameters"
import { MandateDialog } from "@/features/guardrails/MandateDialog"
import { useGuardrailParameterForm } from "@/features/guardrails/useGuardrailParameterForm"
import {
  SELECT_SLOT,
  scopeLabel,
  WorkspaceScope,
} from "@/features/guardrails/WorkspaceScope"
import { canManage } from "@/features/organization/roles"
import {
  useBuiltInGuardrailCatalog,
  useOrganizationGuardrailDefinitions,
} from "@/shared/api/guardrails"
import { useOrganizationContext } from "@/shared/api/organizations"
import {
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
  const parameters = useGuardrailParameterForm(
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

  const isBusy = update.isPending || remove.isPending

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
      {/* Two widths down the entry rather than six. Each control sized to its
          own content put 101, 178, 90, 288 and 208px in one row, and the scope
          select below it stretched to the full 1126: the reader gets a new edge
          per control and the row reads as debris. `SELECT_SLOT` is the common
          one and the endpoint keeps the wider box, because a URL is the one
          value here that is long. This is a rhythm, not the control lane
          `SettingRow` draws: the entry is a dense row rather than a settings
          list, and adopting the lane would make it eight rows tall, which is a
          question for the design of an entry and not for its alignment. */}
      <div className="flex flex-wrap items-end gap-2">
        <div className={SELECT_SLOT}>
          <FilterSelect
            label="Mode"
            value={mode}
            onChange={(next) => setMode(next as Mode)}
            options={MODE_OPTIONS}
            disabled={isBusy}
            fullWidth
          />
        </div>
        <div className={SELECT_SLOT}>
          <FilterSelect
            label="If unreachable"
            value={onUnavailable}
            onChange={(next) => setOnUnavailable(next as Mode)}
            options={UNAVAILABLE_OPTIONS}
            disabled={isBusy}
            fullWidth
          />
        </div>
        <div className={SELECT_SLOT}>
          <FilterSelect
            label="Status"
            value={enabled ? "on" : "off"}
            onChange={(next) => setEnabled(next === "on")}
            options={[
              { value: "on", label: "Active" },
              { value: "off", label: "Paused" },
            ]}
            disabled={isBusy}
            fullWidth
          />
        </div>
        <input
          type="text"
          inputMode="url"
          aria-label={`Endpoint for ${guardrail.profile}`}
          value={url}
          disabled={isBusy}
          placeholder="blank uses the URL above"
          onChange={(event) => setUrl(event.target.value)}
          className={`w-full sm:w-72 ${INPUT_CLASS}`}
        />
        <input
          type="password"
          autoComplete="new-password"
          aria-label={`New credential for ${guardrail.profile}`}
          value={credential}
          disabled={isBusy}
          placeholder={
            guardrail.has_credential ? "replace credential" : "add credential"
          }
          onChange={(event) => setCredential(event.target.value)}
          className={`${SELECT_SLOT} ${INPUT_CLASS}`}
        />
      </div>
      <WorkspaceScope
        scopeName={guardrail.profile}
        appliesEverywhere={everywhere}
        selected={scope}
        workspaces={workspaces}
        disabled={isBusy}
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
        isDescribed={describedProfile}
        disabled={isBusy}
        onChange={parameters.setValue}
        onExtraJsonChange={parameters.setExtraJson}
      />
      <div className="flex flex-wrap items-center gap-2">
        <Button
          size="sm"
          variant="primary"
          aria-label={`Save ${guardrail.profile}`}
          isDisabled={isBusy}
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
          isDisabled={isBusy}
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
  const builtInCatalog = useBuiltInGuardrailCatalog(manages)
  const definitions = useOrganizationGuardrailDefinitions(manages)
  const workspaces = useWorkspaces()
  const [open, setOpen] = useState<"" | "mandate" | "definition">("")
  // Bumped on every open and used as the dialog's key, so the draft is cleared
  // on the way in rather than on the way out.
  const [openCount, setOpenCount] = useState(0)
  const openDialog = (which: "mandate" | "definition") => {
    setOpenCount((count) => count + 1)
    setOpen(which)
  }
  const entries = guardrails.data ?? []
  const defined = definitions.data ?? []
  const known = workspaces.data ?? []

  return (
    <>
      {/* Outside the group, not inside it: `FormDialog` renders its trigger
          slot as a real element, and a group's rows are a `divide-y` container
          where one more child changes which row is last.

          Keyed on the open count, so each open remounts a blank form. */}
      {manages ? (
        <MandateDialog
          key={`mandate-${openCount}`}
          isOpen={open === "mandate"}
          onClose={() => setOpen("")}
          definitions={defined}
          // `isFetched` rather than `isPending`: an errored query returns to
          // pending when its observers remount, which would leave the control
          // stuck waiting on a read that already answered.
          isDefinitionsSettled={definitions.isFetched}
          builtInCatalog={builtInCatalog.data}
          remoteCatalog={catalog.data}
          isRemoteCatalogPending={!catalog.isFetched}
          workspaces={known}
          onSetUpDefinition={() => openDialog("definition")}
          onSaved={onSaved}
        />
      ) : null}
      {manages ? (
        <DefinitionDialog
          key={`definition-${openCount}`}
          isOpen={open === "definition"}
          onClose={() => setOpen("")}
          catalog={builtInCatalog.data}
          isCatalogPending={!builtInCatalog.isFetched}
          takenNames={defined.map((row) => row.name)}
          onSaved={(saved) => onSaved(`${saved.name} set up`)}
        />
      ) : null}
      <SettingsGroup
        isBounded
        title="Organization guardrails"
        description="Guardrails that run on every request from the workspaces below, whether the caller asked for them or not. They compose with the deployment settings above rather than replacing them: an entry with no endpoint of its own is sent to the guardrails URL set there, and an organization that mandates nothing leaves every request checked exactly as it is today."
        action={
          manages ? (
            <div className="flex flex-wrap gap-2">
              <Button variant="ghost" onPress={() => openDialog("definition")}>
                Configure guardrail
              </Button>
              <Button variant="primary" onPress={() => openDialog("mandate")}>
                Mandate a guardrail
              </Button>
            </div>
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
