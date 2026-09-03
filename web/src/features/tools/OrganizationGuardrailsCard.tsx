import { Button } from "@heroui/react"
import { useEffect, useState } from "react"

import type { OrganizationGuardrail, Workspace } from "@/client"
import { canManage } from "@/features/organization/roles"
import {
  useCreateOrganizationGuardrail,
  useDeleteOrganizationGuardrail,
  useOrganizationContext,
  useOrganizationGuardrails,
  useUpdateOrganizationGuardrail,
  useWorkspaces,
} from "@/shared/api/hooks"
import { SettingsGroup } from "@/shared/components/surface"
import {
  Badge,
  Checkbox,
  ConfirmButton,
  ErrorBanner,
  errorMessage,
  FilterSelect,
  INPUT_CLASS,
  InfoBanner,
} from "@/shared/components/ui"

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
  onEverywhere,
  onToggle,
}: {
  /** Names the workspace group, so a box reads as "Beta" inside "prompt-injection". */
  scopeName: string
  everywhere: boolean
  selected: readonly string[]
  workspaces: readonly Workspace[]
  disabled: boolean
  onEverywhere: (value: boolean) => void
  onToggle: (workspaceId: string) => void
}) {
  return (
    <div className="flex flex-col gap-2">
      <FilterSelect
        label="Runs in"
        value={everywhere ? "all" : "chosen"}
        onChange={(next) => onEverywhere(next === "all")}
        options={[
          { value: "all", label: "Every workspace" },
          { value: "chosen", label: "Chosen workspaces" },
        ]}
        disabled={disabled}
      />
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

function GuardrailRow({
  guardrail,
  workspaces,
  onSaved,
}: {
  guardrail: OrganizationGuardrail
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
        <ConfirmButton
          confirmLabel="Remove"
          isPending={busy}
          onConfirm={() => {
            setError("")
            remove.mutate(guardrail.id, {
              onSuccess: () => onSaved(`${guardrail.profile} removed`),
              onError: (err) => setError(errorMessage(err)),
            })
          }}
        >
          Remove
        </ConfirmButton>
      </div>
      {error ? (
        <span className="break-words text-caption text-danger">{error}</span>
      ) : null}
    </div>
  )
}

function AddGuardrailForm({
  workspaces,
  onSaved,
}: {
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
  const [error, setError] = useState("")

  const submit = () => {
    setError("")
    const named = profile.trim()
    create.mutate(
      {
        profile: named,
        mode,
        url: url.trim() === "" ? null : url.trim(),
        credential: credential === "" ? null : credential,
        applies_to_all_workspaces: everywhere,
        workspace_ids: everywhere ? [] : scope,
      },
      {
        onSuccess: () => {
          setProfile("")
          setUrl("")
          setCredential("")
          setScope([])
          onSaved(`${named} added`)
        },
        onError: (err) => setError(errorMessage(err)),
      },
    )
  }

  return (
    <div className="flex flex-col gap-2 py-4">
      <span className="text-body">Mandate a guardrail</span>
      <div className="flex flex-wrap items-end gap-2">
        <input
          type="text"
          aria-label="Guardrail profile"
          value={profile}
          placeholder="profile, e.g. prompt-injection"
          disabled={create.isPending}
          onChange={(event) => setProfile(event.target.value)}
          className={`w-full sm:w-52 ${INPUT_CLASS}`}
        />
        <FilterSelect
          ariaLabel="Guardrail mode"
          value={mode}
          onChange={(next) => setMode(next as Mode)}
          options={MODE_OPTIONS}
          disabled={create.isPending}
        />
        <input
          type="text"
          inputMode="url"
          aria-label="Guardrails endpoint"
          value={url}
          disabled={create.isPending}
          placeholder="endpoint (blank uses the URL above)"
          onChange={(event) => setUrl(event.target.value)}
          className={`w-full sm:w-72 ${INPUT_CLASS}`}
        />
        <input
          type="password"
          autoComplete="new-password"
          aria-label="Guardrail credential"
          value={credential}
          disabled={create.isPending}
          placeholder="credential (needs an https endpoint)"
          onChange={(event) => setCredential(event.target.value)}
          className={`w-full sm:w-52 ${INPUT_CLASS}`}
        />
      </div>
      <WorkspaceScope
        scopeName={profile || "New guardrail"}
        everywhere={everywhere}
        selected={scope}
        workspaces={workspaces}
        disabled={create.isPending}
        onEverywhere={setEverywhere}
        onToggle={(workspaceId) =>
          setScope((current) =>
            current.includes(workspaceId)
              ? current.filter((id) => id !== workspaceId)
              : [...current, workspaceId],
          )
        }
      />
      <div className="flex items-center gap-2">
        <Button
          size="sm"
          variant="primary"
          isDisabled={profile.trim() === "" || create.isPending}
          onPress={submit}
        >
          {create.isPending ? "Adding…" : "Add"}
        </Button>
      </div>
      <span className="text-caption">
        The profile has to exist on the guardrails service. A caller can tighten
        a mandated guardrail but never weaken it. A credential needs an https
        endpoint of its own, since the URL above may be a plain-http sidecar,
        and <code className="font-mono">OTARI_SECRET_KEY</code> set on the
        gateway.
      </span>
      {error ? (
        <span className="break-words text-caption text-danger">{error}</span>
      ) : null}
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
  const workspaces = useWorkspaces()
  const entries = guardrails.data ?? []
  const known = workspaces.data ?? []

  return (
    <SettingsGroup
      title="Organization guardrails"
      description="Guardrails that run on every request from the workspaces below, whether the caller asked for them or not. They compose with the deployment settings above rather than replacing them: an entry with no endpoint of its own is sent to the guardrails URL set there, and an organization that mandates nothing leaves every request checked exactly as it is today."
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
          <AddGuardrailForm workspaces={known} onSaved={onSaved} />
        </>
      ) : null}
    </SettingsGroup>
  )
}
