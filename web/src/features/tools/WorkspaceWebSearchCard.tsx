import type { UpdateWorkspaceWebSearchConfigRequest } from "@/client"
import { canManageWorkspace } from "@/features/organization/roles"
import {
  ceilingParser,
  type Parse,
  PolicyRow,
  parsePhrase,
} from "@/features/tools/PolicyRow"
import { usePolicyWriter } from "@/features/tools/usePolicyWriter"
import { useOrganizationContext } from "@/shared/api/organizations"
import {
  useClearWorkspaceWebSearchConfig,
  useSetWorkspaceWebSearchConfig,
  useWorkspaceWebSearchConfig,
} from "@/shared/api/tools"
import { InfoBanner } from "@/shared/components/feedback/InfoBanner"
import { SettingRow } from "@/shared/components/layout/SettingRow"
import { SettingsGroup } from "@/shared/components/layout/SettingsGroup"
import { FilterSelect } from "@/shared/components/navigation/FilterSelect"
import { useSelectedWorkspace } from "@/shared/hooks/SelectedWorkspace"
import { useAutosave } from "@/shared/hooks/useAutosave"

// The layer above the deployment-wide web-search settings this group sits
// under: the settings above say which backend runs a search, this says which
// workspaces may ask for one and how far it may reach. A row can only narrow,
// so there is no control here that turns anything on the deployment has not
// configured.
//
// Three states, not two, which is why the first control is a select rather than
// a toggle: a workspace can be allowed, blocked, or carry no row at all.
// "Deployment default" is the last of those and is a delete, not a saved
// `enabled: true`. While it is chosen there is no row to narrow, so the four
// rows below it have nothing to write and are disabled.

type Stance = "default" | "allowed" | "blocked"

// The server's own bounds (`workspace_web_search_service`): a ceiling above
// what the backend honors could never take effect, and the list bound stops one
// workspace's row growing without limit.
export const MAX_RESULTS = 20
const MAX_DOMAINS = 100

// Anything that means the entry is not a bare host. The server compares each
// entry against a result URL's hostname, so a scheme, port or path matches
// nothing at all: on a block-list that is a guardrail that reads as configured
// and blocks nothing. Refused here as well as server-side so the message lands
// on the field rather than arriving as a 422 banner.
const NOT_IN_A_HOSTNAME = /[/:@?#*\\\s]/

// Comma-separated in the form, a list on the wire. Blank entries are dropped
// rather than sent, so a trailing comma is not a domain named "". A leading dot
// is stripped, matching the server: an entry already covers its subdomains, so
// `.example.com` is the same rule in cookie syntax.
const parseDomains: Parse<string[] | null> = (raw) => {
  const hosts = raw
    .split(",")
    .map((host) => host.trim().toLowerCase().replace(/^\.+/, ""))
    .filter((host) => host !== "")
  const malformed = hosts.find((host) => NOT_IN_A_HOSTNAME.test(host))
  if (malformed !== undefined) {
    return {
      value: null,
      error: `"${malformed}" is not a bare hostname. Give a domain such as example.com, with no scheme, port or path.`,
    }
  }
  if (hosts.length > MAX_DOMAINS) {
    return { value: null, error: `At most ${MAX_DOMAINS} domains.` }
  }
  return { value: hosts.length > 0 ? hosts : null, error: "" }
}

/**
 * Whether requests billed to this workspace may search the web, and how far a
 * search may reach.
 *
 * Blocking covers both doors: the `otari_web_search` tool and `POST /v1/search`.
 * The rest narrows the in-loop tool only. Nothing here grants a backend the
 * deployment has not configured, and nothing here holds a credential.
 */
export function WorkspaceWebSearchCard({ docsHref }: { docsHref: string }) {
  const { selected, isLoading: workspaceLoading } = useSelectedWorkspace()
  const context = useOrganizationContext()
  // The client half of the gate the service enforces, and it gates the *read*
  // too: the row is the workspace's posture rather than one member's allowance,
  // so a member who cannot manage the workspace cannot see it either, and
  // asking would earn a 403 banner over a form they cannot use.
  const manages = canManageWorkspace(context.data, selected?.role)
  const workspaceId = selected && manages ? selected.workspace_id : null
  const query = useWorkspaceWebSearchConfig(workspaceId)
  const setConfig = useSetWorkspaceWebSearchConfig()
  const clearConfig = useClearWorkspaceWebSearchConfig()
  const stanceSave = useAutosave()
  // One writer for the group: a PUT replaces the whole row, so two rows saving
  // at once would each carry the other's pre-save value.
  const write = usePolicyWriter({
    server: query.data,
    resetKey: selected?.workspace_id ?? "",
    toBody: (stored) => ({
      enabled: stored.enabled,
      max_results: stored.max_results,
      purpose_hint: stored.purpose_hint,
      allowed_domains: stored.allowed_domains,
      blocked_domains: stored.blocked_domains,
      // Not editable here: an opaque per-backend bag with no form that could
      // validate it, preserved so a save from the dashboard never clears a
      // value set over the API.
      provider_options: stored.provider_options,
    }),
    put: (body: UpdateWorkspaceWebSearchConfigRequest) =>
      setConfig.mutateAsync({
        workspaceId: selected?.workspace_id as string,
        body,
      }),
  })

  if (!selected) {
    return (
      <InfoBanner>
        {workspaceLoading
          ? "Reading the workspaces you belong to."
          : "Per-workspace web search is set on a workspace you belong to. An owner or admin can add you to one on the Workspaces page."}
      </InfoBanner>
    )
  }

  if (!manages) {
    return (
      <InfoBanner>
        Web search for {selected.name} is set by an owner or admin of the
        workspace, or of the organization.
      </InfoBanner>
    )
  }

  const config = query.data
  const stance: Stance = !config?.configured
    ? "default"
    : config.enabled
      ? "allowed"
      : "blocked"

  // Disabled until the read has succeeded. Without that a failed GET leaves the
  // rows sitting at "Deployment default" over a workspace that may well have a
  // stored row, and one blur issues the write that drops it.
  const unreadable = query.isLoading || query.isError || !config
  const narrowingDisabled = unreadable || stance === "default"

  // `enabled` is the one field a patch always restates: the stance select is
  // the only control that changes it, and every other row must not flip it.
  const commitField = (patch: Partial<UpdateWorkspaceWebSearchConfigRequest>) =>
    write({ enabled: stance !== "blocked", ...patch })

  const setStance = (next: Stance) =>
    void stanceSave.run(() =>
      next === "default"
        ? clearConfig.mutateAsync({ workspaceId: selected.workspace_id })
        : commitField({ enabled: next === "allowed" }),
    )

  return (
    <SettingsGroup
      bounded
      title="This workspace"
      description={`Narrows what the deployment allows for requests billed to ${selected.name}. Never widens it, and holds no credential.`}
      docsHref={docsHref}
    >
      {config && !config.web_search_configured ? (
        <div className="px-4 py-3">
          <InfoBanner>
            This deployment has no in-loop search backend configured, so
            otari_web_search is unavailable here whatever this workspace allows.
            Blocking still takes effect on POST /v1/search.
          </InfoBanner>
        </div>
      ) : null}

      <SettingRow
        label="Web search"
        help="Allow or block the otari_web_search tool and POST /v1/search for requests billed here."
        error={
          stanceSave.error ||
          (query.isError
            ? "Could not read this workspace's setting. Reload before editing."
            : "")
        }
        control={
          <FilterSelect
            ariaLabel="Web search for this workspace"
            value={stance}
            onChange={(next) => setStance(next as Stance)}
            options={[
              { value: "default", label: "Deployment default" },
              { value: "allowed", label: "Allowed" },
              // Named for what it covers: an admin choosing this is also
              // switching off the workspace's POST /v1/search calls, which
              // "Blocked" alone would not have told them.
              { value: "blocked", label: "Blocked (tool and /v1/search)" },
            ]}
            disabled={unreadable || stanceSave.isSaving}
          />
        }
      />

      <PolicyRow
        key={`results-${selected.workspace_id}`}
        label="Max results"
        help="Lowers how many results one search returns. Never raises it."
        placeholder="10"
        numeric
        committed={
          config?.max_results == null ? "" : String(config.max_results)
        }
        parse={ceilingParser(MAX_RESULTS, "results")}
        commit={(max_results) => commitField({ max_results })}
        disabled={narrowingDisabled}
      />
      <PolicyRow
        key={`hint-${selected.workspace_id}`}
        label="Prompt hint"
        help="Used when a request declares otari_web_search without a hint of its own."
        placeholder="Prefer official sources"
        committed={config?.purpose_hint ?? ""}
        parse={parsePhrase}
        commit={(purpose_hint) => commitField({ purpose_hint })}
        disabled={narrowingDisabled}
      />
      <PolicyRow
        key={`allowed-${selected.workspace_id}`}
        label="Allowed domains"
        help="Results are kept only from these; a request's own list is narrowed to both."
        placeholder="mozilla.org, wikipedia.org"
        machine
        committed={(config?.allowed_domains ?? []).join(", ")}
        parse={parseDomains}
        commit={(allowed_domains) => commitField({ allowed_domains })}
        disabled={narrowingDisabled}
      />
      <PolicyRow
        key={`blocked-${selected.workspace_id}`}
        label="Blocked domains"
        help="Always dropped, whatever a request asks for."
        placeholder="reddit.com, pinterest.com"
        machine
        committed={(config?.blocked_domains ?? []).join(", ")}
        parse={parseDomains}
        commit={(blocked_domains) => commitField({ blocked_domains })}
        disabled={narrowingDisabled}
      />
    </SettingsGroup>
  )
}
