import type { ReactNode } from "react"
import type {
  UpdateWorkspaceWebSearchConfigRequest,
  WorkspaceWebSearchConfig,
} from "@/client"
import { ErrorBanner } from "@/design-system/feedback/ErrorBanner"
import { Toggle } from "@/design-system/forms/Toggle"
import { SettingRow } from "@/design-system/layout/SettingRow"
import { SettingsGroup } from "@/design-system/layout/SettingsGroup"
import { canManageWorkspace } from "@/features/organization/roles"
import { AdvancedRows } from "@/features/tools/AdvancedRows"
import {
  ceilingParser,
  type Parse,
  PolicyRow,
} from "@/features/tools/PolicyRow"
import { usePolicyWriter } from "@/features/tools/usePolicyWriter"
import { WorkspaceToolStatus } from "@/features/tools/WorkspaceToolStatus"
import { useOrganizationContext } from "@/shared/api/organizations"
import {
  useClearWorkspaceWebSearchConfig,
  useSetWorkspaceWebSearchConfig,
  useWorkspaceWebSearchConfig,
} from "@/shared/api/tools"
import { useSelectedWorkspace } from "@/shared/hooks/SelectedWorkspace"
import { useAutosave } from "@/shared/hooks/useAutosave"

// The stored row has three states (allowed, blocked, none) and can also narrow
// results and domains. The card shows the switch with the narrowing under
// Advanced; the purpose hint and provider options are set through the API.
//
// What "none" means depends on the deployment. Standalone, no row narrows
// nothing, so it reads as on, and switching on deletes a row that narrows
// nothing else. Hosted, the platform reads a workspace with no row as off, so
// the switch reads the stored `enabled` alone and only ever writes it.

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

/** Whether a stored row narrows anything beyond allowing or blocking. */
function narrowsAnything(config: WorkspaceWebSearchConfig): boolean {
  return [
    config.max_results,
    config.purpose_hint,
    config.allowed_domains,
    config.blocked_domains,
    config.provider_options,
  ].some((value) => value != null)
}

/**
 * Whether requests billed to this workspace may use the web tools, and how far
 * they may reach.
 *
 * Blocking covers `otari_web_search`, `otari_web_fetch`, and `POST /api/v1/search`.
 * Nothing here grants a backend the deployment has not configured, and nothing
 * here holds a credential.
 */
export function WorkspaceWebSearchCard({
  leading,
  isHosted = false,
  isAvailable = true,
}: {
  /** Rows above the switch, so the tools and their switch read as one card. */
  leading?: ReactNode
  /** A hosted control plane, where no row means off. */
  isHosted?: boolean
  /** Whether this deployment can run either tool at all. */
  isAvailable?: boolean
}) {
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
  const save = useAutosave()
  // One writer for the card: a PUT replaces the whole row, so two rows saving
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
      <SettingsGroup isBounded>
        {leading}
        <SettingRow
          label="Workspace access"
          help={
            workspaceLoading
              ? "Reading the workspaces you belong to."
              : "Per-workspace web access is set on a workspace you belong to. An owner or admin can add you to one on the Workspaces page."
          }
          control={null}
        />
      </SettingsGroup>
    )
  }

  if (!manages) {
    // The playground's answer reads a missing row as on, which a hosted
    // control plane does not, so there a member is shown the tools alone.
    if (isHosted) {
      return leading ? <SettingsGroup isBounded>{leading}</SettingsGroup> : null
    }
    return (
      <WorkspaceToolStatus
        tool="web_search"
        leading={leading}
        workspace={selected}
      />
    )
  }

  const config = query.data
  const isOn = isHosted
    ? Boolean(config?.configured && config.enabled)
    : !(config?.configured && !config.enabled)
  const organizationName = context.data?.organization.name
  // Disabled until the read has succeeded. Without that the switch sits on over
  // a workspace that may well have a stored row, and one change issues the
  // write that drops it.
  const isUnreadable = query.isLoading || query.isError || !config
  const narrowingDisabled = isUnreadable || !isAvailable || !isOn

  const setAllowed = (allowed: boolean) =>
    save.run(() =>
      isHosted || !allowed || (config && narrowsAnything(config))
        ? write({ enabled: allowed })
        : clearConfig.mutateAsync({ workspaceId: selected.workspace_id }),
    )
  // Reachable only while the switch is on, so the row it writes stays on.
  const commitField = (patch: Partial<UpdateWorkspaceWebSearchConfigRequest>) =>
    write({ enabled: true, ...patch })

  return (
    <SettingsGroup isBounded>
      {leading}
      {query.error ? (
        <div className="px-4 py-3">
          <ErrorBanner error={query.error} />
        </div>
      ) : null}

      <SettingRow
        label={`Allow in ${selected.name}`}
        help={
          organizationName
            ? `Workspace in ${organizationName}. Also covers the search API.`
            : "Also covers the search API."
        }
        error={save.error}
        note={
          isAvailable ? null : (
            <p className="text-caption text-subtle">
              Neither tool can run on this deployment, so web access is off
              everywhere.
            </p>
          )
        }
        control={
          <Toggle
            label="Allow web access"
            isSelected={isAvailable && isOn}
            onChange={(next) => void setAllowed(next)}
            isDisabled={isUnreadable || !isAvailable || save.isSaving}
          />
        }
      />

      <AdvancedRows>
        <PolicyRow
          key={`results-${selected.workspace_id}`}
          label="Max results for this workspace"
          help="Search only. Lowers how many results one search returns."
          placeholder="10"
          isNumeric
          committed={
            config?.max_results == null ? "" : String(config.max_results)
          }
          parse={ceilingParser(MAX_RESULTS, "results")}
          commit={(max_results) => commitField({ max_results })}
          disabled={narrowingDisabled}
        />
        <PolicyRow
          key={`allowed-${selected.workspace_id}`}
          label="Allowed domains"
          help="Only these sites and their subdomains, for Search results and Fetch."
          placeholder="mozilla.org, wikipedia.org"
          isMachineReadable
          committed={(config?.allowed_domains ?? []).join(", ")}
          parse={parseDomains}
          commit={(allowed_domains) => commitField({ allowed_domains })}
          disabled={narrowingDisabled}
        />
        <PolicyRow
          key={`blocked-${selected.workspace_id}`}
          label="Blocked domains"
          help="Never these sites, for Search results or Fetch."
          placeholder="reddit.com, pinterest.com"
          isMachineReadable
          committed={(config?.blocked_domains ?? []).join(", ")}
          parse={parseDomains}
          commit={(blocked_domains) => commitField({ blocked_domains })}
          disabled={narrowingDisabled}
        />
      </AdvancedRows>
    </SettingsGroup>
  )
}
