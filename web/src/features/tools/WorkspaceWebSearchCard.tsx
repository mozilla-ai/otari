import { Button } from "@heroui/react"
import { useEffect, useState } from "react"

import { canManageWorkspace } from "@/features/organization/roles"
import {
  useClearWorkspaceWebSearchConfig,
  useOrganizationContext,
  useSetWorkspaceWebSearchConfig,
  useWorkspaceWebSearchConfig,
} from "@/shared/api/hooks"
import { Field } from "@/shared/components/Field"
import { Dot, Section } from "@/shared/components/surface"
import {
  ErrorBanner,
  errorMessage,
  FilterSelect,
  InfoBanner,
} from "@/shared/components/ui"
import { useSelectedWorkspace } from "@/shared/hooks/SelectedWorkspace"

// The layer above the deployment-wide web-search settings this card sits under:
// the settings above say which backend runs a search, this says which
// workspaces may ask for one and how far it may reach. A row can only narrow,
// so there is no control here that turns anything on the deployment has not
// configured; when it has configured nothing, the banner says so rather than
// letting the form imply otherwise.
//
// Three states, not two, which is why the first control is a select rather than
// a toggle: a workspace can be allowed, blocked, or carry no row at all.
// "Deployment default" is the last of those and is a delete, not a saved
// `enabled: true`. Same shape as `WorkspaceCodeExecutionPolicyCard`, which is
// the sibling plane.

type Stance = "default" | "allowed" | "blocked"

// The server's own bounds (`workspace_web_search_service`): a ceiling above
// what the backend honors could never take effect, and the list bound stops one
// workspace's row growing without limit.
export const MAX_RESULTS = 20
export const MAX_DOMAINS = 100

function parseCeiling(
  raw: string,
  max: number,
): { value: number | null; valid: boolean } {
  const trimmed = raw.trim()
  if (trimmed === "") return { value: null, valid: true }
  // Digits only, so `0x10` and `1e1` are refused rather than silently read as
  // 16 and 10 by `Number`.
  if (!/^\d+$/.test(trimmed)) return { value: null, valid: false }
  const parsed = Number(trimmed)
  if (!Number.isSafeInteger(parsed) || parsed <= 0 || parsed > max) {
    return { value: null, valid: false }
  }
  return { value: parsed, valid: true }
}

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
function parseDomains(raw: string): {
  value: string[] | null
  invalid: string | null
  tooMany: boolean
} {
  const hosts = raw
    .split(",")
    .map((host) => host.trim().toLowerCase().replace(/^\.+/, ""))
    .filter((host) => host !== "")
  const invalid = hosts.find((host) => NOT_IN_A_HOSTNAME.test(host)) ?? null
  return {
    value: hosts.length > 0 ? hosts : null,
    invalid,
    tooMany: hosts.length > MAX_DOMAINS,
  }
}

export function WorkspaceWebSearchCard({
  onSaved,
}: {
  onSaved: (message: string) => void
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

  const [stance, setStance] = useState<Stance>("default")
  const [maxResults, setMaxResults] = useState("")
  const [hint, setHint] = useState("")
  const [allowedDomains, setAllowedDomains] = useState("")
  const [blockedDomains, setBlockedDomains] = useState("")
  const [error, setError] = useState("")

  const config = query.data
  // Hydrate from whatever the server last said, including after a save or a
  // clear, so the form never drifts from the stored row.
  useEffect(() => {
    if (!config) return
    setStance(
      !config.configured ? "default" : config.enabled ? "allowed" : "blocked",
    )
    setMaxResults(config.max_results !== null ? String(config.max_results) : "")
    setHint(config.purpose_hint ?? "")
    setAllowedDomains((config.allowed_domains ?? []).join(", "))
    setBlockedDomains((config.blocked_domains ?? []).join(", "))
  }, [config])

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

  // Disabled while the row is in flight, so a save cannot race the load that
  // would overwrite the form under it, and disabled outright until the read has
  // succeeded. Without that last part a failed GET leaves `isLoading` false and
  // `config` undefined, so the form sits at its initial "Deployment default"
  // stance over a workspace that may well have a stored row, and one click on
  // an apparently harmless Save issues the DELETE that drops it.
  const busy =
    setConfig.isPending ||
    clearConfig.isPending ||
    query.isLoading ||
    query.isError ||
    !config

  const save = () => {
    setError("")
    if (stance === "default") {
      clearConfig.mutate(
        { workspaceId: selected.workspace_id },
        {
          onSuccess: () =>
            onSaved(`${selected.name} uses the deployment default`),
          onError: (err) => setError(errorMessage(err)),
        },
      )
      return
    }
    const results = parseCeiling(maxResults, MAX_RESULTS)
    if (!results.valid) {
      setError(`Max results must be a whole number from 1 to ${MAX_RESULTS}.`)
      return
    }
    const allowed = parseDomains(allowedDomains)
    const blocked = parseDomains(blockedDomains)
    if (allowed.tooMany || blocked.tooMany) {
      setError(`A domain list may name at most ${MAX_DOMAINS} domains.`)
      return
    }
    const malformed = allowed.invalid ?? blocked.invalid
    if (malformed !== null) {
      setError(
        `"${malformed}" is not a bare hostname. Give a domain such as example.com, with no scheme, port or path.`,
      )
      return
    }
    setConfig.mutate(
      {
        workspaceId: selected.workspace_id,
        body: {
          enabled: stance === "allowed",
          max_results: results.value,
          purpose_hint: hint.trim() === "" ? null : hint.trim(),
          allowed_domains: allowed.value,
          blocked_domains: blocked.value,
          // Not editable here. The opaque provider bag is a per-backend knob
          // with no form that could validate it, so the card preserves whatever
          // the API holds rather than clearing it on every save: this is a PUT,
          // and omitting it would silently drop a value set over the API.
          provider_options: config?.provider_options ?? null,
        },
      },
      {
        onSuccess: () =>
          onSaved(`Web search settings saved for ${selected.name}`),
        onError: (err) => setError(errorMessage(err)),
      },
    )
  }

  return (
    <Section
      aria-label={`This workspace (${selected.name})`}
      className="border-y border-border py-5"
      contentClassName="flex flex-col gap-4"
    >
      <div className="flex flex-col gap-2">
        <h2 className="text-title">This workspace ({selected.name})</h2>
        <p className="max-w-prose text-sm text-muted">
          Whether requests billed to this workspace may search the web, and how
          far a search may reach. Blocking covers both doors: the
          otari_web_search tool and POST /v1/search. The rest narrows the
          in-loop tool only. These settings can only narrow what the deployment
          above allows; they never grant a backend the deployment has not
          configured, and they hold no credential.
        </p>
      </div>
      <ErrorBanner error={query.error} />
      {/* A ceiling, not a caution: nothing is broken and nothing on this
              page can change it, so it reads on the subtle dot. The danger
              dot is for the things worth acting on. */}
      {config && !config.web_search_configured ? (
        <InfoBanner>
          This deployment has no in-loop search backend configured, so
          otari_web_search is unavailable here whatever this workspace allows.
          The search URL is set above. Blocking still takes effect on POST
          /v1/search, which runs off the search tools below.
        </InfoBanner>
      ) : null}

      <div className="flex flex-wrap items-center gap-3">
        <FilterSelect
          label="Web search"
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
          disabled={busy}
        />
        {config?.configured === false ? (
          // The absence of a stored row, stated rather than boxed: this
          // workspace has not departed from the deployment default.
          <span className="flex items-center gap-2 text-mono-caption text-subtle">
            <Dot className="bg-text-subtle" />
            NOTHING SET
          </span>
        ) : null}
      </div>

      <Field
        label="Max results"
        value={maxResults}
        onChange={setMaxResults}
        placeholder={`Blank for the request's own limit (max ${MAX_RESULTS})`}
        description="Lowers how many results one search returns. It never raises the number."
      />
      <Field
        label="Prompt hint"
        value={hint}
        onChange={setHint}
        placeholder="Leave blank to use the deployment's hint"
        description="Used only when a request declares otari_web_search without a hint of its own."
      />
      <Field
        label="Allowed domains"
        value={allowedDomains}
        onChange={setAllowedDomains}
        placeholder="Comma separated, blank for no restriction"
        description="Results are kept only from these domains. A request that names its own list is narrowed to the domains on both."
      />
      <Field
        label="Blocked domains"
        value={blockedDomains}
        onChange={setBlockedDomains}
        placeholder="Comma separated, blank for none"
        description="Results from these domains are always dropped, whatever a request asks for."
      />

      {error ? (
        <p role="alert" className="text-caption text-danger">
          {error}
        </p>
      ) : null}
      <div className="flex justify-end">
        <Button size="sm" isDisabled={busy} onPress={save}>
          Save
        </Button>
      </div>
    </Section>
  )
}
