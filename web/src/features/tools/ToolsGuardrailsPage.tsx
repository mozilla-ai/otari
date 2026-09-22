import { Link } from "@tanstack/react-router"
import { Fragment } from "react"
import type {
  ToolServiceName,
  ToolSettingField,
  UpdateToolSettingsRequest,
} from "@/client"
import { ErrorBanner } from "@/design-system/feedback/ErrorBanner"
import { Skeleton } from "@/design-system/feedback/Skeleton"
import { PageIntro } from "@/design-system/layout/PageIntro"
import { CONTROL_LANE } from "@/design-system/layout/SettingRow"
import { SettingsGroup } from "@/design-system/layout/SettingsGroup"
import { isDeploymentOperator } from "@/features/organization/roles"
import { SearchToolsCard } from "@/features/tools/SearchToolsCard"
import type { FieldCopy } from "@/features/tools/ToolSettingRows"
import { ToolPriceRow, ToolSettingRow } from "@/features/tools/ToolSettingRows"
import { ToolStatusGroup } from "@/features/tools/ToolStatusGroup"
import { WorkspaceCodeExecutionPolicyCard } from "@/features/tools/WorkspaceCodeExecutionPolicyCard"
import { WorkspaceMcpServersCard } from "@/features/tools/WorkspaceMcpServersCard"
import { WorkspaceWebSearchCard } from "@/features/tools/WorkspaceWebSearchCard"
import { useOrganizationContext } from "@/shared/api/organizations"
import { usePricing, useSetPricing } from "@/shared/api/pricing"
import {
  useToolSettings,
  useTools,
  useUpdateToolSettings,
} from "@/shared/api/tools"
import { docsSourceHref } from "@/shared/helpers/docs"
import { useSurfaces } from "@/shared/hooks/useDeployment"

// One settable field maps onto one key of the update request; cast at this one
// boundary (the keys come from the backend's field list).
function oneField(
  key: string,
  value: boolean | number | string | null,
): UpdateToolSettingsRequest {
  return { [key]: value } as UpdateToolSettingsRequest
}

/**
 * What each key is called in the dashboard, and the one line under it.
 *
 * The backend's own `description` is written for an API reference: it names the
 * env var's semantics in full, at two or three lines. A row's help line has to
 * fit under a label, so the page carries its own copy and falls back to the
 * backend's for a key it has not been told about yet.
 */
const FIELD_COPY: Record<string, FieldCopy & { defaultLabel?: string }> = {
  web_search_url: {
    label: "Backend URL",
    help: "While unset, otari_web_search requests are rejected with 400.",
    placeholder: "http://searxng:8080",
    isMachineReadable: true,
  },
  web_search_engines: {
    label: "Engines",
    help: "Comma-separated SearXNG engines. Blank uses the backend's defaults.",
    placeholder: "google,bing,duckduckgo",
    isMachineReadable: true,
  },
  web_search_max_results: {
    label: "Max results",
    help: "Cap on hits per call. A per-tool max_results still overrides it.",
    placeholder: "10",
  },
  web_search_extract: {
    label: "Extract page content",
    help: "On: page text is extracted in-process. Off: snippet-only results.",
    placeholder: "",
    defaultLabel: "Default (on)",
  },
  web_search_intercept: {
    label: "Intercept provider web search",
    help: "Run a bare web_search declaration here instead of at the provider. Needs a backend URL.",
    placeholder: "",
    defaultLabel: "Default (off)",
  },
  web_search_purpose_hint: {
    label: "Purpose hint",
    help: "Sent to the backend when a tool entry has none of its own.",
    placeholder: "Answer from official docs",
  },
  sandbox_provider: {
    label: "Provider",
    help: "What runs the code: protocol talks to the backend URL below, e2b runs it on E2B from this process and needs no backend URL.",
    placeholder: "protocol",
    isMachineReadable: true,
  },
  sandbox_container_idle_ttl_sec: {
    label: "Hold a sandbox for (seconds)",
    help: "How long a sandbox stays resumable after a request that asked to hold one. Requests that do not ask are never held. 0 holds nothing at all.",
    placeholder: "600",
    isMachineReadable: true,
  },
  sandbox_container_max_lifetime_sec: {
    label: "Longest sandbox life (seconds)",
    help: "The most a resumed sandbox may live from its first lease, whatever the idle clock says.",
    placeholder: "3600",
    isMachineReadable: true,
  },
  sandbox_url: {
    label: "Backend URL",
    help: "Where the protocol provider runs code. While unset, otari_code_execution requests are rejected with 400 unless sandbox_provider names a hosted one.",
    placeholder: "http://sandbox:8080",
    isMachineReadable: true,
  },
  sandbox_session_image: {
    label: "Session image",
    help: "The image a leased session runs, for the protocol provider. Blank lets the backend choose.",
    placeholder: "mzdotai/otari-sandbox-container:latest",
    isMachineReadable: true,
  },
  sandbox_purpose_hint: {
    label: "Purpose hint",
    help: "Sent to the backend when a tool entry has none of its own.",
    placeholder: "Run untrusted analysis code",
  },
  code_execution_executor: {
    label: "Who runs provider code tools",
    help: "For a request that declares a provider's own code tool (Anthropic code_execution, OpenAI code_interpreter). Auto keeps it with a provider that runs it natively and brings it here otherwise.",
    placeholder: "",
    defaultLabel: "Default (auto)",
    optionLabels: {
      auto: "Auto: provider when native, else here",
      otari: "Always here, on this sandbox",
      provider: "Always the provider",
    },
  },
  guardrails_url: {
    label: "Backend URL",
    help: "Used when a request does not pass a guardrail URL of its own.",
    placeholder: "http://guardrails:8000",
    isMachineReadable: true,
  },
}

const EXECUTOR_NEEDS_BACKEND =
  "Takes effect once a Backend URL is set above. Until then provider code tools are always forwarded."

function copyFor(field: ToolSettingField): FieldCopy & {
  defaultLabel?: string
} {
  return (
    FIELD_COPY[field.key] ?? {
      label: field.key,
      help: field.description ?? "",
      placeholder: "",
      isMachineReadable: true,
    }
  )
}

interface ManagedToolSpec {
  toolId: string
  pricingKey: string
  /**
   * Whether the service's backend URL is what this tool waits on. A tool gated
   * on anything else must not be sent to that field, which would not turn it on.
   */
  urlBacked?: boolean
  /** The unavailable status, when "no backend" is not the reason. */
  unavailableSummary?: string
  /** What turns the tool on, in the same case. */
  unavailableHelp?: string
  /** A heading in `docs/tools.md`, when the service's own is not the tool's. */
  docsAnchor?: string
}

interface GroupSpec {
  title: string
  blurb: string
  /** A heading in `docs/tools.md`. */
  docsAnchor: string
  keys: string[]
  /** The group the tool's own per-call price belongs in. */
  isPriced?: boolean
  /** Where a key the backend added but this page has not been told about goes. */
  catchAll?: boolean
}

interface ServiceSpec {
  key: ToolServiceName
  label: string
  intro: string
  docsAnchor: string
  /** Gateway-run tools whose status and per-call prices belong to this service. */
  managedTools?: ManagedToolSpec[]
  groups: GroupSpec[]
}

const SERVICES: ServiceSpec[] = [
  {
    key: "web_search",
    label: "Web search",
    intro:
      "Give models live Search and Fetch tools and decide which workspaces may use them. Changes apply immediately.",
    docsAnchor: "web-search",
    managedTools: [
      {
        toolId: "otari_web_search",
        pricingKey: "otari:web_search",
        urlBacked: true,
      },
      {
        toolId: "otari_web_fetch",
        pricingKey: "otari:web_fetch",
        // Fetch has no backend of its own, so the default "no backend" reason
        // would send an operator to a URL field that cannot turn it on. It is
        // off unless the deployment says otherwise, and the switch is
        // startup-only: not in SETTABLE_KEYS, so no screen here can flip it.
        unavailableSummary: "Unavailable · not enabled",
        unavailableHelp:
          "Fetch is off on this gateway. Set OTARI_WEB_FETCH_ENABLED=true (or web_fetch_enabled in config.yml) and restart.",
        docsAnchor: "web-fetch",
      },
    ],
    groups: [
      {
        title: "Backend",
        blurb:
          "Search uses a SearXNG-shaped service at the URL below or a licensed API. Fetch uses the gateway's bounded retrieval service and needs no separate backend.",
        docsAnchor: "web-search",
        keys: [
          "web_search_url",
          "web_search_engines",
          "web_search_max_results",
        ],
        isPriced: true,
      },
      {
        title: "Behavior",
        blurb:
          "How a search runs once the backend answers. Each default applies unless a request says otherwise.",
        docsAnchor: "web-search-interception",
        keys: [
          "web_search_extract",
          "web_search_intercept",
          "web_search_purpose_hint",
        ],
        catchAll: true,
      },
    ],
  },
  {
    key: "sandbox",
    label: "Code execution",
    intro:
      "Give models a sandbox to run generated code in, and decide which workspaces may use it. Changes apply immediately.",
    docsAnchor: "code-execution",
    managedTools: [
      {
        toolId: "otari_code_execution",
        pricingKey: "otari:code_execution",
        urlBacked: true,
      },
    ],
    groups: [
      {
        title: "Backend",
        blurb:
          "The sandbox that runs generated code. Uploaded files a request references are seeded into it, and files the code writes come back through the files API.",
        docsAnchor: "code-execution",
        keys: ["sandbox_url", "sandbox_session_image"],
        isPriced: true,
      },
      {
        title: "Behavior",
        blurb:
          "Who runs a provider's own code tool, and what the gateway sends the sandbox when a request does not.",
        docsAnchor: "code-execution-executor",
        keys: ["code_execution_executor", "sandbox_purpose_hint"],
        catchAll: true,
      },
    ],
  },
  {
    key: "guardrails",
    label: "Guardrails",
    intro:
      "The input-guardrails service this deployment checks requests against.",
    docsAnchor: "who-runs-a-tool",
    groups: [
      {
        title: "Backend",
        blurb:
          "Used when a request does not pass a guardrail URL of its own. Guardrails are a check, so they are never priced.",
        docsAnchor: "who-runs-a-tool",
        keys: ["guardrails_url"],
        catchAll: true,
      },
    ],
  },
]

const toolsDocs = (anchor?: string) => docsSourceHref("tools.md", anchor)

/** The frames, at the real row height, so settings arriving does not move the page. */
function LoadingGroups() {
  return (
    <>
      {[0, 1].map((group) => (
        <SettingsGroup isBounded key={group}>
          {[0, 1, 2].map((row) => (
            <div
              key={row}
              className="flex min-h-11 flex-col gap-2.5 px-4 py-3 md:flex-row md:items-center md:gap-6"
            >
              <div className="flex min-w-0 flex-1 flex-col gap-1.5">
                <Skeleton className="h-4 w-40" />
                <Skeleton className="h-3.5 w-64" />
              </div>
              <Skeleton className={`h-8 ${CONTROL_LANE}`} />
            </div>
          ))}
        </SettingsGroup>
      ))}
    </>
  )
}

/**
 * The tool and guardrail service settings, whole or narrowed to one service.
 *
 * `only` is what makes the sidebar's Tools group work: each child route renders
 * this page filtered to its own service rather than scrolling one long page.
 * Omitted, every service renders, which is the /tools route the group's parent
 * still points at.
 *
 * Every control on the page saves itself: text on blur or Enter, selects on
 * change, each row reporting its own outcome where it happened. There is no
 * Save button and no page-level toast, because a page of independent settings
 * has nothing for one button to be about.
 */
export function ToolsGuardrailsPage({ only }: { only?: ToolServiceName } = {}) {
  // The Tools group is member-visible for the workspace and organization cards
  // below. Of the deployment-wide reads above them, only the tool settings now
  // answer a tenant, without the service endpoints in them (otari-ai#1969), so
  // that one is asked unconditionally and rendered read-only. The pricing rows
  // and the /api/v1/search tools stay operator-only on the server, so they are
  // still gated on the same answer the sidebar uses rather than fired into a 403.
  const organization = useOrganizationContext()
  const isOperator = isDeploymentOperator(organization.data)
  const query = useToolSettings()
  const tools = useTools(isOperator)
  const pricing = usePricing(isOperator)
  const setPricing = useSetPricing()
  const update = useUpdateToolSettings()
  const serves = useSurfaces()

  // Latest rate per key. /api/v1/pricing is history-shaped (one row per
  // effective_at), and the newest row is the one in force.
  const currentRates = (pricing.data ?? []).reduce(
    (rates, row) =>
      rates.has(row.model_key)
        ? rates
        : rates.set(row.model_key, row.input_price_per_million),
    new Map<string, number>(),
  )

  const data = query.data
  const disabled = !data
  const byKey = new Map((data?.fields ?? []).map((field) => [field.key, field]))
  const shown = SERVICES.filter((service) => !only || service.key === only)
  const narrowed = only ? shown[0] : undefined

  return (
    <div className="flex flex-col gap-10 pb-10">
      <PageIntro
        title={narrowed?.label ?? "Tools & Guardrails"}
        docsHref={toolsDocs(narrowed?.docsAnchor)}
      >
        {/* Two readings: an operator configures the service endpoints, and a
            caller who does not is told what the deployment's tools do to their
            requests instead of how to configure a backend they cannot reach. */}
        {isOperator
          ? (narrowed?.intro ??
            "Configure the built-in tool and guardrail service endpoints without a restart. Changes apply immediately and persist.")
          : "How this deployment's built-in tools behave on your requests, what your workspace may use of them, and what your organization mandates."}
      </PageIntro>

      <ErrorBanner error={query.error} />

      {query.isLoading ? <LoadingGroups /> : null}

      {shown.map((service) => {
        const managed = (service.managedTools ?? []).flatMap((spec) => {
          const tool = (tools.data?.data ?? []).find(
            (candidate) => candidate.id === spec.toolId,
          )
          return tool ? [{ spec, tool }] : []
        })
        // Where the "no backend" case sends the operator. Found by type rather
        // than by position, so it survives a group's keys being reordered.
        const urlField = (data?.fields ?? []).find(
          (field) => field.service === service.key && field.type === "url",
        )
        const known = new Set(service.groups.flatMap((group) => group.keys))
        // A key the backend reports for this service that no group lists still
        // renders, so a backend addition is visible without a frontend change.
        const unlisted = (data?.fields ?? []).filter(
          (field) => field.service === service.key && !known.has(field.key),
        )

        return (
          <Fragment key={service.key}>
            {/* The question an operator arrives with, above the settings that
                answer it: can this deployment run the tool at all. */}
            {managed.map(({ spec, tool }) => (
              <ToolStatusGroup
                key={tool.id}
                tool={tool}
                docsHref={toolsDocs(spec.docsAnchor ?? service.docsAnchor)}
                urlFieldKey={spec.urlBacked ? urlField?.key : undefined}
                unavailableSummary={spec.unavailableSummary}
                unavailableHelp={spec.unavailableHelp}
              />
            ))}

            {service.groups.map((group) => {
              const fields = [
                ...group.keys
                  .map((key) => byKey.get(key))
                  .filter((field): field is ToolSettingField => Boolean(field)),
                ...(group.catchAll ? unlisted : []),
              ]
              // Operator-only inside a group a member also sees: the rate
              // comes from /api/v1/pricing, whose read is still operator-gated, so
              // a member would get an editable "unpriced" row that can only
              // fail on save.
              const pricedTools =
                group.isPriced && isOperator ? (service.managedTools ?? []) : []
              if (fields.length === 0 && pricedTools.length === 0) return null
              return (
                <SettingsGroup
                  isBounded
                  key={group.title}
                  // On the combined page the service is not otherwise named,
                  // and three groups called "Backend" say nothing about which
                  // service each one configures.
                  title={
                    only ? group.title : `${service.label} · ${group.title}`
                  }
                  description={group.blurb}
                  docsHref={toolsDocs(group.docsAnchor)}
                >
                  {fields.map((field) => {
                    const copy = copyFor(field)
                    return (
                      <ToolSettingRow
                        key={field.key}
                        field={field}
                        copy={copy}
                        defaultLabel={copy.defaultLabel}
                        commit={(key, value) =>
                          update.mutateAsync(oneField(key, value))
                        }
                        disabled={disabled}
                        readOnly={!isOperator}
                        // The executor only decides anything once there is a
                        // sandbox to bring code to; without one every provider
                        // declaration is forwarded whatever this says.
                        note={
                          field.key === "code_execution_executor" &&
                          !urlField?.value
                            ? EXECUTOR_NEEDS_BACKEND
                            : undefined
                        }
                      />
                    )
                  })}
                  {pricedTools.map(({ pricingKey }) => (
                    <ToolPriceRow
                      key={pricingKey}
                      pricingKey={pricingKey}
                      configured={currentRates.get(pricingKey) ?? null}
                      commit={(perMillion) =>
                        setPricing.mutateAsync({
                          model_key: pricingKey,
                          input_price_per_million: perMillion,
                          // A tool call is one unit; there is no output side
                          // of it to price.
                          output_price_per_million: 0,
                          unit: "requests",
                        })
                      }
                      // Also disabled when the load failed: an errored query
                      // leaves the rate unknown, and a blur would overwrite a
                      // price nobody can see.
                      disabled={pricing.isLoading || Boolean(pricing.error)}
                      loadError={
                        pricing.error
                          ? "Could not read the current price. Reload before editing."
                          : undefined
                      }
                    />
                  ))}
                </SettingsGroup>
              )
            })}

            {/* Directly below the in-loop web-search settings, because a searxng
                search tool that declares no backend URL of its own inherits the
                one set just above it. Operator-only, like those settings: its
                rows are the deployment's own /api/v1/search credentials. The
                workspace group goes below both, because it narrows the backend
                above it and the /api/v1/search tools beside it. */}
            {service.key === "web_search" ? (
              <>
                {isOperator ? (
                  <SearchToolsCard docsHref={toolsDocs("direct-search")} />
                ) : null}
                <WorkspaceWebSearchCard
                  docsHref={toolsDocs("per-workspace-search-policy")}
                />
              </>
            ) : null}
            {service.key === "sandbox" ? (
              <WorkspaceCodeExecutionPolicyCard
                docsHref={toolsDocs("per-workspace-code-policy")}
              />
            ) : null}
            {service.key === "guardrails" &&
            serves("organization_guardrails") ? (
              // The organization's own guardrails moved to a page of their own;
              // this rail keeps the deployment's service and says where they went.
              <p className="text-sm text-muted">
                The guardrails your organization runs on every request, and the
                ones Otari runs itself, are on the organization&rsquo;s{" "}
                <Link
                  to="/organization/guardrails"
                  className="font-medium text-link hover:text-link-hover"
                >
                  Guardrails
                </Link>{" "}
                page.
              </p>
            ) : null}
          </Fragment>
        )
      })}

      {/* Beside the services rather than under one of them, and left out of
          every narrowed view, each of which is one service. `/tools/mcp-servers`
          renders the same card. */}
      {only ? null : <WorkspaceMcpServersCard />}
    </div>
  )
}
