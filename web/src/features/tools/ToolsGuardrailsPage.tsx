import { Fragment, useEffect, useRef, useState } from "react"
import { FiCheck } from "react-icons/fi"
import type {
  ToolServiceName,
  ToolSettingField,
  UpdateToolSettingsRequest,
} from "@/client"
import { ErrorBanner } from "@/design-system/feedback/ErrorBanner"
import { Skeleton } from "@/design-system/feedback/Skeleton"
import { PageIntro } from "@/design-system/layout/PageIntro"
import { SettingsGroup } from "@/design-system/layout/SettingsGroup"
import { isDeploymentOperator } from "@/features/organization/roles"
import { OrganizationGuardrailsCard } from "@/features/tools/OrganizationGuardrailsCard"
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
    machine: true,
  },
  web_search_engines: {
    label: "Engines",
    help: "Comma-separated SearXNG engines. Blank uses the backend's defaults.",
    placeholder: "google,bing,duckduckgo",
    machine: true,
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
  sandbox_url: {
    label: "Backend URL",
    help: "While unset, otari_code_execution requests are rejected with 400.",
    placeholder: "http://sandbox:8080",
    machine: true,
  },
  sandbox_session_image: {
    label: "Session image",
    help: "The image a leased session runs. Blank lets the backend choose.",
    placeholder: "mzdotai/otari-sandbox-container:latest",
    machine: true,
  },
  sandbox_purpose_hint: {
    label: "Purpose hint",
    help: "Sent to the backend when a tool entry has none of its own.",
    placeholder: "Run untrusted analysis code",
  },
  guardrails_url: {
    label: "Backend URL",
    help: "Used when a request does not pass a guardrail URL of its own.",
    placeholder: "http://guardrails:8000",
    machine: true,
  },
}

function copyFor(field: ToolSettingField): FieldCopy & {
  defaultLabel?: string
} {
  return (
    FIELD_COPY[field.key] ?? {
      label: field.key,
      help: field.description ?? "",
      placeholder: "",
      machine: true,
    }
  )
}

interface GroupSpec {
  title: string
  blurb: string
  /** A heading in `docs/tools.md`. */
  docsAnchor: string
  keys: string[]
  /** The group the tool's own per-call price belongs in. */
  priced?: boolean
  /** Where a key the backend added but this page has not been told about goes. */
  catchAll?: boolean
}

interface ServiceSpec {
  key: ToolServiceName
  label: string
  intro: string
  docsAnchor: string
  /** The pricing key for a tool Otari runs itself. Guardrails is a check, not billable work. */
  pricingKey?: string
  /** The `/tools` id whose status heads the page. Guardrails declares none. */
  toolId?: string
  groups: GroupSpec[]
}

const SERVICES: ServiceSpec[] = [
  {
    key: "web_search",
    label: "Web search",
    intro:
      "Give models a live search tool and decide which workspaces may use it. Changes apply immediately.",
    docsAnchor: "web-search",
    pricingKey: "otari:web_search",
    toolId: "otari_web_search",
    groups: [
      {
        title: "Backend",
        blurb:
          "A SearXNG-shaped service at the URL below, or a licensed API (web_search_provider), which needs no URL.",
        docsAnchor: "web-search",
        keys: [
          "web_search_url",
          "web_search_engines",
          "web_search_max_results",
        ],
        priced: true,
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
    pricingKey: "otari:code_execution",
    toolId: "otari_code_execution",
    groups: [
      {
        title: "Backend",
        blurb: "The sandbox that runs generated code for otari_code_execution.",
        docsAnchor: "code-execution",
        keys: ["sandbox_url", "sandbox_session_image"],
        priced: true,
      },
      {
        title: "Behavior",
        blurb: "What the gateway sends the sandbox when a request does not.",
        docsAnchor: "code-execution",
        keys: ["sandbox_purpose_hint"],
        catchAll: true,
      },
    ],
  },
  {
    key: "guardrails",
    label: "Guardrails",
    intro:
      "The input-guardrails service this deployment checks requests against, and what your organization mandates.",
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

// Two sibling cards still submit with a Save button of their own and have no
// row to report into, so they keep the page-level acknowledgement. The setting
// rows do not use it: each says what happened where it happened.
function useSaveToast(): [string | null, (message: string) => void] {
  const [message, setMessage] = useState<string | null>(null)
  const timer = useRef<number | undefined>(undefined)
  useEffect(() => () => window.clearTimeout(timer.current), [])
  return [
    message,
    (next: string) => {
      setMessage(next)
      window.clearTimeout(timer.current)
      timer.current = window.setTimeout(() => setMessage(null), 2500)
    },
  ]
}

function SaveToast({ message }: { message: string | null }) {
  if (!message) return null
  return (
    <div
      role="status"
      aria-live="polite"
      className="fixed right-4 bottom-4 z-50 flex items-center gap-2 rounded-lg border border-success bg-success-subtle px-4 py-3 text-sm font-medium text-success shadow-elevation-lg"
    >
      <FiCheck aria-hidden="true" className="h-5 w-5" />
      {message}
    </div>
  )
}

/** The frames, at the real row height, so settings arriving does not move the page. */
function LoadingGroups() {
  return (
    <>
      {[0, 1].map((group) => (
        <SettingsGroup bounded key={group}>
          {[0, 1, 2].map((row) => (
            <div
              key={row}
              className="flex min-h-11 items-center gap-6 px-4 py-3"
            >
              <div className="flex min-w-0 flex-1 flex-col gap-1.5">
                <Skeleton className="h-4 w-40" />
                <Skeleton className="h-3.5 w-64" />
              </div>
              <Skeleton className="h-8 w-[13.75rem] shrink-0" />
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
  const [toast, showToast] = useSaveToast()

  // Latest rate per key. /api/v1/pricing is history-shaped (one row per
  // effective_at), and the newest row is the one in force.
  const currentRates = new Map<string, number>()
  for (const row of pricing.data ?? []) {
    if (!currentRates.has(row.model_key)) {
      currentRates.set(row.model_key, row.input_price_per_million)
    }
  }

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
        const managed = service.toolId
          ? (tools.data?.data ?? []).find((tool) => tool.id === service.toolId)
          : undefined
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
            {managed ? (
              <ToolStatusGroup
                tool={managed}
                docsHref={toolsDocs(service.docsAnchor)}
                urlFieldKey={urlField?.key}
              />
            ) : null}

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
              const pricingKey =
                group.priced && isOperator ? service.pricingKey : undefined
              if (fields.length === 0 && pricingKey === undefined) return null
              return (
                <SettingsGroup
                  bounded
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
                      />
                    )
                  })}
                  {pricingKey ? (
                    <ToolPriceRow
                      pricingKey={pricingKey}
                      configured={currentRates.get(pricingKey) ?? null}
                      commit={(perMillion) =>
                        setPricing.mutateAsync({
                          model_key: pricingKey,
                          input_price_per_million: perMillion,
                          // A tool call is one unit; there is no output side
                          // of it to price.
                          output_price_per_million: 0,
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
                  ) : null}
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
            {service.key === "guardrails" ? (
              <OrganizationGuardrailsCard onSaved={showToast} />
            ) : null}
          </Fragment>
        )
      })}

      {/* Beside the services rather than under one of them, and left out of
          every narrowed view, each of which is one service. `/tools/mcp-servers`
          renders the same card. */}
      {only ? null : <WorkspaceMcpServersCard />}

      <SaveToast message={toast} />
    </div>
  )
}
