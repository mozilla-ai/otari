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
import { CONTROL_LANE, SettingRow } from "@/design-system/layout/SettingRow"
import { SettingsGroup } from "@/design-system/layout/SettingsGroup"
import { AdvancedRows } from "@/features/tools/AdvancedRows"
import { SearchToolsCard } from "@/features/tools/SearchToolsCard"
import type { FieldCopy } from "@/features/tools/ToolSettingRows"
import { ToolPriceRow, ToolSettingRow } from "@/features/tools/ToolSettingRows"
import {
  ToolStatusGroup,
  ToolStatusRow,
} from "@/features/tools/ToolStatusGroup"
import { WorkspaceCodeExecutionPolicyCard } from "@/features/tools/WorkspaceCodeExecutionPolicyCard"
import { WorkspaceMcpServersCard } from "@/features/tools/WorkspaceMcpServersCard"
import { WorkspaceWebSearchCard } from "@/features/tools/WorkspaceWebSearchCard"
import { useDeploymentOperator } from "@/shared/api/organizations"
import { usePricing, useSetPricing } from "@/shared/api/pricing"
import {
  useToolSettings,
  useTools,
  useUpdateToolSettings,
} from "@/shared/api/tools"
import { docsSourceHref } from "@/shared/helpers/docs"
import { useDeployment, useSurfaces } from "@/shared/hooks/useDeployment"

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
    help: "A SearXNG-compatible service. Not needed when web_search_provider names Tavily or Brave; otherwise, while unset, otari_web_search requests are rejected with 400.",
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
    label: "Default max results (all workspaces)",
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
  sandbox_url: {
    label: "Sandbox URL",
    help: "A sandbox you run that speaks the code-execution protocol. While unset, otari_code_execution requests are rejected with 400.",
    placeholder: "http://sandbox:8080",
    isMachineReadable: true,
  },
  sandbox_session_image: {
    label: "Session image",
    help: "The image a sandbox session runs. Blank lets the sandbox choose.",
    placeholder: "mzdotai/otari-sandbox-container:latest",
    isMachineReadable: true,
  },
  code_execution_executor: {
    label: "Who runs provider code tools",
    help: "For requests that declare a provider's own code tool (Anthropic code_execution, OpenAI code_interpreter).",
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

const TENANT_UNAVAILABLE = "Not available on this deployment."

const EXECUTOR_NEEDS_SANDBOX =
  "Takes effect once a sandbox is configured. Until then provider code tools are always forwarded."

// Fields only the `protocol` provider reads; E2B ignores both.
const PROTOCOL_ONLY_KEYS = new Set(["sandbox_url", "sandbox_session_image"])

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
  /** A line for people, in place of the description the model is given. */
  help?: string
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
  /** Keys collapsed under an Advanced row at the end of the group. */
  advanced?: string[]
  /** Hidden from a reader who does not operate the deployment. */
  isOperatorOnly?: boolean
}

interface ServiceSpec {
  key: ToolServiceName
  label: string
  intro: string
  docsAnchor: string
  /** Gateway-run tools whose status and per-call prices belong to this service. */
  managedTools?: ManagedToolSpec[]
  groups: GroupSpec[]
  /** Keys the backend reports that this page deliberately leaves to the API. */
  omit?: string[]
  /**
   * Run by the data plane, so a hosted control plane's own settings for it run
   * nothing: the platform's do.
   */
  isDataPlane?: boolean
}

const SERVICES: ServiceSpec[] = [
  {
    key: "web_search",
    label: "Web search",
    intro: "Live web search and page reading while models work on a request.",
    docsAnchor: "web-search",
    managedTools: [
      {
        toolId: "otari_web_search",
        pricingKey: "otari:web_search",
        urlBacked: true,
        help: "Lets the model search the web for current information.",
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
        help: "Lets the model read a web page by its URL.",
      },
    ],
    groups: [
      {
        title: "Search backend",
        blurb:
          "What runs the model's searches, for every workspace. Fetch needs no backend.",
        docsAnchor: "web-search",
        keys: ["web_search_url"],
        advanced: [
          "web_search_engines",
          "web_search_max_results",
          "web_search_extract",
          "web_search_intercept",
        ],
        isPriced: true,
        catchAll: true,
        isOperatorOnly: true,
      },
    ],
    omit: ["web_search_purpose_hint"],
    isDataPlane: true,
  },
  {
    key: "sandbox",
    label: "Code execution",
    intro:
      "A sandbox where models write and run Python while working on a request.",
    docsAnchor: "code-execution",
    managedTools: [
      {
        toolId: "otari_code_execution",
        pricingKey: "otari:code_execution",
        urlBacked: true,
        unavailableSummary: "Unavailable · no sandbox",
        unavailableHelp:
          "No sandbox is configured, so every call is rejected with 400.",
        help: "Lets the model run Python to do math, analyze data, and make charts or files.",
      },
    ],
    groups: [
      {
        title: "Sandbox",
        blurb: "Where generated code runs, for every workspace.",
        docsAnchor: "code-execution",
        keys: ["sandbox_url"],
        advanced: ["sandbox_session_image", "code_execution_executor"],
        isPriced: true,
        catchAll: true,
        isOperatorOnly: true,
      },
    ],
    omit: ["sandbox_purpose_hint"],
    isDataPlane: true,
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

/** Which sandbox runs the code. Chosen at startup, so it is stated, not edited. */
function SandboxProviderRow({ provider }: { provider: "protocol" | "e2b" }) {
  return (
    <SettingRow
      label="Provider"
      configKey="sandbox_provider"
      help={
        provider === "e2b"
          ? "E2B runs the code. Set with sandbox_provider and E2B_API_KEY at startup."
          : "A sandbox you run. To use E2B instead, set sandbox_provider: e2b and E2B_API_KEY, then restart."
      }
      control={
        <span className="text-caption text-foreground">
          {provider === "e2b" ? "E2B" : "Self-hosted"}
        </span>
      }
    />
  )
}

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
  // below. Of the deployment-wide reads above them, the tool settings answer a
  // tenant without the service endpoints in them (otari-ai#1969), and the tool
  // list is a catalog read any session may make, so both are asked
  // unconditionally. The pricing rows
  // and the /api/v1/search tools stay operator-only on the server, so they are
  // still gated on the same answer the sidebar uses rather than fired into a 403.
  const { isOperator } = useDeploymentOperator()
  const query = useToolSettings()
  const tools = useTools()
  const pricing = usePricing(isOperator)
  const setPricing = useSetPricing()
  const update = useUpdateToolSettings()
  const serves = useSurfaces()
  const isHosted = useDeployment().deployment_type === "hosted"

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
  const sandboxProvider = data?.sandbox_provider ?? undefined
  const byKey = new Map(
    (data?.fields ?? [])
      .filter(
        (field) =>
          sandboxProvider !== "e2b" || !PROTOCOL_ONLY_KEYS.has(field.key),
      )
      .map((field) => [field.key, field]),
  )
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
            requests instead of how to configure a backend they cannot reach.
            A narrowed service's intro says only what its tools do, so it reads
            the same to both. */}
        {narrowed
          ? narrowed.intro
          : isOperator
            ? "Configure the built-in tool and guardrail service endpoints without a restart. Changes apply immediately and persist."
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
        const urlField = [...byKey.values()].find(
          (field) => field.service === service.key && field.type === "url",
        )
        const known = new Set([
          ...service.groups.flatMap((group) => [
            ...group.keys,
            ...(group.advanced ?? []),
          ]),
          ...(service.omit ?? []),
        ])
        // A key the backend reports for this service that no group lists still
        // renders, so a backend addition is visible without a frontend change.
        const unlisted = [...byKey.values()].filter(
          (field) => field.service === service.key && !known.has(field.key),
        )
        const fieldsFor = (keys: string[]) =>
          keys
            .map((key) => byKey.get(key))
            .filter((field): field is ToolSettingField => Boolean(field))
        const renderField = (field: ToolSettingField) => {
          const copy = copyFor(field)
          return (
            <ToolSettingRow
              key={field.key}
              field={field}
              copy={copy}
              defaultLabel={copy.defaultLabel}
              commit={(key, value) => update.mutateAsync(oneField(key, value))}
              disabled={disabled}
              readOnly={!isOperator}
              // The executor only decides anything once there is a sandbox to
              // bring code to; without one every provider declaration is
              // forwarded whatever this says.
              note={
                field.key === "code_execution_executor" &&
                sandboxProvider !== "e2b" &&
                !urlField?.value
                  ? EXECUTOR_NEEDS_SANDBOX
                  : undefined
              }
            />
          )
        }

        const statusRow = ({
          spec,
          tool,
          asRow = false,
        }: (typeof managed)[number] & { asRow?: boolean }) => {
          const Status = asRow ? ToolStatusRow : ToolStatusGroup
          return (
            <Status
              key={tool.id}
              tool={tool}
              // A hosted control plane never runs a tool itself, so its own
              // config says nothing about whether the data plane can.
              showsAvailability={!isHosted}
              docsHref={toolsDocs(spec.docsAnchor ?? service.docsAnchor)}
              help={spec.help}
              urlFieldKey={
                spec.urlBacked && isOperator ? urlField?.key : undefined
              }
              // A tenant cannot set up a backend, so they are told the tool
              // is off here rather than how to turn it on.
              unavailableSummary={
                isOperator ? spec.unavailableSummary : "Unavailable"
              }
              unavailableHelp={
                isOperator ? spec.unavailableHelp : TENANT_UNAVAILABLE
              }
            />
          )
        }

        const leading =
          managed.length > 0
            ? managed.map((entry) => statusRow({ ...entry, asRow: true }))
            : undefined

        return (
          <Fragment key={service.key}>
            {/* The tools and the switch most readers came for, as one card
                above the one-time setup. */}
            {service.key === "sandbox" ? (
              <WorkspaceCodeExecutionPolicyCard
                isHosted={isHosted}
                leading={leading}
              />
            ) : service.key === "web_search" ? (
              <WorkspaceWebSearchCard
                isHosted={isHosted}
                // Unknown until the tool list answers, and a failed read must
                // not lock the switch over tools that may well run.
                isAvailable={
                  isHosted ||
                  managed.length === 0 ||
                  managed.some(({ tool }) => tool.available)
                }
                leading={leading}
              />
            ) : (
              managed.map((entry) => statusRow(entry))
            )}

            {service.groups.map((group) => {
              if (group.isOperatorOnly && !isOperator) return null
              // A hosted control plane serves no inference, so its own tool
              // settings run nothing: the platform's do.
              if (service.isDataPlane && isHosted) return null
              const fields = [
                ...fieldsFor(group.keys),
                ...(group.catchAll ? unlisted : []),
              ]
              const advanced = fieldsFor(group.advanced ?? [])
              // Operator-only inside a group a member also sees: the rate
              // comes from /api/v1/pricing, whose read is still operator-gated, so
              // a member would get an editable "unpriced" row that can only
              // fail on save.
              const pricedTools =
                group.isPriced && isOperator ? (service.managedTools ?? []) : []
              const providerRow =
                service.key === "sandbox" && sandboxProvider ? (
                  <SandboxProviderRow provider={sandboxProvider} />
                ) : null
              if (
                fields.length === 0 &&
                advanced.length === 0 &&
                pricedTools.length === 0 &&
                !providerRow
              )
                return null
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
                  {providerRow}
                  {fields.map(renderField)}
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
                  {advanced.length > 0 ? (
                    <AdvancedRows>{advanced.map(renderField)}</AdvancedRows>
                  ) : null}
                </SettingsGroup>
              )
            })}

            {/* Last, because it configures the direct search API rather than
                the model's tools. Below the backend, because a searxng search
                tool with no URL of its own inherits the one set there.
                Operator-only: its rows are the deployment's own credentials.
                A hosted control plane serves no /api/v1/search at all. */}
            {service.key === "web_search" && isOperator && !isHosted ? (
              <SearchToolsCard docsHref={toolsDocs("direct-search")} />
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
