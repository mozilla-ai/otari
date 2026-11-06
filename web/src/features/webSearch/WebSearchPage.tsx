import { useState } from "react"

import { EmptyState } from "@/design-system/feedback/EmptyState"
import { ErrorBanner } from "@/design-system/feedback/ErrorBanner"
import { InfoBanner } from "@/design-system/feedback/InfoBanner"
import { PageLoading } from "@/design-system/feedback/PageLoading"
import { PageIntro } from "@/design-system/layout/PageIntro"
import { ToolRateForm } from "@/features/toolPricing/ToolRateForm"
import { type ToolRateRow, toRowId } from "@/features/toolPricing/toolRates"
import { useDeploymentOperator } from "@/shared/api/organizations"
import { usePricesInEffect } from "@/shared/api/pricing"
import { useSettings } from "@/shared/api/settings"
import {
  useSearchProviders,
  useSearchTools,
  useToolSettings,
  useUpdateToolSettings,
} from "@/shared/api/tools"
import { docsSourceHref } from "@/shared/helpers/docs"

import { RemoveWebToolDialog } from "./RemoveWebToolDialog"
import { WebSearchSettings } from "./WebSearchSettings"
import { WebToolDialog } from "./WebToolDialog"
import { WebToolsTable, type WebToolTableRow } from "./WebToolsTable"
import {
  BUILTIN_FETCH,
  isNoSearch,
  type ListedWebTool,
  listWebTools,
  ORGANIZATION_KEY_RATE,
  ORGANIZATION_KEY_ROW,
  rateOnly,
  type ToolPrice,
  type WebToolKind,
} from "./webTools"

// The search and fetch tools this deployment runs for every workspace, what
// each call costs, and which tool each of the model's web tools uses.
//
// Deployment-scoped and operator-only: `/search-tools` and the writes to
// `/pricing` and `/tool-settings` refuse anyone else with a 403, so the rail
// shows the row and the page words the refusal itself, asking nothing first.
// The workspace's own web access stays on Tools > Web search.

type Dialog =
  | { type: "none" }
  | { type: "add"; kind: WebToolKind }
  | { type: "edit"; tool: ListedWebTool }
  | { type: "price"; row: ToolRateRow; current: number | undefined }
  | { type: "remove"; tool: ListedWebTool }

export function WebSearchPage() {
  const { isOperator, isSettled } = useDeploymentOperator()
  const searchList = useSearchTools("search", isOperator)
  const fetchList = useSearchTools("fetch", isOperator)
  const searchCatalog = useSearchProviders("search", isOperator)
  const fetchCatalog = useSearchProviders("fetch", isOperator)
  const toolSettings = useToolSettings(isOperator)
  const settings = useSettings(isOperator)
  const setDefault = useUpdateToolSettings()
  const [dialog, setDialog] = useState<Dialog>({ type: "none" })
  const [openCount, setOpenCount] = useState(0)

  // Gated on the answer as well as the queries: a disabled query still answers
  // whatever another page left in the cache.
  const searchTools = isOperator ? listWebTools("search", searchList.data) : []
  const fetchTools = isOperator ? listWebTools("fetch", fetchList.data) : []
  // A tool's price comes with the listing; the organization keys' rows are
  // read as rows.
  const priceKeys = [
    { kind: "model" as const, modelKey: ORGANIZATION_KEY_RATE.search },
    { kind: "model" as const, modelKey: ORGANIZATION_KEY_RATE.fetch },
  ]
  const reads = usePricesInEffect(priceKeys, isOperator)

  if (!isSettled) return <PageLoading label="Loading web search…" />
  if (!isOperator) {
    return (
      <div className="flex flex-col">
        <PageIntro title="Web search" />
        <EmptyState
          title="Web search is not available to you"
          description="Setting up the search and fetch tools this deployment runs is for its operators. Ask one of them if you need access here."
        />
      </div>
    )
  }

  const readOf = new Map(
    priceKeys.map((key, index) => [toRowId(key), reads[index]]),
  )
  const modelRead = (modelKey: string) =>
    readOf.get(toRowId({ kind: "model", modelKey }))
  const organizationKeyPrice = (kind: WebToolKind): ToolPrice =>
    rateOnly(modelRead(ORGANIZATION_KEY_RATE[kind]))
  const rowsFor = (
    kind: WebToolKind,
    tools: readonly ListedWebTool[],
  ): WebToolTableRow[] => [
    ...tools.map((tool) => ({
      type: "tool" as const,
      tool,
      price: tool.price,
    })),
    { type: "organization" as const, price: organizationKeyPrice(kind) },
  ]
  const searchRows = rowsFor("search", searchTools)
  const fetchRows = rowsFor("fetch", fetchTools)

  const fields = toolSettings.data?.fields ?? []
  const settingOf = (key: string) => {
    const value = fields.find((field) => field.key === key)?.value
    return typeof value === "string" ? value : ""
  }
  const searchDefault = isNoSearch(settingOf("web_search_default_tool"))
    ? ""
    : settingOf("web_search_default_tool")
  const fetchDefault = settingOf("web_fetch_default_tool") || BUILTIN_FETCH
  const takenNames = [...searchTools, ...fetchTools].map((tool) => tool.name)

  const isUnpriced = [...searchRows, ...fetchRows].some(
    (row) => row.price.state === "unpriced",
  )

  const openAdd = (kind: WebToolKind) => {
    setOpenCount((count) => count + 1)
    setDialog({ type: "add", kind })
  }
  const close = () => setDialog({ type: "none" })
  const priceRow = (
    kind: ToolRateRow["kind"],
    modelKey: string,
    label: string,
  ): ToolRateRow => ({ kind, modelKey, label, role: "" })
  const currentOf = (price: ToolPrice) =>
    price.state === "own" ? price.perCall : undefined

  const tableProps = (kind: WebToolKind) => ({
    kind,
    providers:
      (kind === "search" ? searchCatalog.data : fetchCatalog.data) ?? [],
    fetchDefault,
    onAdd: () => openAdd(kind),
    onEdit: (tool: ListedWebTool) => setDialog({ type: "edit", tool }),
    onEditPrice: (tool: ListedWebTool) =>
      setDialog({
        type: "price",
        row: priceRow(kind, tool.name, tool.name),
        current: currentOf(tool.price),
      }),
    onEditOrganizationKeyPrice: () =>
      setDialog({
        type: "price",
        row: priceRow(
          "model",
          ORGANIZATION_KEY_RATE[kind],
          ORGANIZATION_KEY_ROW[kind].label,
        ),
        current: currentOf(organizationKeyPrice(kind)),
      }),
    onRemove: (tool: ListedWebTool) => setDialog({ type: "remove", tool }),
    onSetDefault: (name: string | null) =>
      setDefault.mutate(
        kind === "search"
          ? { web_search_default_tool: name }
          : { web_fetch_default_tool: name },
      ),
    isSettingDefault: setDefault.isPending,
  })

  const dialogKind =
    dialog.type === "add"
      ? dialog.kind
      : dialog.type === "edit"
        ? dialog.tool.kind
        : undefined

  return (
    <div className="flex flex-col gap-4">
      <PageIntro
        title="Web search"
        docsHref={docsSourceHref("tools.md", "web-search")}
      >
        The search and fetch tools this deployment runs for every workspace,
        what each call costs, and which tool the model uses. A call is billed at
        its tool&rsquo;s own rate, and a tool with none is unpriced. Each
        table&rsquo;s last row prices a call on an organization&rsquo;s own key,
        and the search table&rsquo;s a search on the legacy settings too.
      </PageIntro>

      <ErrorBanner
        error={searchCatalog.error ?? fetchCatalog.error ?? setDefault.error}
      />
      {isUnpriced && settings.data?.require_pricing === true ? (
        <InfoBanner tone="warning">
          require_pricing is on, so a call on an unpriced tool is refused with
          402 before it runs. A rate of 0 prices a tool as free.
        </InfoBanner>
      ) : null}

      <WebToolsTable
        {...tableProps("search")}
        rows={searchRows}
        defaultName={searchDefault}
        isLoading={searchList.isLoading}
        error={searchList.error}
      />
      <WebToolsTable
        {...tableProps("fetch")}
        rows={fetchRows}
        defaultName={fetchDefault}
        isLoading={fetchList.isLoading}
        error={fetchList.error}
      />

      <ErrorBanner error={toolSettings.error} />
      <WebSearchSettings fields={fields} disabled={!toolSettings.data} />

      {dialogKind ? (
        <WebToolDialog
          // Every opening starts from blank, or from the stored tool.
          key={dialog.type === "edit" ? dialog.tool.name : openCount}
          kind={dialogKind}
          editing={dialog.type === "edit" ? dialog.tool : undefined}
          providers={
            (dialogKind === "search"
              ? searchCatalog.data
              : fetchCatalog.data) ?? []
          }
          takenNames={takenNames}
          fetchTools={fetchTools}
          fetchProviders={fetchCatalog.data ?? []}
          fetchDefault={fetchDefault}
          currentPrice={
            dialog.type === "edit" ? currentOf(dialog.tool.price) : undefined
          }
          onClose={close}
          onAddAnother={() => openAdd(dialogKind)}
        />
      ) : null}
      {dialog.type === "price" ? (
        <ToolRateForm
          key={`${dialog.row.kind}:${dialog.row.modelKey}`}
          row={dialog.row}
          current={dialog.current}
          onClose={close}
        />
      ) : null}
      {dialog.type === "remove" ? (
        <RemoveWebToolDialog
          key={dialog.tool.name}
          tool={dialog.tool}
          isDefault={
            dialog.tool.name ===
            (dialog.tool.kind === "search" ? searchDefault : fetchDefault)
          }
          onClose={close}
        />
      ) : null}
    </div>
  )
}
