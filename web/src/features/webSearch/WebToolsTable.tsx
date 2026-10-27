import { useState } from "react"
import {
  FiActivity,
  FiEdit2,
  FiStar,
  FiTrash2,
  FiXCircle,
} from "react-icons/fi"

import type { SearchProviderInfo } from "@/client"
import { Button } from "@/design-system/actions/Button"
import { RowAction, RowActionRow } from "@/design-system/actions/RowAction"
import { DataTable, type DataTableColumn } from "@/design-system/data/DataTable"
import { ErrorBanner } from "@/design-system/feedback/ErrorBanner"
import { errorMessage } from "@/design-system/feedback/errorMessage"
import { Badge } from "@/design-system/indicators/Badge"
import { Dot } from "@/design-system/indicators/Dot"
import { SettingsGroup } from "@/design-system/layout/SettingsGroup"
import { toPerCall } from "@/features/toolPricing/toolRates"
import { useTestSearchTool } from "@/shared/api/tools"
import { formatUnitRate } from "@/shared/helpers/format"

import {
  BUILTIN_FETCH,
  describeOptions,
  servesTheModel,
  TEST_INPUT,
  type ToolPrice,
  testOutcome,
  type WebToolKind,
  type WebToolRow,
} from "./webTools"

// One kind's tools as a table, each beside its price, and the kind's fallback
// rate as the last row, since it prices what no tool in the table prices.
// Clicking a row opens its details in a band under it; the actions stay on the
// row.

/** A tool's row, or the fallback rate's. */
export type WebToolTableRow =
  | { type: "tool"; tool: WebToolRow; price: ToolPrice }
  | { type: "fallback"; price: ToolPrice }

const FALLBACK: Readonly<Record<WebToolKind, { label: string; role: string }>> =
  {
    search: {
      label: "Web search",
      role: "A model's search on a tool with no price of its own, and a search on an organization's own key.",
    },
    fetch: {
      label: "Web fetch",
      role: "A fetch on a tool with no price of its own, and a fetch on an organization's own key.",
    },
  }

const COPY: Readonly<
  Record<
    WebToolKind,
    { title: string; add: string; description: string; empty: string }
  >
> = {
  search: {
    title: "Search tools",
    add: "Add search tool",
    description:
      "What the model's otari_web_search and POST /api/v1/search/{tool} run on, for every workspace.",
    empty: "No search tool yet.",
  },
  fetch: {
    title: "Fetch tools",
    add: "Add fetch tool",
    description:
      "What the model's otari_web_fetch, and search hits with no text, are read with.",
    empty: "No fetch tool yet.",
  },
}

/** The fallback row's key: a tool's name never holds a "/", so no tool can take it. */
const FALLBACK_ROW = "/fallback"

function rowKey(row: WebToolTableRow): string {
  return row.type === "tool" ? row.tool.name : FALLBACK_ROW
}

/** A problem worth a mark: a square dot and a sentence. */
function StatusMark({ children }: { children: string }) {
  return (
    <span className="flex items-center gap-1.5 text-caption">
      <Dot className="bg-warning" />
      {children}
    </span>
  )
}

/** The marks a tool's row carries, shortest first. */
function toolMarks(
  tool: WebToolRow,
  provider: SearchProviderInfo | undefined,
): string[] {
  const marks: string[] = []
  if (!tool.isKeyReadable) marks.push("Key unreadable: check OTARI_SECRET_KEY")
  if (tool.kind === "search" && !servesTheModel(provider)) {
    marks.push("Search API only: the model cannot search with it yet")
  }
  if (tool.kind === "fetch" && provider?.reports_final_url === false) {
    marks.push("Redirects unchecked against a workspace's domain lists")
  }
  return marks
}

export function PriceCell({ price }: { price: ToolPrice }) {
  switch (price.state) {
    case "pending":
      return <span className="text-muted">…</span>
    case "error":
      return <Badge tone="warn">Could not read</Badge>
    case "unpriced":
      return <Badge tone="warn">Unpriced</Badge>
    case "own":
      return (
        <span className="flex flex-col gap-0.5">
          <span className="tabular-nums">
            {formatUnitRate(toPerCall(price.perMillion))}
          </span>
          {price.fromConfig ? (
            <span className="text-caption">From config.yml</span>
          ) : null}
        </span>
      )
    case "fallback":
      return (
        <span className="flex flex-col gap-0.5">
          <span className="tabular-nums">
            {formatUnitRate(toPerCall(price.perMillion))}
          </span>
          <span className="text-caption">
            From {price.fallbackKey}
            {price.direct === "unpriced" ? ", for the model's searches" : ""}
          </span>
          {price.direct === "unpriced" ? (
            <Badge tone="warn">Unpriced on the search API</Badge>
          ) : null}
        </span>
      )
  }
}

/** What the row's star does: make the tool the default, stop using it, or nothing. */
type DefaultAction =
  | { type: "make"; refusal: string; onPress: () => void }
  | { type: "stop"; onPress: () => void }
  | { type: "none" }

/** The row's actions, and its last connection test. */
function ToolActions({
  tool,
  price,
  defaultAction,
  isSettingDefault,
  onEdit,
  onEditPrice,
  onRemove,
}: {
  tool: WebToolRow
  price: ToolPrice
  defaultAction: DefaultAction
  isSettingDefault: boolean
  onEdit: () => void
  onEditPrice: () => void
  onRemove: () => void
}) {
  const test = useTestSearchTool()
  const isStored = tool.source === "stored"
  const isPriceFromConfig = price.state === "own" && price.fromConfig
  return (
    <div className="flex flex-col items-end gap-1.5">
      <RowActionRow>
        {defaultAction.type === "make" ? (
          <RowAction
            icon={FiStar}
            label="Make default"
            ariaLabel={
              defaultAction.refusal
                ? `Make ${tool.name} the default. Unavailable: ${defaultAction.refusal}`
                : `Make ${tool.name} the default`
            }
            isDisabled={isSettingDefault || defaultAction.refusal !== ""}
            onPress={defaultAction.onPress}
          />
        ) : defaultAction.type === "stop" ? (
          <RowAction
            icon={FiXCircle}
            label="Stop using as default"
            ariaLabel={
              tool.kind === "search"
                ? `Stop using ${tool.name} as the default; the default goes back to the configuration file's, else automatic`
                : `Stop using ${tool.name} as the default; ${BUILTIN_FETCH} takes over`
            }
            isDisabled={isSettingDefault}
            onPress={defaultAction.onPress}
          />
        ) : null}
        <RowAction
          icon={FiActivity}
          label="Test"
          ariaLabel={`Test ${tool.name}`}
          // A key no secret reads cannot be tested; Edit replaces it.
          isDisabled={test.isPending || !tool.isKeyReadable}
          onPress={() =>
            test.mutate({
              name: tool.name,
              body: TEST_INPUT[tool.kind],
            })
          }
        />
        {isStored ? (
          <>
            <RowAction
              icon={FiEdit2}
              label="Edit"
              ariaLabel={`Edit ${tool.name}`}
              onPress={onEdit}
            />
            <RowAction
              icon={FiTrash2}
              label="Remove"
              ariaLabel={`Remove ${tool.name}`}
              onPress={onRemove}
            />
          </>
        ) : isPriceFromConfig ? null : (
          <RowAction
            icon={FiEdit2}
            label="Edit price"
            ariaLabel={`Edit the price of ${tool.name}`}
            isDisabled={price.state === "pending" || price.state === "error"}
            onPress={onEditPrice}
          />
        )}
      </RowActionRow>
      {isPriceFromConfig && !isStored ? (
        <span className="text-caption">Price in config.yml</span>
      ) : null}
      {test.isPending ? (
        <span className="text-caption">Testing…</span>
      ) : test.data ? (
        <span role="status" className="text-caption">
          {testOutcome(tool.kind, test.data)}
        </span>
      ) : test.error ? (
        <span role="status" className="text-caption text-danger">
          {errorMessage(test.error)}
        </span>
      ) : null}
    </div>
  )
}

/** A tool's details, in the band under its row. */
function ToolDetail({
  tool,
  provider,
  fetchDefault,
}: {
  tool: WebToolRow
  provider: SearchProviderInfo | undefined
  fetchDefault: string
}) {
  const marks = toolMarks(tool, provider)
  const facts: { label: string; value: string }[] = [
    {
      label: "Endpoint",
      value:
        tool.apiBase ||
        (tool.source === "builtin" ? "None" : "The provider's own"),
    },
    ...(tool.kind === "search"
      ? [
          {
            label: "Enriched by",
            value: tool.fetchTool || `The fetch tool (${fetchDefault})`,
          },
        ]
      : []),
    ...(tool.source === "stored"
      ? [
          {
            label: "Timeout",
            value:
              tool.timeout === undefined
                ? "Gateway default"
                : `${tool.timeout} s`,
          },
          { label: "Options", value: describeOptions(tool.options) },
        ]
      : [
          {
            label: "Defined in",
            value:
              tool.source === "builtin"
                ? "Otari, always present"
                : "config.yml",
          },
        ]),
  ]
  return (
    <div className="flex flex-col gap-3 px-4 py-3">
      <dl className="grid gap-3 sm:grid-cols-3">
        {facts.map((fact) => (
          <div key={fact.label} className="flex min-w-0 flex-col gap-1">
            <dt className="text-overline">{fact.label}</dt>
            <dd className="break-words text-caption">{fact.value}</dd>
          </div>
        ))}
      </dl>
      {marks.length > 0 ? (
        <div className="flex flex-col gap-1">
          {marks.map((mark) => (
            <StatusMark key={mark}>{mark}</StatusMark>
          ))}
        </div>
      ) : null}
    </div>
  )
}

export function WebToolsTable({
  kind,
  rows,
  providers,
  defaultName,
  fetchDefault,
  isLoading,
  error,
  onAdd,
  onEdit,
  onEditPrice,
  onEditFallbackPrice,
  onRemove,
  onSetDefault,
  isSettingDefault,
}: {
  kind: WebToolKind
  rows: readonly WebToolTableRow[]
  /** This kind's catalog, for the marks a provider earns its tools. */
  providers: readonly SearchProviderInfo[]
  /** The tool a default names, or `""`. */
  defaultName: string
  /** The fetch tool a search tool that names none is enriched by. */
  fetchDefault: string
  isLoading: boolean
  error: unknown
  onAdd: () => void
  onEdit: (tool: WebToolRow) => void
  onEditPrice: (tool: WebToolRow) => void
  onEditFallbackPrice: () => void
  onRemove: (tool: WebToolRow) => void
  /** Names the default tool, or clears the setting with `null`. */
  onSetDefault: (name: string | null) => void
  isSettingDefault: boolean
}) {
  const [openKey, setOpenKey] = useState<string | null>(null)
  const copy = COPY[kind]
  const providerOf = (tool: WebToolRow) =>
    providers.find((entry) => entry.id === tool.provider)
  const toolCount = rows.filter((row) => row.type === "tool").length

  // Fetch names builtin_fetch outright, since a cleared setting falls back to
  // the configuration file's, which may name another tool; builtin_fetch is
  // the one fetch default nothing replaces, so it is never stopped. Search has
  // no value meaning "automatic", so stopping clears the setting.
  const defaultActionOf = (tool: WebToolRow): DefaultAction => {
    if (kind === "fetch" && tool.name === BUILTIN_FETCH) {
      return defaultName === BUILTIN_FETCH
        ? { type: "none" }
        : {
            type: "make",
            refusal: "",
            onPress: () => onSetDefault(BUILTIN_FETCH),
          }
    }
    if (tool.name === defaultName) {
      return {
        type: "stop",
        onPress: () => onSetDefault(kind === "fetch" ? BUILTIN_FETCH : null),
      }
    }
    const refusal =
      kind === "search" && !servesTheModel(providerOf(tool))
        ? `the model cannot search with ${tool.provider} yet`
        : ""
    return { type: "make", refusal, onPress: () => onSetDefault(tool.name) }
  }

  const columns: DataTableColumn<WebToolTableRow>[] = [
    {
      id: "tool",
      header: "Tool",
      isRowHeader: true,
      cell: (row) => {
        if (row.type === "fallback") {
          return (
            <span className="flex flex-col gap-0.5">
              <span className="flex flex-wrap items-center gap-2">
                <span className="font-medium text-foreground">
                  {FALLBACK[kind].label}
                </span>
                <Badge tone="muted">Fallback</Badge>
              </span>
              <span className="text-caption">{FALLBACK[kind].role}</span>
            </span>
          )
        }
        const { tool } = row
        const marks = toolMarks(tool, providerOf(tool))
        return (
          <span className="flex min-w-0 flex-col gap-0.5">
            <span className="flex flex-wrap items-center gap-2">
              <span className="font-medium text-foreground">{tool.name}</span>
              {tool.name === defaultName ? (
                <Badge tone="muted">Default</Badge>
              ) : null}
              {tool.source === "builtin" ? (
                <Badge tone="muted">Built in</Badge>
              ) : null}
              {tool.source === "config" ? (
                <Badge tone="muted">config.yml</Badge>
              ) : null}
              {tool.overridesConfig ? (
                <Badge tone="muted">Overrides config.yml</Badge>
              ) : null}
            </span>
            <span className="break-words text-caption">
              {tool.source === "builtin"
                ? "Otari's own fetcher"
                : tool.apiBase
                  ? `${tool.provider} · ${tool.apiBase}`
                  : tool.provider}
            </span>
            {marks.length > 0 ? <StatusMark>{marks[0]}</StatusMark> : null}
          </span>
        )
      },
    },
    {
      id: "key",
      header: "API key",
      cell: (row) =>
        row.type === "fallback" ? null : row.tool.last4 ? (
          <code className="text-muted">••••{row.tool.last4}</code>
        ) : (
          <span className="text-muted">
            {row.tool.source === "builtin"
              ? "Not needed"
              : row.tool.hasKey
                ? "In config.yml"
                : "None"}
          </span>
        ),
    },
    {
      id: "price",
      header: "Price per call",
      cell: (row) => <PriceCell price={row.price} />,
    },
    {
      id: "actions",
      header: "Actions",
      align: "end",
      cell: (row) =>
        row.type === "fallback" ? (
          <RowActionRow>
            <RowAction
              icon={FiEdit2}
              label="Edit price"
              ariaLabel={`Edit the price of ${FALLBACK[kind].label}`}
              isDisabled={
                row.price.state === "pending" || row.price.state === "error"
              }
              onPress={onEditFallbackPrice}
            />
          </RowActionRow>
        ) : (
          <ToolActions
            tool={row.tool}
            price={row.price}
            defaultAction={defaultActionOf(row.tool)}
            isSettingDefault={isSettingDefault}
            onEdit={() => onEdit(row.tool)}
            onEditPrice={() => onEditPrice(row.tool)}
            onRemove={() => onRemove(row.tool)}
          />
        ),
    },
  ]

  return (
    <SettingsGroup
      title={copy.title}
      count={isLoading ? undefined : toolCount}
      description={copy.description}
      action={
        <Button
          variant="primary"
          className="min-h-11 md:min-h-9"
          onPress={onAdd}
        >
          {copy.add}
        </Button>
      }
    >
      {error ? (
        <div className="py-3">
          <ErrorBanner error={error} />
        </div>
      ) : null}
      <DataTable
        ariaLabel={copy.title}
        columns={columns}
        rows={[...rows]}
        getRowKey={rowKey}
        isLoading={isLoading}
        emptyContent={copy.empty}
        mobileLayout="cards"
        onRowAction={(key) =>
          key === FALLBACK_ROW
            ? undefined
            : setOpenKey((current) => (current === key ? null : key))
        }
        detailKey={openKey}
        renderDetail={(row) =>
          row.type === "tool" ? (
            <ToolDetail
              tool={row.tool}
              provider={providerOf(row.tool)}
              fetchDefault={fetchDefault}
            />
          ) : null
        }
      />
    </SettingsGroup>
  )
}
