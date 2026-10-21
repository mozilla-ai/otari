import type { UseQueryResult } from "@tanstack/react-query"
import { useState } from "react"
import { FiEdit2 } from "react-icons/fi"

import type { PricingResponse } from "@/client"
import { RowAction, RowActionRow } from "@/design-system/actions/RowAction"
import { DataTable, type DataTableColumn } from "@/design-system/data/DataTable"
import { EmptyState } from "@/design-system/feedback/EmptyState"
import { ErrorBanner } from "@/design-system/feedback/ErrorBanner"
import { InfoBanner } from "@/design-system/feedback/InfoBanner"
import { PageLoading } from "@/design-system/feedback/PageLoading"
import { Badge } from "@/design-system/indicators/Badge"
import { PageIntro } from "@/design-system/layout/PageIntro"
import { useDeploymentOperator } from "@/shared/api/organizations"
import { usePricesInEffect } from "@/shared/api/pricing"
import { useSettings } from "@/shared/api/settings"
import { useSearchTools } from "@/shared/api/tools"
import { docsSourceHref } from "@/shared/helpers/docs"
import { formatDateTimeMinutes, formatUnitRate } from "@/shared/helpers/format"

import { ToolRateForm } from "./ToolRateForm"
import {
  GATEWAY_ROWS,
  instanceRows,
  isSetInConfig,
  perCall,
  rowId,
  type ToolPricingService,
  type ToolRateRow,
} from "./toolRates"

// What the deployment charges for each call of a tool the gateway runs itself.
// Deployment-scoped, as Hosted providers is for models: one price list for
// every organization, which an organization's own override can still narrow.
//
// Operator-only. `/pricing` refuses everyone else with a 403, so the rail shows
// the row and the page words the refusal itself rather than rendering a table
// of errors.

const TITLES: Readonly<Record<ToolPricingService, string>> = {
  web_search: "Web search pricing",
  code_execution: "Code execution pricing",
}

const INTROS: Readonly<Record<ToolPricingService, string>> = {
  web_search:
    "What this deployment charges for each successful search and fetch the gateway runs. A call is billed at its instance's own rate, then at the Web search or Web fetch rate. A rate marked config.yml comes from price_per_call in the configuration and is changed there.",
  code_execution:
    "What this deployment charges for each successful call the gateway runs on its sandbox. A tool a model's provider runs itself is billed by that provider, not here.",
}

/** The `otari:<tool>` row a call on an instance with no rate of its own falls back to. */
const FALLBACK_KEY: Readonly<Record<"search" | "fetch", string>> = {
  search: "otari:web_search",
  fetch: "otari:web_fetch",
}

const KIND_LABEL: Readonly<Record<"search" | "fetch", string>> = {
  search: "Search instance",
  fetch: "Fetch instance",
}

type PriceQuery = UseQueryResult<PricingResponse | null>

interface PricedRow {
  row: ToolRateRow
  price: PriceQuery
  /** The fallback row's price, for an instance with none of its own. */
  fallback: PriceQuery | undefined
}

/** The rate a call on the row is billed at, own or inherited, or `undefined` while unpriced or unknown. */
function billedAt({ price, fallback }: PricedRow): number | undefined {
  return (
    price.data?.input_price_per_million ??
    fallback?.data?.input_price_per_million ??
    undefined
  )
}

function PriceCell({ entry }: { entry: PricedRow }) {
  const { price, fallback } = entry
  if (price.isError) return <Badge tone="warn">Could not read</Badge>
  if (price.isPending) return <span className="text-muted">…</span>
  if (price.data) {
    return (
      <span className="tabular-nums">
        {formatUnitRate(perCall(price.data.input_price_per_million))}
      </span>
    )
  }
  if (fallback?.data) {
    return (
      <span className="flex flex-col gap-0.5">
        <span className="tabular-nums">
          {formatUnitRate(perCall(fallback.data.input_price_per_million))}
        </span>
        <span className="text-caption">From {fallback.data.model_key}</span>
      </span>
    )
  }
  return <Badge tone="warn">Unpriced</Badge>
}

export function ToolPricingPage({ service }: { service: ToolPricingService }) {
  const { isOperator, isSettled } = useDeploymentOperator()
  const listsInstances = isOperator && service === "web_search"
  const searchTools = useSearchTools("search", listsInstances)
  const fetchTools = useSearchTools("fetch", listsInstances)
  const settings = useSettings(isOperator)
  const [editingId, setEditingId] = useState("")

  // Gated on `listsInstances` as well as the queries: a disabled query still
  // answers whatever another page left in the cache.
  const rows = [
    ...GATEWAY_ROWS[service],
    ...(listsInstances
      ? [
          ...instanceRows("search", searchTools.data),
          ...instanceRows("fetch", fetchTools.data),
        ]
      : []),
  ]
  const prices = usePricesInEffect(rows, isOperator)

  if (!isSettled) {
    return <PageLoading label={`Loading ${TITLES[service].toLowerCase()}…`} />
  }
  if (!isOperator) {
    return (
      <div className="flex flex-col">
        <PageIntro title={TITLES[service]} />
        <EmptyState
          title={`${TITLES[service]} is not available to you`}
          description="Setting what this deployment charges for its tools is for its operators. Ask one of them if you need access here."
        />
      </div>
    )
  }

  const priceOf = new Map(rows.map((row, index) => [rowId(row), prices[index]]))
  const entries: PricedRow[] = rows.map((row, index) => ({
    row,
    price: prices[index],
    fallback:
      row.kind === "model"
        ? undefined
        : priceOf.get(
            rowId({ kind: "model", modelKey: FALLBACK_KEY[row.kind] }),
          ),
  }))
  const editing = entries.find((entry) => rowId(entry.row) === editingId)
  const hasUnpriced = entries.some(
    (entry) =>
      !entry.price.isPending &&
      !entry.price.isError &&
      billedAt(entry) === undefined,
  )

  const columns: DataTableColumn<PricedRow>[] = [
    {
      id: "tool",
      header: "Tool",
      isRowHeader: true,
      cell: ({ row }) => (
        <span className="flex flex-col gap-0.5">
          <span className="flex flex-wrap items-center gap-2">
            <span className="font-medium text-foreground">{row.label}</span>
            {row.provider && row.provider !== row.label ? (
              <span className="text-caption">{row.provider}</span>
            ) : null}
            {row.kind === "model" ? null : (
              <Badge tone="muted">{KIND_LABEL[row.kind]}</Badge>
            )}
            {row.isConfigured ? <Badge tone="muted">config.yml</Badge> : null}
          </span>
          <span className="text-caption">{row.role}</span>
        </span>
      ),
    },
    {
      id: "key",
      header: "Priced under",
      cell: ({ row }) => (
        <code className="font-mono text-caption">{row.modelKey}</code>
      ),
    },
    {
      id: "price",
      header: "Price per call",
      cell: (entry) => <PriceCell entry={entry} />,
    },
    {
      id: "since",
      header: "Since",
      cell: ({ price }) => (
        <span className="text-muted">
          {formatDateTimeMinutes(price.data?.effective_at)}
        </span>
      ),
    },
    {
      id: "actions",
      header: "Actions",
      align: "end",
      cell: (entry) =>
        isSetInConfig(entry.row, entry.price.data?.origin) ? (
          <span className="text-caption">Edit in config.yml</span>
        ) : (
          <RowActionRow>
            <RowAction
              icon={FiEdit2}
              label="Edit"
              ariaLabel={`Edit the price of ${entry.row.label}`}
              // An unread rate would be overwritten by a price nobody saw.
              isDisabled={entry.price.isPending || entry.price.isError}
              onPress={() => setEditingId(rowId(entry.row))}
            />
          </RowActionRow>
        ),
    },
  ]

  return (
    <div className="flex flex-col gap-4">
      <PageIntro
        title={TITLES[service]}
        docsHref={docsSourceHref("tools.md", "pricing-a-gateway-run-tool")}
      >
        {INTROS[service]}
      </PageIntro>

      <ErrorBanner error={searchTools.error ?? fetchTools.error} />
      {hasUnpriced && settings.data?.require_pricing === true ? (
        <InfoBanner tone="warning">
          require_pricing is on, so a request that would run an unpriced tool is
          refused with 402 before the model is called. A rate of 0 prices a tool
          as free.
        </InfoBanner>
      ) : null}

      <DataTable
        ariaLabel={TITLES[service]}
        columns={columns}
        rows={entries}
        getRowKey={(entry) => rowId(entry.row)}
        isLoading={searchTools.isLoading || fetchTools.isLoading}
        mobileLayout="cards"
      />

      {editing ? (
        <ToolRateForm
          key={editingId}
          row={editing.row}
          current={editing.price.data?.input_price_per_million}
          onClose={() => setEditingId("")}
        />
      ) : null}
    </div>
  )
}
