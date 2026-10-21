import { useState } from "react"
import { FiEdit2 } from "react-icons/fi"

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
import { formatDateTimeMinutes } from "@/shared/helpers/format"

import { ToolRateCell } from "./ToolRateCell"
import { ToolRateForm } from "./ToolRateForm"
import {
  GATEWAY_ROWS,
  hasUnpricedRow,
  isSetInConfig,
  listInstanceRows,
  type PricedRow,
  pairWithRates,
  type ToolPricingService,
  toRowId,
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

const KIND_LABEL: Readonly<Record<"search" | "fetch", string>> = {
  search: "Search instance",
  fetch: "Fetch instance",
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
          ...listInstanceRows("search", searchTools.data),
          ...listInstanceRows("fetch", fetchTools.data),
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

  const entries = pairWithRates(rows, prices)
  const editing = entries.find((entry) => toRowId(entry.row) === editingId)

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
      cell: (entry) => <ToolRateCell entry={entry} />,
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
        isSetInConfig(entry.row, entry.price.data?.origin ?? "") ? (
          <span className="text-caption">Edit in config.yml</span>
        ) : (
          <RowActionRow>
            <RowAction
              icon={FiEdit2}
              label="Edit"
              ariaLabel={`Edit the price of ${entry.row.label}`}
              // An unread rate would be overwritten by a price nobody saw.
              isDisabled={entry.price.isPending || entry.price.isError}
              onPress={() => setEditingId(toRowId(entry.row))}
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
      {hasUnpricedRow(entries) && settings.data?.require_pricing === true ? (
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
        getRowKey={(entry) => toRowId(entry.row)}
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
