import { Fragment, useCallback, useMemo, useState } from "react"
import { FiArrowRight, FiChevronDown, FiChevronRight } from "react-icons/fi"

import type { AutomaticFallback, FallbackOffering } from "@/client"
import { CopyableValue } from "@/design-system/actions/CopyField"
import { RowAction } from "@/design-system/actions/RowAction"
import { DataTable, type DataTableColumn } from "@/design-system/data/DataTable"
import { ErrorBanner } from "@/design-system/feedback/ErrorBanner"
import { SettingsGroup } from "@/design-system/layout/SettingsGroup"
import { TableScrollFrame } from "@/design-system/layout/TableScrollFrame"
import { docsSourceHref } from "@/shared/helpers/docs"
import { formatNumber, formatRate } from "@/shared/helpers/format"
import { providerDisplayName } from "@/shared/helpers/providers"

/** Shorter than the shared display names, for a chain read across a row. */
const SHORT_NAMES: Record<string, string> = { bedrock: "Bedrock" }

function providerName(provider: string): string {
  return SHORT_NAMES[provider] ?? providerDisplayName(provider)
}

/**
 * One entry per provider, first offering kept: a provider's second offering
 * (Bedrock's US endpoint after its global one) reads as a repeat in a chain
 * about providers.
 */
function byProvider(offerings: FallbackOffering[]): FallbackOffering[] {
  return offerings.filter(
    (offering, index) =>
      offerings.findIndex((other) => other.provider === offering.provider) ===
      index,
  )
}

/** The share of requests that sent the catalog ID, rounded down so 100% means all of them. */
function fallbackShare(row: AutomaticFallback): number {
  if (row.requests === 0) return 0
  return Math.floor(((row.requests - row.pinned) * 100) / row.requests)
}

function Order({ offerings }: { offerings: FallbackOffering[] }) {
  return (
    <span className="flex flex-wrap items-center gap-x-2 gap-y-1 text-sm">
      {byProvider(offerings).map((offering, index) => (
        <Fragment key={offering.selector}>
          {index > 0 ? (
            <FiArrowRight
              aria-label="then"
              className="size-3 shrink-0 text-subtle"
            />
          ) : null}
          <span className="text-foreground">
            {providerName(offering.provider)}
            {offering.input_price == null ? null : (
              <span className="text-muted">
                {" "}
                {formatRate(offering.input_price)}
              </span>
            )}
          </span>
        </Fragment>
      ))}
    </span>
  )
}

function PinnedKeys({ row }: { row: AutomaticFallback }) {
  return (
    <table className="mx-4 my-2 text-sm">
      <tbody>
        {row.pinned_keys.map((entry) => (
          <tr key={`${entry.api_key_id ?? ""}:${entry.provider}`}>
            <td className="py-1 pr-6 text-foreground">
              {entry.key_name ?? entry.api_key_id ?? "No key"}
            </td>
            <td className="py-1 pr-6 text-muted">
              {providerName(entry.provider)}
            </td>
            <td className="py-1 text-right tabular-nums text-muted">
              {formatNumber(entry.requests)}
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  )
}

/**
 * The models used in the last 30 days, the providers their catalog ID tries,
 * and how much of their traffic sent that ID. A row whose traffic named a
 * provider opens on the keys that did.
 */
export function AutomaticFallbacks({
  fallbacks,
  isLoading = false,
  error,
}: {
  fallbacks: AutomaticFallback[]
  isLoading?: boolean
  error?: unknown
}) {
  const [expanded, setExpanded] = useState<string | null>(null)

  // Stable so DataTable's row cache holds; see its docstring.
  const renderDetail = useCallback(
    (row: AutomaticFallback) => <PinnedKeys row={row} />,
    [],
  )

  const columns = useMemo<DataTableColumn<AutomaticFallback>[]>(
    () => [
      {
        id: "model",
        header: "Model",
        isRowHeader: true,
        cell: (row) => (
          <CopyableValue
            value={row.id}
            label="catalog ID"
            className="text-sm text-foreground"
          >
            {row.id}
          </CopyableValue>
        ),
      },
      {
        id: "requests",
        header: "Requests",
        align: "end",
        cell: (row) => (
          <span className="text-sm tabular-nums text-muted">
            {formatNumber(row.requests)}
          </span>
        ),
      },
      {
        id: "fallback",
        header: "Fallback",
        align: "end",
        cell: (row) => {
          const hasBackup = byProvider(row.offerings).length > 1
          const share = fallbackShare(row)
          const isOpen = expanded === row.id
          return (
            <span className="inline-flex items-center gap-2">
              {hasBackup ? (
                <span
                  className={`text-sm tabular-nums ${share < 50 ? "text-warning" : "text-muted"}`}
                >
                  {share}%
                </span>
              ) : (
                <span className="text-sm text-subtle">—</span>
              )}
              {/* A slot on every row, so the shares line up. */}
              <span className="inline-flex w-4">
                {hasBackup && row.pinned_keys.length > 0 ? (
                  <RowAction
                    icon={isOpen ? FiChevronDown : FiChevronRight}
                    label={isOpen ? "Hide keys" : "Keys that named a provider"}
                    onPress={() => setExpanded(isOpen ? null : row.id)}
                  />
                ) : null}
              </span>
            </span>
          )
        },
      },
      {
        id: "order",
        header: "Order",
        cell: (row) => <Order offerings={row.offerings} />,
      },
    ],
    [expanded],
  )

  return (
    <SettingsGroup
      title="Automatic fallback"
      docsHref={docsSourceHref("models.md", "catalog-spellings")}
    >
      {error ? (
        <ErrorBanner error={error} />
      ) : (
        <TableScrollFrame className="otari-fallbacks-table">
          <DataTable
            ariaLabel="Automatic fallback"
            columns={columns}
            rows={fallbacks}
            getRowKey={(row) => row.id}
            detailKey={expanded}
            renderDetail={renderDetail}
            isLoading={isLoading}
            emptyContent="No traffic in the last 30 days."
          />
        </TableScrollFrame>
      )}
    </SettingsGroup>
  )
}
