import { Button, Card } from "@heroui/react";
import clsx from "clsx";
import { useEffect, useMemo, useRef, useState } from "react";
import type { ReactNode } from "react";

import {
  useDeleteUsage,
  useKeys,
  useSetUsagePrice,
  useUsageCount,
  useUsageLogs,
  useUsageSummary,
  useUsers,
} from "@/api/hooks";
import type { UsageEntry, UsageFilters, UsageMutationSelection } from "@/api/types";
import { BulkActionBar } from "@/components/BulkActionBar";
import { ConfirmDialog } from "@/components/ConfirmDialog";
import { DataTable, type DataTableColumn } from "@/components/DataTable";
import { SetPriceDialog, type ManualRates } from "@/components/SetPriceDialog";
import { TablePagination } from "@/components/TablePagination";
import { ErrorBanner, FilterComboBox, FilterSelect, PageHeader } from "@/components/ui";
import { resolveSelectedIds, useTableSelection } from "@/lib/tableSelection";
import { useUrlState } from "@/lib/urlState";

// ---------- formatting ----------

const usd = new Intl.NumberFormat(undefined, { style: "currency", currency: "USD", maximumFractionDigits: 4 });

function formatUSD(value: number | null): string {
  return value === null ? "—" : usd.format(value);
}

function formatTokens(value: number | null): string {
  return value === null ? "—" : value.toLocaleString();
}

// Humanize a millisecond duration: "820 ms", "1.4 s". Null (historical rows,
// batch jobs) renders as an em-dash so the column reads cleanly.
function formatLatency(ms: number | null): string {
  if (ms === null) return "—";
  if (ms < 1000) return `${ms} ms`;
  return `${(ms / 1000).toFixed(ms < 10_000 ? 2 : 1)} s`;
}

function absolute(iso: string): string {
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? iso : d.toLocaleString();
}

// Relative time reads better in a scan than a full timestamp; the absolute value
// stays available as a tooltip.
function timeAgo(iso: string): string {
  const then = new Date(iso).getTime();
  if (Number.isNaN(then)) return iso;
  const secs = Math.max(0, Math.round((Date.now() - then) / 1000));
  if (secs < 60) return `${secs}s ago`;
  const mins = Math.round(secs / 60);
  if (mins < 60) return `${mins}m ago`;
  const hours = Math.round(mins / 60);
  if (hours < 24) return `${hours}h ago`;
  return `${Math.round(hours / 24)}d ago`;
}

// ---------- filter option sets ----------

const HOUR_S = 3_600;
const DAY_S = 86_400;

const RANGE_PRESETS: { key: string; label: string; seconds: number | null }[] = [
  { key: "1h", label: "1h", seconds: HOUR_S },
  { key: "24h", label: "24h", seconds: DAY_S },
  { key: "7d", label: "7d", seconds: 7 * DAY_S },
  { key: "30d", label: "30d", seconds: 30 * DAY_S },
  { key: "all", label: "All", seconds: null },
];

const DEFAULT_RANGE = "24h";

const STATUS_OPTIONS: { label: string; value: string }[] = [
  { label: "All", value: "" },
  { label: "Success", value: "success" },
  { label: "Error", value: "error" },
];

const PRICED_OPTIONS: { label: string; value: string }[] = [
  { label: "All", value: "" },
  { label: "Priced", value: "true" },
  { label: "Unpriced", value: "false" },
];

const DEFAULT_PAGE_SIZE = 50;

// All filter + pagination state, with defaults, kept in the URL.
const URL_DEFAULTS = {
  range: DEFAULT_RANGE,
  start_date: "",
  end_date: "",
  status: "",
  model: "",
  user_id: "",
  api_key_id: "",
  priced: "",
  page: "0",
  size: String(DEFAULT_PAGE_SIZE),
} as const;

function isoAgo(seconds: number): string {
  return new Date(Date.now() - seconds * 1000).toISOString();
}

// Resolve the query window. Explicit start_date/end_date bounds (a custom range,
// or a drill-down from the Usage page) take precedence; otherwise a preset anchors
// `start` to "now minus N", and "all" (or an empty custom range) leaves it open.
function resolveWindow(range: string, start: string, end: string): { start?: string; end?: string } {
  if (start || end) {
    return { start: start || undefined, end: end || undefined };
  }
  if (range === "custom") {
    return {};
  }
  const preset = RANGE_PRESETS.find((p) => p.key === range) ?? RANGE_PRESETS.find((p) => p.key === DEFAULT_RANGE);
  const seconds = preset?.seconds ?? null;
  return { start: seconds == null ? undefined : isoAgo(seconds), end: undefined };
}

// ---------- small presentational pieces ----------

// Status as a pill, failure-forward: errors use the shared red status surface so
// they pop in a scan; success uses the muted brand tint.
function StatusPill({ status }: { status: string }) {
  const cls =
    status === "error"
      ? "border-red-200 bg-red-50 text-red-700"
      : "border-[var(--otari-line)] bg-[var(--otari-brand-tint)] text-[var(--otari-brand-dark)]";
  return (
    <span className={`inline-flex items-center rounded-full border px-2 py-0.5 text-xs font-medium ${cls}`}>
      {status}
    </span>
  );
}

// Friendly labels for known provenance sources; unknown sources render their slug.
const SOURCE_LABELS: Record<string, string> = { gateway: "Gateway", claude_code: "Claude Code", codex: "Codex" };

function sourceLabel(source: string): string {
  return SOURCE_LABELS[source] ?? source;
}

export async function copyToClipboard(
  text: string,
  clipboard: Pick<Clipboard, "writeText"> | undefined = navigator.clipboard,
): Promise<boolean> {
  if (!clipboard) return false;
  try {
    await clipboard.writeText(text);
    return true;
  } catch {
    return false;
  }
}

function DetailField({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div className="flex flex-col gap-0.5">
      <span className="text-[11px] font-medium uppercase tracking-wide text-[var(--otari-muted)]">{label}</span>
      <span className="text-sm text-[var(--otari-ink)] break-all">{children}</span>
    </div>
  );
}

// The detail panel for one request: a safe error summary plus the metadata that
// does not fit the row. Provider diagnostics stay server-side.
function RequestDetail({ entry }: { entry: UsageEntry }) {
  return (
    <div className="flex flex-col gap-4 px-4 py-4">
      {entry.error_message ? (
        <div className="flex flex-col gap-1.5">
          <span className="text-[11px] font-medium uppercase tracking-wide text-[var(--otari-muted)]">Error</span>
          <pre className="max-h-48 overflow-auto rounded-lg border border-red-200 bg-red-50 p-3 text-xs whitespace-pre-wrap break-all text-red-700">
            The provider returned an error. Inspect gateway logs for details.
          </pre>
        </div>
      ) : null}
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <DetailField label="Provider">{entry.provider ?? "—"}</DetailField>
        <DetailField label="Endpoint">{entry.endpoint}</DetailField>
        <DetailField label="Source">{sourceLabel(entry.source)}</DetailField>
        {entry.source_label ? <DetailField label="Session">{entry.source_label}</DetailField> : null}
        <DetailField label="User">{entry.user_id ?? "—"}</DetailField>
        <DetailField label="API key">{entry.api_key_id ?? "—"}</DetailField>
        <DetailField label="Prompt tokens">{formatTokens(entry.prompt_tokens)}</DetailField>
        <DetailField label="Completion tokens">{formatTokens(entry.completion_tokens)}</DetailField>
        <DetailField label="Total tokens">{formatTokens(entry.total_tokens)}</DetailField>
        <DetailField label="Cost">{formatUSD(entry.cost)}</DetailField>
        <DetailField label="Cache read tokens">{formatTokens(entry.cache_read_tokens)}</DetailField>
        <DetailField label="Cache write tokens">{formatTokens(entry.cache_write_tokens)}</DetailField>
        <DetailField label="1h cache writes">{formatTokens(entry.cache_write_1h_tokens ?? null)}</DetailField>
        <DetailField label="Total time">{formatLatency(entry.latency_ms)}</DetailField>
        <DetailField label="Request ID">{entry.id}</DetailField>
      </div>
      {entry.pricing_breakdown?.length ? (
        <div className="flex flex-col gap-2">
          <span className="text-[11px] font-medium uppercase tracking-wide text-[var(--otari-muted)]">
            Billed meters
          </span>
          <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
            {entry.pricing_breakdown.map((line) => (
              <DetailField key={line.meter} label={line.meter.replaceAll("_", " ")}>
                {formatTokens(line.units)} at {formatUSD(line.rate_per_million)} / 1M, {formatUSD(line.cost)}
              </DetailField>
            ))}
          </div>
        </div>
      ) : null}
    </div>
  );
}

// ---------- page ----------

export function ActivityPage() {
  const users = useUsers();
  const keys = useKeys();
  const keyLabels = useMemo(() => {
    const map = new Map<string, string>();
    for (const k of keys.data ?? []) map.set(k.id, k.key_name ?? `${k.id.slice(0, 8)}…`);
    return map;
  }, [keys.data]);
  const apiKeyLabel = (id: string | null): string => (id === null ? "—" : (keyLabels.get(id) ?? `${id.slice(0, 8)}…`));

  // Filter + pagination state lives in the URL, so a filtered view is shareable
  // and survives the back button. `patch` batches related changes into one entry.
  const url = useUrlState(URL_DEFAULTS);
  const range = url.get("range");
  const startParam = url.get("start_date");
  const endParam = url.get("end_date");
  const statusFilter = url.get("status");
  const modelFilter = url.get("model");
  const userFilter = url.get("user_id");
  const apiKeyFilter = url.get("api_key_id");
  const pricedFilter = url.get("priced");
  const page = url.getNumber("page");
  const pageSize = url.getNumber("size");

  // Snapshot the window so a rolling preset does not recompute "now" every render
  // (which would churn the query key). Re-anchored when the range changes or on refresh.
  const [win, setWin] = useState(() => resolveWindow(range, startParam, endParam));
  useEffect(() => {
    setWin(resolveWindow(range, startParam, endParam));
  }, [range, startParam, endParam]);

  // A custom range is active whenever explicit bounds are set (including a
  // drill-down), otherwise the URL's preset drives the highlight.
  const activeRange = startParam || endParam ? "custom" : range;
  const hasWindow = Boolean(win.start || win.end);

  const priced = pricedFilter === "true" ? true : pricedFilter === "false" ? false : undefined;

  const filters: UsageFilters = useMemo(
    () => ({
      start_date: win.start,
      end_date: win.end,
      status: statusFilter || undefined,
      model: modelFilter.trim() || undefined,
      user_id: userFilter || undefined,
      api_key_id: apiKeyFilter || undefined,
      priced,
    }),
    [win, statusFilter, modelFilter, userFilter, apiKeyFilter, priced],
  );

  const selection = useTableSelection();

  // Any change to the filter set returns to the first page and drops the
  // selection, but not on mount, so a shared URL keeps its page.
  const filtersKey = JSON.stringify(filters);
  const prevFiltersKey = useRef(filtersKey);
  useEffect(() => {
    if (prevFiltersKey.current !== filtersKey) {
      prevFiltersKey.current = filtersKey;
      url.patch({ page: 0 });
      selection.clear();
    }
  }, [filtersKey, url, selection]);

  const usage = useUsageLogs(filters, page, pageSize);
  const count = useUsageCount(filters);

  // Model suggestions: models with usage in the window (other filters applied, the
  // model filter omitted so the full list stays offered).
  const modelSuggestFilters: UsageFilters = useMemo(
    () => ({
      start_date: win.start,
      end_date: win.end,
      status: statusFilter || undefined,
      user_id: userFilter || undefined,
      api_key_id: apiKeyFilter || undefined,
    }),
    [win, statusFilter, userFilter, apiKeyFilter],
  );
  const modelSummary = useUsageSummary(modelSuggestFilters, "day");
  const modelOptions =
    modelSummary.data?.by_model?.filter((r) => !r.is_other && r.key !== null).map((r) => r.key as string) ?? [];
  const keyOptions = (keys.data ?? []).map((k) => ({ value: k.id, label: k.key_name ?? `${k.id.slice(0, 8)}…` }));

  const rows = usage.data ?? [];
  const totalIsExact = count.isSuccess && !count.isPlaceholderData;
  const total = totalIsExact ? (count.data?.total ?? 0) : null;
  const anyFilter = Boolean(
    statusFilter || modelFilter.trim() || userFilter || apiKeyFilter || pricedFilter || hasWindow,
  );
  // On mobile the status/priced/key/user/model controls collapse behind a
  // "Filters" toggle (labelled with the active count) so the request table sits
  // near the top; desktop shows them inline.
  const [filtersOpen, setFiltersOpen] = useState(false);
  const activeFilterCount = [statusFilter, pricedFilter, apiKeyFilter, userFilter, modelFilter.trim()].filter(
    Boolean,
  ).length;

  // Selection targets imported rows only; enforced gateway rows are disabled so
  // bulk delete / set-price can never reach them.
  const selectableKeys = useMemo(() => rows.filter((r) => !r.counts_toward_budget).map((r) => r.id), [rows]);
  const disabledKeys = useMemo(() => rows.filter((r) => r.counts_toward_budget).map((r) => r.id), [rows]);
  const selectedIds = resolveSelectedIds(selection.selectedKeys, selectableKeys);
  const pageSelectedCount = selectedIds.length;
  const hasSelection = selection.allMatching || pageSelectedCount > 0;

  // Total imported rows matching the filter, for the "select all N" affordance
  // and the bulk-op copy; only fetched once there is a selection.
  const importedFilters = useMemo<UsageFilters>(() => ({ ...filters, counts_toward_budget: false }), [filters]);
  const importedCount = useUsageCount(importedFilters, hasSelection);
  const matchingTotal = importedCount.isSuccess ? (importedCount.data?.total ?? null) : null;
  const allPageSelected = selectableKeys.length > 0 && pageSelectedCount === selectableKeys.length;
  const canSelectAllMatching = allPageSelected && matchingTotal != null && matchingTotal > pageSelectedCount;
  const effectiveCount = selection.allMatching ? (matchingTotal ?? pageSelectedCount) : pageSelectedCount;

  const deleteUsage = useDeleteUsage();
  const setPrice = useSetUsagePrice();
  const [deleteOpen, setDeleteOpen] = useState(false);
  const [priceOpen, setPriceOpen] = useState(false);
  const [expandedId, setExpandedId] = useState<string | null>(null);
  const expanded = rows.find((r) => r.id === expandedId) ?? null;

  // A bulk op targets either the current page selection (ids) or, once the operator
  // opted into "all matching", the filter itself (by_filter). The server scopes
  // either to imported rows.
  const selectionBody = (): UsageMutationSelection =>
    selection.allMatching
      ? {
          by_filter: true,
          model: filters.model,
          user_id: filters.user_id,
          api_key_id: filters.api_key_id,
          status: filters.status,
          start_date: filters.start_date,
          end_date: filters.end_date,
          priced: filters.priced,
        }
      : { ids: selectedIds };

  const onDeleteConfirm = () => {
    deleteUsage.mutate(selectionBody(), {
      onSuccess: () => {
        setDeleteOpen(false);
        selection.clear();
      },
    });
  };

  const onSetPrice = (rates: ManualRates) => {
    setPrice.mutate(
      { ...selectionBody(), ...rates },
      {
        onSuccess: () => {
          setPriceOpen(false);
          selection.clear();
        },
      },
    );
  };

  const clearFilters = () => {
    // "All" time plus no filters is the true-empty baseline.
    url.patch({
      range: "all",
      start_date: "",
      end_date: "",
      status: "",
      model: "",
      user_id: "",
      api_key_id: "",
      priced: "",
    });
  };

  const refresh = () => {
    setWin(resolveWindow(range, startParam, endParam));
    void usage.refetch();
    void count.refetch();
  };

  const columns: DataTableColumn<UsageEntry>[] = [
    {
      id: "time",
      header: "Time",
      cell: (e) => (
        <span title={absolute(e.timestamp)} className="text-[var(--otari-muted)]">
          {timeAgo(e.timestamp)}
        </span>
      ),
    },
    { id: "user", header: "User", cell: (e) => e.user_id ?? "—" },
    { id: "model", header: "Model", isRowHeader: true, cell: (e) => e.model },
    { id: "api_key", header: "API key", cell: (e) => <span className="text-[var(--otari-muted)]">{apiKeyLabel(e.api_key_id)}</span> },
    { id: "tokens", header: "Tokens", align: "end", cell: (e) => formatTokens(e.total_tokens) },
    { id: "cost", header: "Cost", align: "end", cell: (e) => formatUSD(e.cost) },
    { id: "latency", header: "Total time", align: "end", cell: (e) => formatLatency(e.latency_ms) },
    { id: "status", header: "Status", cell: (e) => <StatusPill status={e.status} /> },
  ];

  return (
    <div className="flex flex-col gap-6">
      <PageHeader
        title="Activity"
        description="A per-request log of what the gateway served: tokens, cost, latency, and failures. No request or response content is stored."
        action={
          <Button variant="outline" onPress={refresh} isDisabled={usage.isFetching}>
            Refresh
          </Button>
        }
      />

      <ErrorBanner error={usage.error ?? count.error} />

      <div className="flex flex-wrap items-end gap-3">
        <div className="flex flex-wrap gap-2">
          {RANGE_PRESETS.map((preset) => (
            <Button
              key={preset.key}
              size="sm"
              variant={activeRange === preset.key ? "primary" : "outline"}
              onPress={() => url.patch({ range: preset.key, start_date: "", end_date: "" })}
            >
              {preset.label}
            </Button>
          ))}
          <Button
            size="sm"
            variant={activeRange === "custom" ? "primary" : "outline"}
            onPress={() => url.patch({ range: "custom" })}
          >
            Custom…
          </Button>
        </div>
        {activeRange === "custom" ? (
          <div className="flex flex-wrap items-end gap-2">
            <label className="flex flex-col gap-1 text-xs font-medium text-[var(--otari-muted)]">
              From
              <input
                type="datetime-local"
                value={startParam}
                onChange={(event) => url.patch({ start_date: event.target.value })}
                className="rounded-lg border border-[var(--otari-line)] bg-[var(--otari-bg)] px-3 py-2 text-sm text-[var(--otari-ink)] focus:border-[var(--otari-brand)] focus:outline-none"
              />
            </label>
            <label className="flex flex-col gap-1 text-xs font-medium text-[var(--otari-muted)]">
              To
              <input
                type="datetime-local"
                value={endParam}
                onChange={(event) => url.patch({ end_date: event.target.value })}
                className="rounded-lg border border-[var(--otari-line)] bg-[var(--otari-bg)] px-3 py-2 text-sm text-[var(--otari-ink)] focus:border-[var(--otari-brand)] focus:outline-none"
              />
            </label>
          </div>
        ) : null}
        <Button
          size="sm"
          variant="outline"
          className="md:hidden"
          onPress={() => setFiltersOpen((open) => !open)}
          aria-expanded={filtersOpen}
          aria-controls="activity-filters"
        >
          Filters{activeFilterCount ? ` (${activeFilterCount})` : ""}
        </Button>
        <div
          id="activity-filters"
          className={clsx("flex-wrap items-end gap-3 md:flex", filtersOpen ? "flex w-full md:w-auto" : "hidden")}
        >
          <FilterSelect
            id="filter-status"
            label="Status"
            value={statusFilter}
            onChange={(value) => url.patch({ status: value })}
          >
            {STATUS_OPTIONS.map((opt) => (
              <option key={opt.value} value={opt.value}>
                {opt.label}
              </option>
            ))}
          </FilterSelect>
          <FilterSelect
            id="filter-priced"
            label="Priced?"
            value={pricedFilter}
            onChange={(value) => url.patch({ priced: value })}
          >
            {PRICED_OPTIONS.map((opt) => (
              <option key={opt.value} value={opt.value}>
                {opt.label}
              </option>
            ))}
          </FilterSelect>
          <FilterComboBox
            label="API key"
            value={apiKeyFilter}
            onChange={(value) => url.patch({ api_key_id: value })}
            placeholder="All keys"
            options={keyOptions}
          />
          <FilterComboBox
            label="User"
            value={userFilter}
            onChange={(value) => url.patch({ user_id: value })}
            placeholder="All users"
            options={(users.data ?? []).map((u) => ({
              value: u.user_id,
              label: u.alias ? `${u.alias} (${u.user_id})` : u.user_id,
            }))}
          />
          <FilterComboBox
            label="Model"
            value={modelFilter}
            onChange={(value) => url.patch({ model: value })}
            allowsCustom
            placeholder="Any model"
            options={(modelFilter && !modelOptions.includes(modelFilter)
              ? [modelFilter, ...modelOptions]
              : modelOptions
            ).map((m) => ({ value: m, label: m }))}
          />
          {anyFilter ? (
            <Button size="sm" variant="ghost" onPress={clearFilters}>
              Clear filters
            </Button>
          ) : null}
        </div>
      </div>

      {hasSelection ? (
        <BulkActionBar
          selectedCount={effectiveCount}
          allMatching={selection.allMatching}
          matchingTotal={matchingTotal}
          canSelectAllMatching={canSelectAllMatching}
          onSelectAllMatching={selection.enableAllMatching}
          onClear={selection.clear}
        >
          <Button size="sm" variant="primary" onPress={() => setPriceOpen(true)}>
            Set price
          </Button>
          <Button size="sm" variant="danger" onPress={() => setDeleteOpen(true)}>
            Delete
          </Button>
        </BulkActionBar>
      ) : null}

      <DataTable
        ariaLabel="Activity log"
        columns={columns}
        rows={rows}
        getRowKey={(e) => e.id}
        isLoading={usage.isLoading}
        emptyContent={anyFilter ? "No requests match these filters." : "No requests recorded yet."}
        selectionMode="multiple"
        selectedKeys={selection.selectedKeys}
        onSelectionChange={selection.onSelectionChange}
        disabledKeys={disabledKeys}
        onRowAction={(key) => setExpandedId((current) => (current === key ? null : key))}
        rowClassName={(e) => (e.status === "error" ? "bg-red-50" : undefined)}
      />

      {expanded ? (
        <Card>
          <Card.Content className="p-0">
            <div className="flex items-center justify-between border-b border-[var(--otari-line)] px-4 py-2">
              <span className="text-sm font-medium text-[var(--otari-ink)]">Request detail</span>
              <Button size="sm" variant="ghost" onPress={() => setExpandedId(null)}>
                Close
              </Button>
            </div>
            <RequestDetail entry={expanded} />
          </Card.Content>
        </Card>
      ) : null}

      <TablePagination
        page={page}
        pageSize={pageSize}
        total={total}
        rowsOnPage={rows.length}
        onPageChange={(next) => url.patch({ page: next })}
        onPageSizeChange={(size) => url.patch({ size, page: 0 })}
        isFetching={usage.isFetching}
        hasNextFallback={rows.length === pageSize}
      />

      <ConfirmDialog
        isOpen={deleteOpen}
        onOpenChange={setDeleteOpen}
        heading="Delete usage rows"
        body={`Delete ${effectiveCount.toLocaleString()} imported ${
          effectiveCount === 1 ? "row" : "rows"
        }? Only imported rows are removed, and this cannot be undone.`}
        confirmLabel="Delete"
        isPending={deleteUsage.isPending}
        error={deleteUsage.error}
        onConfirm={onDeleteConfirm}
      />

      <SetPriceDialog
        isOpen={priceOpen}
        onOpenChange={setPriceOpen}
        targetCount={effectiveCount}
        isPending={setPrice.isPending}
        error={setPrice.error}
        onSubmit={onSetPrice}
      />
    </div>
  );
}
