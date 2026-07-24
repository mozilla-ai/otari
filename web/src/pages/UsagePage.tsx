import { Button, Spinner } from "@heroui/react";
import clsx from "clsx";
import { useEffect, useMemo, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";

import {
  useDeleteUsage,
  useKeys,
  useSetUsagePrice,
  useUsageCount,
  useUsageLogs,
  useUsageSummary,
  useUsers,
} from "@/api/hooks";
import type {
  UsageBucket,
  UsageEntry,
  UsageFilters,
  UsageGroupRow,
  UsageMutationSelection,
  UsageSeriesPoint,
} from "@/api/types";
import { BulkActionBar } from "@/components/BulkActionBar";
import { BarTrendChart, Sparkline } from "@/components/charts";
import { ConfirmDialog } from "@/components/ConfirmDialog";
import { DataTable, type DataTableColumn } from "@/components/DataTable";
import { SetPriceDialog, type ManualRates } from "@/components/SetPriceDialog";
import { TablePagination } from "@/components/TablePagination";
import { DeltaHint, ErrorBanner, FilterComboBox, PageHeader, StatCard } from "@/components/ui";
import { deltaFraction, formatPct, formatTokens, formatUsd } from "@/lib/format";
import { resolveSelectedIds, useTableSelection } from "@/lib/tableSelection";

// ---------- formatting ----------

// Compact currency (formatUsd), token counts (formatTokens), percentages
// (formatPct) and the period-over-period helpers (deltaFraction / DeltaHint) are
// shared with the overview page from @/lib/format and @/components/ui. Only the
// two formatters specific to this page stay local.
function formatCount(value: number): string {
  return value.toLocaleString();
}

function formatLatency(ms: number | null): string {
  if (ms === null) return "—";
  if (ms < 1000) return `${Math.round(ms)} ms`;
  return `${(ms / 1000).toFixed(2)} s`;
}

// ---------- filter option sets (mirrors ActivityPage, plus 30d for spend) ----------

const HOUR_S = 3_600;
const DAY_S = 86_400;

interface Preset {
  label: string;
  seconds: number | null;
  bucket: UsageBucket;
}

// Sub-day windows bucket hourly; longer ranges bucket daily. "All" has no lower
// bound, so no previous period to compare against (deltas hide).
const TIME_PRESETS: Preset[] = [
  { label: "Last hour", seconds: HOUR_S, bucket: "hour" },
  { label: "24h", seconds: DAY_S, bucket: "hour" },
  { label: "7d", seconds: 7 * DAY_S, bucket: "day" },
  { label: "30d", seconds: 30 * DAY_S, bucket: "day" },
  { label: "All", seconds: null, bucket: "day" },
];

const DEFAULT_PRESET = TIME_PRESETS[3]; // 30d: a spend investigation is usually monthly.

const TABLE_TOP_N = 15;

function isoAgo(seconds: number): string {
  return new Date(Date.now() - seconds * 1000).toISOString();
}

// ---------- breakdown table ----------

interface BreakdownProps {
  title: string;
  rows: UsageGroupRow[];
  totalCost: number;
  emptyLabel: string;
  // Turns a row key into the Activity-page filter to drill into (model vs user).
  onDrill: (key: string) => void;
  loading: boolean;
}

// One breakdown (by model / by user). Rows are spend-ranked with an inline
// share-of-total bar; clicking a named row drills into the Activity log filtered
// to that dimension. The synthesized "other" fold row (null key) is shown but not
// clickable, so the visible spend still reconciles with the total-spend tile.
const OTHER_KEY = "__other__";
const UNKNOWN_KEY = "__unknown__";

function BreakdownTable({
  title,
  rows,
  totalCost,
  emptyLabel,
  onDrill,
  loading,
}: BreakdownProps) {
  const [showAll, setShowAll] = useState(false);
  const visible = showAll ? rows : rows.slice(0, TABLE_TOP_N);
  const hidden = rows.length - visible.length;

  // Fold and deleted-user rows both carry a null key and are not drill targets;
  // give them stable sentinel keys so the collection stays unique.
  const rowKey = (row: UsageGroupRow) => (row.is_other ? OTHER_KEY : (row.key ?? UNKNOWN_KEY));

  const columns: DataTableColumn<UsageGroupRow>[] = [
    {
      id: "name",
      header: title.replace("Spend by ", ""),
      isRowHeader: true,
      cell: (row) => {
        const share = totalCost > 0 ? row.cost / totalCost : 0;
        return (
          <div className="flex flex-col gap-1">
            <span className="truncate text-[var(--otari-ink)]">
              {row.is_other
                ? `Other (${row.requests.toLocaleString()} req)`
                : row.key === null
                  ? "(unknown)"
                  : row.key}
            </span>
            <span className="h-1 w-full overflow-hidden rounded-full bg-[var(--otari-line)]">
              <span
                className="block h-full rounded-full bg-[var(--otari-brand)]"
                style={{ width: `${Math.min(100, share * 100)}%` }}
              />
            </span>
          </div>
        );
      },
    },
    { id: "requests", header: "Requests", align: "end", cell: (row) => <span className="text-[var(--otari-muted)]">{formatCount(row.requests)}</span> },
    { id: "spend", header: "Spend", align: "end", cell: (row) => <span className="text-[var(--otari-ink)]">{formatUsd(row.cost)}</span> },
  ];

  return (
    <div className="flex flex-col gap-2">
      <h2 className="text-sm font-semibold text-[var(--otari-ink)]">{title}</h2>
      <DataTable
        ariaLabel={title}
        columns={columns}
        rows={visible}
        getRowKey={rowKey}
        isLoading={loading}
        emptyContent={emptyLabel}
        onRowAction={(key) => {
          // Only real groups drill; the fold and deleted-user rows have no id to filter on.
          if (key !== OTHER_KEY && key !== UNKNOWN_KEY) {
            onDrill(key);
          }
        }}
      />
      {!loading && hidden > 0 ? (
        <Button size="sm" variant="ghost" onPress={() => setShowAll(true)}>
          Show all {rows.length}
        </Button>
      ) : null}
      {!loading && showAll && rows.length > TABLE_TOP_N ? (
        <Button size="sm" variant="ghost" onPress={() => setShowAll(false)}>
          Show top {TABLE_TOP_N}
        </Button>
      ) : null}
    </div>
  );
}

// ---------- chart ----------

type ChartMetric = "cost" | "tokens" | "requests";

const METRIC_TABS: { key: ChartMetric; label: string }[] = [
  { key: "cost", label: "Cost" },
  { key: "tokens", label: "Tokens" },
  { key: "requests", label: "Requests" },
];

function metricValue(point: UsageSeriesPoint, metric: ChartMetric): number {
  return metric === "cost" ? point.cost : metric === "tokens" ? point.tokens : point.requests;
}

function formatMetric(value: number, metric: ChartMetric): string {
  return metric === "cost" ? formatUsd(value) : metric === "tokens" ? formatTokens(value) : formatCount(value);
}

function formatBucketLabel(iso: string, bucket: UsageBucket): string {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  if (bucket === "hour") {
    return d.toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit", timeZone: "UTC" });
  }
  return d.toLocaleDateString(undefined, { month: "short", day: "numeric", timeZone: "UTC" });
}

// The selected metric's series shaped for the shared trend chart, plus the true
// peak and count for the caption. The peak is the real data max (0 for an
// all-zero window), so the caption and the chart's auto-scaled bars agree; the
// chart is only rendered for a non-empty series, so the empty guard is defensive.
function trendData(series: UsageSeriesPoint[], metric: ChartMetric, bucket: UsageBucket) {
  const points = series.map((point) => ({
    label: formatBucketLabel(point.bucket_start, bucket),
    value: metricValue(point, metric),
  }));
  const peak = points.length ? Math.max(...points.map((p) => p.value)) : 0;
  return { points, peak, count: series.length };
}

// ---------- individual requests (raw rows under the breakdowns) ----------

const REQUESTS_PAGE_SIZE = 50;

function requestTimeAgo(iso: string): string {
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

function RequestStatusPill({ status }: { status: string }) {
  return (
    <span
      className={`inline-flex items-center rounded-full border px-2 py-0.5 text-xs font-medium ${
        status === "error"
          ? "border-red-200 bg-red-50 text-red-700"
          : "border-[var(--otari-line)] bg-[var(--otari-brand-tint)] text-[var(--otari-brand-dark)]"
      }`}
    >
      {status}
    </span>
  );
}

// Column definitions and the row-key getter live at module scope so their
// identity is stable across renders: DataTable caches rendered rows on them,
// and a new array each render would rebuild every row on each selection click.
const REQUEST_COLUMNS: DataTableColumn<UsageEntry>[] = [
  {
    id: "time",
    header: "Time",
    isRowHeader: true,
    cell: (e) => (
      <span className="text-[var(--otari-muted)]" title={new Date(e.timestamp).toLocaleString()}>
        {requestTimeAgo(e.timestamp)}
      </span>
    ),
  },
  { id: "model", header: "Model", cell: (e) => <span className="text-[var(--otari-ink)]">{e.model}</span> },
  { id: "user", header: "User", cell: (e) => <span className="text-[var(--otari-muted)]">{e.user_id ?? "—"}</span> },
  { id: "tokens", header: "Tokens", align: "end", cell: (e) => (e.total_tokens === null ? "—" : e.total_tokens.toLocaleString()) },
  { id: "cost", header: "Cost", align: "end", cell: (e) => (e.cost === null ? "—" : formatUsd(e.cost)) },
  { id: "status", header: "Status", cell: (e) => <RequestStatusPill status={e.status} /> },
];

const getRequestRowKey = (e: UsageEntry): string => e.id;

// The raw usage rows for the current filter window, paginated, with selection
// and the imported-row bulk actions (Delete, Set price). Only imported rows are
// selectable; enforced gateway rows are disabled so bulk ops never touch them.
function UsageRequests({ filters, anyFilter }: { filters: UsageFilters; anyFilter: boolean }) {
  const [page, setPage] = useState(0);
  const [pageSize, setPageSize] = useState(REQUESTS_PAGE_SIZE);
  const selection = useTableSelection();
  const deleteUsage = useDeleteUsage();
  const setPrice = useSetUsagePrice();
  const [deleteOpen, setDeleteOpen] = useState(false);
  const [priceOpen, setPriceOpen] = useState(false);

  const filtersKey = JSON.stringify(filters);
  const prevFiltersKey = useRef(filtersKey);
  useEffect(() => {
    if (prevFiltersKey.current !== filtersKey) {
      prevFiltersKey.current = filtersKey;
      setPage(0);
      selection.clear();
    }
  }, [filtersKey, selection]);

  const usage = useUsageLogs(filters, page, pageSize);
  const count = useUsageCount(filters);
  const rows = usage.data ?? [];
  const totalIsExact = count.isSuccess && !count.isPlaceholderData;
  const total = totalIsExact ? (count.data?.total ?? 0) : null;

  const selectableKeys = rows.filter((r) => !r.counts_toward_budget).map((r) => r.id);
  const disabledKeys = rows.filter((r) => r.counts_toward_budget).map((r) => r.id);
  const selectedIds = resolveSelectedIds(selection.selectedKeys, selectableKeys);
  const pageSelectedCount = selectedIds.length;
  const hasSelection = selection.allMatching || pageSelectedCount > 0;

  const importedFilters = useMemo<UsageFilters>(() => ({ ...filters, counts_toward_budget: false }), [filters]);
  const importedCount = useUsageCount(importedFilters, hasSelection);
  const matchingTotal = importedCount.isSuccess ? (importedCount.data?.total ?? null) : null;
  const allPageSelected = selectableKeys.length > 0 && pageSelectedCount === selectableKeys.length;
  const canSelectAllMatching = allPageSelected && matchingTotal != null && matchingTotal > pageSelectedCount;
  const effectiveCount = selection.allMatching ? (matchingTotal ?? pageSelectedCount) : pageSelectedCount;

  const selectionBody = (): UsageMutationSelection =>
    selection.allMatching
      ? {
          by_filter: true,
          model: filters.model,
          user_id: filters.user_id,
          api_key_id: filters.api_key_id,
          start_date: filters.start_date,
          end_date: filters.end_date,
          priced: filters.priced,
        }
      : { ids: selectedIds };

  const onDeleteConfirm = () =>
    deleteUsage.mutate(selectionBody(), {
      onSuccess: () => {
        setDeleteOpen(false);
        selection.clear();
      },
    });

  const onSetPrice = (rates: ManualRates) =>
    setPrice.mutate(
      { ...selectionBody(), ...rates },
      {
        onSuccess: () => {
          setPriceOpen(false);
          selection.clear();
        },
      },
    );

  return (
    <div className="flex flex-col gap-3">
      <h2 className="text-sm font-semibold text-[var(--otari-ink)]">Individual requests</h2>
      <ErrorBanner error={usage.error ?? count.error} />

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
        ariaLabel="Individual requests"
        columns={REQUEST_COLUMNS}
        rows={rows}
        getRowKey={getRequestRowKey}
        isLoading={usage.isLoading}
        emptyContent={anyFilter ? "No requests match these filters." : "No requests recorded yet."}
        selectionMode="multiple"
        selectedKeys={selection.selectedKeys}
        onSelectionChange={selection.onSelectionChange}
        disabledKeys={disabledKeys}
      />

      <TablePagination
        page={page}
        pageSize={pageSize}
        total={total}
        rowsOnPage={rows.length}
        onPageChange={setPage}
        onPageSizeChange={(size) => {
          setPageSize(size);
          setPage(0);
        }}
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

// ---------- page ----------

export function UsagePage() {
  const navigate = useNavigate();
  const users = useUsers();
  const keys = useKeys();

  const [preset, setPreset] = useState<Preset>(DEFAULT_PRESET);
  const [startDate, setStartDate] = useState<string | undefined>(() =>
    DEFAULT_PRESET.seconds === null ? undefined : isoAgo(DEFAULT_PRESET.seconds),
  );
  // Custom range: an explicit from/to window (like the Activity page), buckets
  // daily. When active it overrides the preset window.
  const [customMode, setCustomMode] = useState(false);
  const [customFrom, setCustomFrom] = useState("");
  const [customTo, setCustomTo] = useState("");
  const [modelFilter, setModelFilter] = useState("");
  const [userFilter, setUserFilter] = useState("");
  const [apiKeyFilter, setApiKeyFilter] = useState("");
  const [metric, setMetric] = useState<ChartMetric>("cost");

  const winStart = customMode ? customFrom || undefined : startDate;
  const winEnd = customMode ? customTo || undefined : undefined;
  const bucket: UsageBucket = customMode ? "day" : preset.bucket;

  const filters: UsageFilters = useMemo(
    () => ({
      start_date: winStart,
      end_date: winEnd,
      model: modelFilter.trim() || undefined,
      user_id: userFilter || undefined,
      api_key_id: apiKeyFilter || undefined,
    }),
    [winStart, winEnd, modelFilter, userFilter, apiKeyFilter],
  );

  // The immediately-preceding window of equal length, for period-over-period
  // deltas. Only meaningful for a bounded range.
  const previousFilters: UsageFilters | null = useMemo(() => {
    if (customMode) {
      if (!winStart || !winEnd) return null;
      const span = new Date(winEnd).getTime() - new Date(winStart).getTime();
      if (!(span > 0)) return null;
      return { ...filters, start_date: new Date(new Date(winStart).getTime() - span).toISOString(), end_date: winStart };
    }
    if (preset.seconds === null || !startDate) return null;
    return {
      ...filters,
      start_date: new Date(new Date(startDate).getTime() - preset.seconds * 1000).toISOString(),
      // Cap the previous window at the current window's start.
      end_date: startDate,
    };
  }, [customMode, winStart, winEnd, filters, preset.seconds, startDate]);

  const summary = useUsageSummary(filters, bucket);
  const previous = useUsageSummary(previousFilters ?? filters, bucket, previousFilters !== null);

  const data = summary.data;
  const totals = data?.totals;
  const prevTotals = previousFilters !== null ? previous.data?.totals : undefined;
  const costDelta = totals ? deltaFraction(totals.cost, prevTotals?.cost) : null;

  // Model typeahead options: the in-window models. Sourced from a summary that
  // omits the model filter, so the list stays complete when a model is selected,
  // and derived directly from query data rather than mirrored into state.
  const modelSuggestFilters: UsageFilters = useMemo(() => ({ ...filters, model: undefined }), [filters]);
  const modelSuggest = useUsageSummary(modelSuggestFilters, bucket);
  const modelOptions =
    modelSuggest.data?.by_model?.filter((r) => !r.is_other && r.key !== null).map((r) => r.key as string) ?? [];

  const userOptions = (users.data ?? []).map((u) => ({
    value: u.user_id,
    label: u.alias ? `${u.alias} (${u.user_id})` : u.user_id,
  }));
  // API key options label by name (falling back to a short id), value is the id.
  const keyOptions = (keys.data ?? []).map((k) => ({
    value: k.id,
    label: k.key_name ?? `${k.id.slice(0, 8)}…`,
  }));
  // Keep a selected-but-not-in-list model visible (e.g. seeded from elsewhere).
  const modelOptionList = (
    modelFilter && !modelOptions.includes(modelFilter) ? [modelFilter, ...modelOptions] : modelOptions
  ).map((m) => ({ value: m, label: m }));

  const timeFiltered = customMode ? Boolean(winStart || winEnd) : preset.seconds !== null;
  const anyFilter = Boolean(modelFilter.trim() || userFilter || apiKeyFilter || timeFiltered);
  // On mobile the user/model/key controls collapse behind a "Filters" toggle so
  // the tiles and breakdowns sit near the top; desktop shows them inline.
  const [filtersOpen, setFiltersOpen] = useState(false);
  const activeFilterCount = [modelFilter.trim(), userFilter, apiKeyFilter].filter(Boolean).length;
  // Distinguish "this gateway has never served a request" from "no rows match
  // these filters": the first is an onboarding state, the second is a filter hint.
  const isEmptyEver = Boolean(data && totals && totals.request_count === 0 && !anyFilter);

  const pickPreset = (next: Preset) => {
    setCustomMode(false);
    setPreset(next);
    setStartDate(next.seconds === null ? undefined : isoAgo(next.seconds));
  };

  const clearFilters = () => {
    pickPreset(TIME_PRESETS[TIME_PRESETS.length - 1]); // All
    setCustomFrom("");
    setCustomTo("");
    setModelFilter("");
    setUserFilter("");
    setApiKeyFilter("");
  };

  const refresh = () => {
    if (preset.seconds !== null) {
      setStartDate(isoAgo(preset.seconds));
    }
    void summary.refetch();
  };

  // Drill from a breakdown row into the Activity log, pre-filtering on the picked
  // dimension plus the current time window.
  const drillTo = (params: Record<string, string | undefined>) => {
    const search = new URLSearchParams();
    if (startDate) search.set("start_date", startDate);
    for (const [k, v] of Object.entries(params)) {
      if (v) search.set(k, v);
    }
    navigate(`/activity?${search.toString()}`);
  };

  const errorRate = totals && totals.request_count > 0 ? totals.error_count / totals.request_count : 0;

  // The bucketed series is already on the wire; reuse it for tile sparklines. A
  // single point has no trend to draw, so sparklines only appear with 2+ buckets.
  const series = data?.series ?? [];
  const hasTrend = series.length > 1;
  const trend = trendData(series, metric, bucket);

  return (
    <div className="flex flex-col gap-6">
      <PageHeader
        title="Usage & analytics"
        description="Aggregate spend, tokens, and request volume over time. Click a model or user to drill into the request log."
        action={
          <Button variant="outline" onPress={refresh} isDisabled={summary.isFetching}>
            Refresh
          </Button>
        }
      />

      <ErrorBanner error={summary.error} />

      {/* Filters */}
      <div className="flex flex-wrap items-end gap-3">
        <div className="flex flex-wrap gap-2">
          {TIME_PRESETS.map((option) => (
            <Button
              key={option.label}
              size="sm"
              variant={!customMode && preset.label === option.label ? "primary" : "outline"}
              onPress={() => pickPreset(option)}
            >
              {option.label}
            </Button>
          ))}
          <Button size="sm" variant={customMode ? "primary" : "outline"} onPress={() => setCustomMode(true)}>
            Custom…
          </Button>
        </div>
        {customMode ? (
          <div className="flex flex-wrap items-end gap-2">
            <label className="flex flex-col gap-1 text-xs font-medium text-[var(--otari-muted)]">
              From
              <input
                type="datetime-local"
                value={customFrom}
                onChange={(event) => setCustomFrom(event.target.value)}
                className="rounded-lg border border-[var(--otari-line)] bg-[var(--otari-bg)] px-3 py-2 text-sm text-[var(--otari-ink)] focus:border-[var(--otari-brand)] focus:outline-none"
              />
            </label>
            <label className="flex flex-col gap-1 text-xs font-medium text-[var(--otari-muted)]">
              To
              <input
                type="datetime-local"
                value={customTo}
                onChange={(event) => setCustomTo(event.target.value)}
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
          aria-controls="usage-filters"
        >
          Filters{activeFilterCount ? ` (${activeFilterCount})` : ""}
        </Button>
        <div
          id="usage-filters"
          className={clsx("flex-wrap items-end gap-3 md:flex", filtersOpen ? "flex w-full md:w-auto" : "hidden")}
        >
          <FilterComboBox
            label="User"
            value={userFilter}
            onChange={setUserFilter}
            options={userOptions}
            placeholder="All users"
          />
          <FilterComboBox
            label="Model"
            value={modelFilter}
            onChange={setModelFilter}
            options={modelOptionList}
            placeholder="All models"
          />
          <FilterComboBox
            label="API key"
            value={apiKeyFilter}
            onChange={setApiKeyFilter}
            options={keyOptions}
            placeholder="All keys"
          />
          {anyFilter ? (
            <Button size="sm" variant="ghost" onPress={clearFilters}>
              Clear filters
            </Button>
          ) : null}
        </div>
      </div>

      {isEmptyEver ? (
        <div className="rounded-xl border border-[var(--otari-line)] bg-[var(--otari-surface)] px-4 py-10 text-center text-sm text-[var(--otari-muted)]">
          No usage yet. Once the gateway serves requests, spend and volume appear here.
        </div>
      ) : (
        <>
          {/* Tiles. A responsive grid, matching OverviewPage: StatCard is
              flex-1/min-w-0 (sized by its track), so a wrapping flex row lets
              all seven tiles shrink onto one unusable line on phones instead
              of wrapping. */}
          <div className="grid grid-cols-2 gap-4 sm:grid-cols-3 xl:grid-cols-4">
            <StatCard
              label="Tracked cost"
              value={totals ? formatUsd(totals.cost) : "—"}
              hint={
                totals ? (
                  <span className="text-[var(--otari-muted)]">
                    <DeltaHint fraction={costDelta} />
                    {totals.unpriced_requests
                      ? `${costDelta !== null ? " · " : ""}${formatCount(totals.unpriced_requests)} unpriced`
                      : null}
                  </span>
                ) : null
              }
              chart={hasTrend ? <Sparkline values={series.map((p) => p.cost)} ariaLabel="Spend trend over the selected window" /> : undefined}
            />
            <StatCard
              label="Requests"
              value={totals ? formatCount(totals.request_count) : "—"}
              hint={
                totals ? (
                  <span className="text-[var(--otari-muted)]">
                    {formatPct(errorRate)} errors
                    {prevTotals ? (
                      <>
                        {" · "}
                        <DeltaHint fraction={deltaFraction(totals.request_count, prevTotals.request_count)} />
                      </>
                    ) : null}
                  </span>
                ) : null
              }
              chart={
                hasTrend ? (
                  <Sparkline values={series.map((p) => p.requests)} ariaLabel="Request volume trend over the selected window" />
                ) : undefined
              }
            />
            <StatCard
              label="Tokens"
              value={totals ? formatTokens(totals.total_tokens) : "—"}
              hint={totals ? <DeltaHint fraction={deltaFraction(totals.total_tokens, prevTotals?.total_tokens)} /> : null}
              chart={hasTrend ? <Sparkline values={series.map((p) => p.tokens)} ariaLabel="Token usage trend over the selected window" /> : undefined}
            />
            <StatCard
              label="Cache read"
              value={totals ? formatTokens(totals.cache_read_tokens) : "—"}
              hint={
                totals ? (
                  <DeltaHint fraction={deltaFraction(totals.cache_read_tokens, prevTotals?.cache_read_tokens)} />
                ) : null
              }
            />
            <StatCard
              label="Cache write"
              value={totals ? formatTokens(totals.cache_write_tokens) : "—"}
              hint={
                totals ? (
                  <DeltaHint fraction={deltaFraction(totals.cache_write_tokens, prevTotals?.cache_write_tokens)} />
                ) : null
              }
            />
            <StatCard
              label="1h cache write"
              value={totals ? formatTokens(totals.cache_write_1h_tokens ?? 0) : "—"}
              hint={
                totals ? (
                  <DeltaHint
                    fraction={deltaFraction(totals.cache_write_1h_tokens ?? 0, prevTotals?.cache_write_1h_tokens ?? 0)}
                  />
                ) : null
              }
            />
            <StatCard label="Avg latency" value={totals ? formatLatency(totals.avg_latency_ms) : "—"} />
          </div>

          {/* Breakdowns: the answer to "where is my money going?", above the trend. */}
          <div className="grid gap-6 lg:grid-cols-2">
            <BreakdownTable
              title="Spend by model"
              rows={data?.by_model ?? []}
              totalCost={totals?.cost ?? 0}
              emptyLabel={anyFilter ? "No usage matches these filters." : "No usage recorded yet."}
              // Drilling into a model keeps the other active filters (user, API key),
              // so the log stays scoped to them instead of showing every request for
              // the model.
              onDrill={(key) =>
                drillTo({ model: key, user_id: userFilter || undefined, api_key_id: apiKeyFilter || undefined })
              }
              loading={summary.isLoading}
            />
            <BreakdownTable
              title="Spend by user"
              rows={data?.by_user ?? []}
              totalCost={totals?.cost ?? 0}
              emptyLabel={anyFilter ? "No usage matches these filters." : "No usage recorded yet."}
              // Likewise, drilling into a user keeps the other active filters (model,
              // API key).
              onDrill={(key) =>
                drillTo({
                  user_id: key,
                  model: modelFilter.trim() || undefined,
                  api_key_id: apiKeyFilter || undefined,
                })
              }
              loading={summary.isLoading}
            />
          </div>

          {/* Raw rows for the same window, with the imported-row bulk actions. */}
          <UsageRequests filters={filters} anyFilter={anyFilter} />

          {/* Trend */}
          <div className="flex flex-col gap-3 rounded-xl border border-[var(--otari-line)] bg-[var(--otari-surface)] p-4">
            <div className="flex items-center justify-between gap-3">
              <h2 className="text-sm font-semibold text-[var(--otari-ink)]">Over time</h2>
              <div className="inline-flex gap-1.5">
                {METRIC_TABS.map((tab) => (
                  <Button
                    key={tab.key}
                    size="sm"
                    variant={metric === tab.key ? "primary" : "outline"}
                    onPress={() => setMetric(tab.key)}
                  >
                    {tab.label}
                  </Button>
                ))}
                {summary.isFetching ? <Spinner size="sm" /> : null}
              </div>
            </div>
            {summary.isLoading ? (
              <div className="flex h-48 items-center justify-center">
                <Spinner size="sm" />
              </div>
            ) : series.length === 0 ? (
              <div className="flex h-48 items-center justify-center text-sm text-[var(--otari-muted)]">
                No data in this range.
              </div>
            ) : (
              <figure className="flex flex-col gap-2">
                <BarTrendChart
                  data={trend.points}
                  formatValue={(value) => formatMetric(value, metric)}
                  ariaLabel={`${metric} per ${bucket}`}
                />
                <figcaption className="text-xs text-[var(--otari-muted)]">
                  {formatMetric(trend.peak, metric)} peak · {trend.count} {bucket === "hour" ? "hours" : "days"} (times
                  in UTC)
                </figcaption>
              </figure>
            )}
          </div>
        </>
      )}
    </div>
  );
}
