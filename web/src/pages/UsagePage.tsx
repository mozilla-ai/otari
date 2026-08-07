import { Button, Spinner } from "@heroui/react";
import { useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";

import { ApiError } from "@/api/client";
import { NO_BREAKDOWNS, useUsageGroupedSeries, useUsageSummary } from "@/api/hooks";
import type {
  SummaryDimension,
  UsageBucket,
  UsageFilters,
  UsageGroupBy,
  UsageGroupRow,
  UsageSeriesPoint,
  UsageSummary,
} from "@/api/types";
import { ChartLegend, Sparkline, TrendChart, type SeriesDef, type StackedPoint } from "@/components/charts";
import { DataTable, type DataTableColumn } from "@/components/DataTable";
import { FilterChips, type FilterChip } from "@/components/FilterChips";
import {
  DeltaHint,
  EmptyState,
  ErrorBanner,
  FilterMultiComboBox,
  FilterSelect,
  PageHeader,
  RefreshButton,
  StatCard,
} from "@/components/ui";
import { deltaFraction, formatPct, formatTokens, formatUsd } from "@/lib/format";
import {
  bucketForWindow,
  findPreset,
  formatWindowLabel,
  isoAgo,
  rangeFromBuckets,
  type RangePreset,
  USAGE_DEFAULT_KEY,
  USAGE_PRESETS,
} from "@/lib/timeRange";

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

function formatBucketLabel(iso: string, bucket: UsageBucket): string {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  if (bucket === "hour") {
    return d.toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit", timeZone: "UTC" });
  }
  return d.toLocaleDateString(undefined, { month: "short", day: "numeric", timeZone: "UTC" });
}

// ---------- window presets ----------
//
// The time presets and window math live in `@/lib/timeRange` and are shared
// with the Activity page. 30d default: a spend investigation is usually monthly.

const DEFAULT_PRESET = findPreset(USAGE_PRESETS, USAGE_DEFAULT_KEY) as RangePreset;

const TABLE_TOP_N = 15;

// ---------- the analytics chart: metric × group-by ----------
//
// The model every major usage dashboard converged on (OpenAI's usage page,
// the Anthropic console, Grafana-style tooling): one main time-series chart, a
// metric selector, and a group-by dimension that splits the series into stacked
// bars. Ungrouped views use the richest encoding each metric has: tokens stack
// their billed composition (the same encoding as the Activity token bar),
// requests split success/error. Dragging across the chart zooms the window.

type ChartMetric = "cost" | "tokens" | "requests";

const METRIC_TABS: { key: ChartMetric; label: string }[] = [
  { key: "cost", label: "Cost" },
  { key: "tokens", label: "Tokens" },
  { key: "requests", label: "Requests" },
];

const GROUP_OPTIONS: { value: "" | UsageGroupBy; label: string }[] = [
  { value: "", label: "None" },
  { value: "model", label: "Model" },
  { value: "user_id", label: "User" },
  { value: "api_key_id", label: "API key" },
  { value: "source", label: "Source" },
];

// Billed token composition, bottom-up: input side first (fresh, then the two
// cache buckets), then output. One hue at four lightnesses — the same encoding,
// same order, and same tokens as the Activity page's per-row bar, so the two
// surfaces read as one system.
const COMPOSITION_SERIES: SeriesDef[] = [
  { key: "fresh", label: "Fresh input", color: "var(--otari-ink)" },
  { key: "cache_read", label: "Cache read", color: "var(--otari-brand)" },
  { key: "cache_write", label: "Cache write", color: "var(--otari-brand-soft)" },
  { key: "output", label: "Output", color: "var(--otari-brand-dark)" },
];

const REQUEST_SERIES: SeriesDef[] = [
  { key: "success", label: "Succeeded", color: "var(--otari-brand)" },
  { key: "errors", label: "Failed", color: "var(--otari-danger)" },
];

// The fixed categorical palette for grouped series (validated in globals.css);
// slot order is the CVD-safety mechanism, so groups take slots in server rank
// order and the fold always wears the neutral.
const CAT_COLORS = [
  "var(--otari-cat-1)",
  "var(--otari-cat-2)",
  "var(--otari-cat-3)",
  "var(--otari-cat-4)",
  "var(--otari-cat-5)",
  "var(--otari-cat-6)",
  "var(--otari-cat-7)",
  "var(--otari-cat-8)",
];
const OTHER_COLOR = "var(--otari-cat-other)";

function metricFormatter(metric: ChartMetric): (value: number) => string {
  return metric === "cost" ? formatUsd : metric === "tokens" ? formatTokens : formatCount;
}

// ---------- breakdown table (tabbed by dimension) ----------

interface BreakdownProps {
  dimensionLabel: string;
  rows: UsageGroupRow[];
  totalCost: number;
  emptyLabel: string;
  // How a real group whose column was NULL reads. The default suits an id that
  // has gone missing (a deleted user); a dimension where NULL is a normal state
  // (gateway rows carry no session label) passes its own wording.
  unknownLabel?: string;
  // Turns a row key into the Activity-page filter to drill into.
  onDrill: (key: string) => void;
  loading: boolean;
}

// One breakdown dimension. Rows are spend-ranked with an inline share-of-total
// bar; clicking a named row drills into the Activity log filtered to that
// dimension. The synthesized "other" fold row (null key) is shown but not
// clickable, so the visible spend still reconciles with the total-spend tile.
const OTHER_KEY = "__other__";
const UNKNOWN_KEY = "__unknown__";

function BreakdownTable({
  dimensionLabel,
  rows,
  totalCost,
  emptyLabel,
  unknownLabel = "(unknown)",
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
      header: dimensionLabel,
      isRowHeader: true,
      cell: (row) => {
        const share = totalCost > 0 ? row.cost / totalCost : 0;
        return (
          <div className="flex flex-col gap-1">
            <span className="truncate text-[var(--otari-ink)]">
              {row.is_other
                ? `Other (${row.requests.toLocaleString()} req)`
                : row.key === null
                  ? unknownLabel
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
    { id: "tokens", header: "Tokens", align: "end", cell: (row) => <span className="text-[var(--otari-muted)]">{formatTokens(row.tokens)}</span> },
    { id: "spend", header: "Spend", align: "end", cell: (row) => <span className="text-[var(--otari-ink)]">{formatUsd(row.cost)}</span> },
  ];

  return (
    <div className="flex flex-col gap-2">
      <DataTable
        ariaLabel={`Spend by ${dimensionLabel.toLowerCase()}`}
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

// Gateway-run tool spend. A separate table from BreakdownTable because the unit is
// different: a tool bills per call, and one request can run several, so "requests"
// and "tokens" would both misdescribe it. Failed calls are shown because they are
// counted and never billed, which is the first thing to check when a cost looks off.
function ToolBreakdownTable({
  rows,
  totalCost,
  onDrill,
  loading,
}: {
  rows: UsageSummary["by_tool"];
  totalCost: number;
  onDrill: (tool: string) => void;
  loading: boolean;
}) {
  const columns: DataTableColumn<UsageSummary["by_tool"][number]>[] = [
    {
      id: "tool",
      header: "Tool",
      isRowHeader: true,
      cell: (row) => {
        const share = totalCost > 0 ? row.cost / totalCost : 0;
        return (
          <div className="flex flex-col gap-1">
            <span className="truncate text-[var(--otari-ink)]">{row.tool.replaceAll("_", " ")}</span>
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
    { id: "calls", header: "Calls", align: "end", cell: (row) => <span className="text-[var(--otari-muted)]">{formatCount(row.calls)}</span> },
    {
      id: "failed",
      header: "Failed",
      align: "end",
      cell: (row) => (
        <span className={row.errors ? "text-red-700" : "text-[var(--otari-muted)]"}>{formatCount(row.errors)}</span>
      ),
    },
    { id: "requests", header: "Requests", align: "end", cell: (row) => <span className="text-[var(--otari-muted)]">{formatCount(row.requests)}</span> },
    { id: "spend", header: "Spend", align: "end", cell: (row) => <span className="text-[var(--otari-ink)]">{formatUsd(row.cost)}</span> },
  ];
  return (
    <DataTable
      ariaLabel="Spend by gateway-run tool"
      columns={columns}
      rows={rows}
      getRowKey={(row) => row.tool}
      isLoading={loading}
      emptyContent="No gateway-run tool calls in this range."
      onRowAction={(key) => onDrill(String(key))}
    />
  );
}

// ---------- which breakdowns the page asks for ----------

// Every breakdown the page renders, and nothing more (each one is a GROUP BY
// over the window server-side; tile-only queries pass NO_BREAKDOWNS instead). There is deliberately no API-key spend table:
// keys identify callers, not workloads, and the User table already answers
// "who". The chart's group-by can still split by key via /v1/usage/series.
const PAGE_BREAKDOWNS: SummaryDimension[] = [
  "model",
  "user",
  "source_label",
  "endpoint",
  "provider",
  "source",
  "tool",
];

// Filter-option suggestions come from one summary that drops the three entity
// filters, so each picker keeps offering every in-window value even while one is
// selected. by_user and by_api_key carry the entity's display name (resolved
// server-side in the same GROUP BY), which is what lets the pickers name their
// options without the page loading the users and api_keys tables in full.
const SUGGEST_BREAKDOWNS: SummaryDimension[] = ["model", "user", "api_key"];

// ---------- breakdown dimensions ----------

// One breakdown tab. Model and user are the two questions asked on every visit;
// the rest answer a real question each, so they share the same tab strip rather
// than stacking seven tables.
interface BreakdownDimensionDef {
  key: SummaryDimension;
  label: string;
  rows: UsageGroupRow[];
  // How a group whose column was NULL reads (see BreakdownTable.unknownLabel).
  unknownLabel?: string;
  drill: (key: string) => void;
}

// ---------- page ----------

export function UsagePage() {
  const navigate = useNavigate();

  const [preset, setPreset] = useState<RangePreset>(DEFAULT_PRESET);
  // Anchored start of the rolling preset window, snapshotted so a re-render does
  // not recompute "now" and churn the query key. Re-anchored on preset change
  // and on refresh. Usage presets are all bounded, so this is always set.
  const [startDate, setStartDate] = useState<string>(() => isoAgo(DEFAULT_PRESET.seconds ?? 0));
  // Custom range: an explicit UTC-bucket window selected by dragging across the
  // analytics chart. When active it overrides the preset window. The selection
  // always yields both bounds, so there is no half-filled window.
  const [customMode, setCustomMode] = useState(false);
  const [customStart, setCustomStart] = useState<string | undefined>();
  const [customEnd, setCustomEnd] = useState<string | undefined>();
  // Entity filters are sets, not single choices: the question this page answers is
  // usually a comparison ("these two models", "this team's three keys"). The
  // endpoints take each one repeatably and match any of its values.
  const [modelFilters, setModelFilters] = useState<string[]>([]);
  const [userFilters, setUserFilters] = useState<string[]>([]);
  const [apiKeyFilters, setApiKeyFilters] = useState<string[]>([]);
  const [metric, setMetric] = useState<ChartMetric>("cost");
  const [groupBy, setGroupBy] = useState<"" | UsageGroupBy>("");

  const winStart = customMode ? customStart : startDate;
  const winEnd = customMode ? customEnd : undefined;
  // Bucket by the window's length, not by whether it is custom, so a short
  // custom range still buckets hourly instead of collapsing to a single point.
  const bucket: UsageBucket = customMode
    ? winStart
      ? bucketForWindow(winStart, winEnd)
      : "day"
    : preset.bucket;

  const filters: UsageFilters = useMemo(
    () => ({
      start_date: winStart,
      end_date: winEnd,
      model: modelFilters.length > 0 ? modelFilters : undefined,
      user_id: userFilters.length > 0 ? userFilters : undefined,
      api_key_id: apiKeyFilters.length > 0 ? apiKeyFilters : undefined,
    }),
    [winStart, winEnd, modelFilters, userFilters, apiKeyFilters],
  );

  // The immediately-preceding window of equal length, for period-over-period
  // deltas. Every preset is bounded, so a comparison always exists.
  const previousFilters: UsageFilters | null = useMemo(() => {
    if (customMode) {
      if (!winStart || !winEnd) return null;
      const span = new Date(winEnd).getTime() - new Date(winStart).getTime();
      if (!(span > 0)) return null;
      return { ...filters, start_date: new Date(new Date(winStart).getTime() - span).toISOString(), end_date: winStart };
    }
    if (!startDate || preset.seconds === null) return null;
    return {
      ...filters,
      start_date: new Date(new Date(startDate).getTime() - preset.seconds * 1000).toISOString(),
      // Cap the previous window at the current window's start.
      end_date: startDate,
    };
  }, [customMode, winStart, winEnd, filters, preset.seconds, startDate]);

  const summary = useUsageSummary(filters, bucket, PAGE_BREAKDOWNS);
  // Deltas read `totals` only, so the previous window skips every breakdown.
  const previous = useUsageSummary(previousFilters ?? filters, bucket, NO_BREAKDOWNS, previousFilters !== null);
  // The per-group stack, fetched only while a dimension is selected.
  const grouped = useUsageGroupedSeries(filters, bucket, groupBy || null);
  // A 404 from the grouped endpoint is version skew: a gateway older than this
  // dashboard (most often one not yet restarted onto the build that ships it).
  // Fall back to the ungrouped view with a notice instead of a bare error.
  const groupingUnsupported =
    groupBy !== "" && grouped.error instanceof ApiError && grouped.error.status === 404;
  const effectiveGroupBy = groupingUnsupported ? "" : groupBy;

  const data = summary.data;
  const totals = data?.totals;
  const prevTotals = previousFilters !== null ? previous.data?.totals : undefined;
  const costDelta = totals ? deltaFraction(totals.cost, prevTotals?.cost) : null;

  // Model typeahead options: the in-window models. Sourced from a summary that
  // omits the model filter, so the list stays complete when a model is selected,
  // and derived directly from query data rather than mirrored into state.
  const suggestFilters: UsageFilters = useMemo(
    () => ({ ...filters, model: undefined, user_id: undefined, api_key_id: undefined }),
    [filters],
  );
  const suggest = useUsageSummary(suggestFilters, bucket, SUGGEST_BREAKDOWNS);
  const realGroups = (rows: UsageGroupRow[] | undefined) =>
    (rows ?? []).filter((r) => !r.is_other && r.key !== null);
  const modelOptions = realGroups(suggest.data?.by_model).map((r) => r.key as string);

  const userOptions = realGroups(suggest.data?.by_user).map((r) => ({
    value: r.key as string,
    label: r.label ? `${r.label} (${r.key})` : (r.key as string),
  }));
  // API key options label by name (falling back to a short id), value is the id.
  const keyOptions = realGroups(suggest.data?.by_api_key).map((r) => ({
    value: r.key as string,
    label: r.label ?? `${(r.key as string).slice(0, 8)}…`,
  }));
  // Just the in-window models: a picked one needs no place in this list, because
  // the picker hides what is already selected and the chips carry the raw name.
  const modelOptionList = modelOptions.map((m) => ({ value: m, label: m }));

  // The default 30d window is the baseline (like the old "All" was), so it does
  // not count as a user-applied time filter: clearing returns to it, and an
  // empty gateway on the default view still reads as onboarding, not "no match".
  const timeFiltered = customMode || preset.key !== USAGE_DEFAULT_KEY;
  const anyFilter =
    modelFilters.length > 0 || userFilters.length > 0 || apiKeyFilters.length > 0 || timeFiltered;

  // Active entity filters as removable chips, one per picked value (time is driven
  // by the presets and the chart selection, so it is not a chip). The chip row is
  // also where a value is removed: it stays visible when the pickers are collapsed.
  const labelFor = (options: { value: string; label: string }[], value: string) =>
    options.find((o) => o.value === value)?.label ?? value;
  const clearEntityFilters = () => {
    setModelFilters([]);
    setUserFilters([]);
    setApiKeyFilters([]);
  };
  const valueChips = (
    dimension: string,
    label: string,
    values: string[],
    display: (value: string) => string,
    setValues: (next: string[]) => void,
  ): FilterChip[] =>
    values.map((value) => ({
      key: `${dimension}:${value}`,
      label,
      value: display(value),
      // The value is part of the control's name: several chips share a dimension,
      // and "Remove User filter" three times over names none of them.
      clearLabel: `Remove ${label} filter ${display(value)}`,
      onClear: () => setValues(values.filter((v) => v !== value)),
    }));
  const filterChips: FilterChip[] = [
    ...valueChips("user", "User", userFilters, (v) => labelFor(userOptions, v), setUserFilters),
    ...valueChips("model", "Model", modelFilters, (v) => v, setModelFilters),
    ...valueChips("key", "API key", apiKeyFilters, (v) => labelFor(keyOptions, v), setApiKeyFilters),
  ];

  // Distinguish "this gateway has never served a request" from "no rows match
  // these filters": the first is an onboarding state, the second is a filter hint.
  const isEmptyEver = Boolean(data && totals && totals.request_count === 0 && !anyFilter);

  // The window the server actually aggregated over, so the caption reflects any
  // default or clamp. Falls back to the client-intended window before the first
  // response lands.
  const effectiveStart = data?.start_date ?? winStart;
  const effectiveEnd = data?.end_date ?? winEnd;

  const pickPreset = (next: RangePreset) => {
    setCustomMode(false);
    setPreset(next);
    setStartDate(isoAgo(next.seconds ?? 0));
    setCustomStart(undefined);
    setCustomEnd(undefined);
  };

  const pickCustom = (startIso: string, endIso: string) => {
    setCustomMode(true);
    setCustomStart(startIso);
    setCustomEnd(endIso);
  };

  // Refetch every window-scoped query, not just the headline summary: in custom
  // mode the query keys do not change, so anything left out would silently stay
  // on cached data after an explicit refresh. The conditional queries are
  // guarded because refetch() ignores `enabled` and would fire pointlessly.
  const refresh = () => {
    if (!customMode) {
      setStartDate(isoAgo(preset.seconds ?? 0));
    }
    void summary.refetch();
    void suggest.refetch();
    if (previousFilters !== null) {
      void previous.refetch();
    }
    if (groupBy) {
      void grouped.refetch();
    }
  };

  // Drill from a breakdown row into the Activity log, pre-filtering on the picked
  // dimension plus the current time window. The bounds are absolute instants, so
  // Activity reads the same window without any local/UTC reinterpretation. A
  // multi-value filter travels whole, as repeated params: Activity reads the same
  // sets, so the log opens on exactly the traffic the chart was showing.
  const drillTo = (params: Record<string, string | string[] | undefined>) => {
    const search = new URLSearchParams();
    if (winStart) search.set("start_date", winStart);
    if (winEnd) search.set("end_date", winEnd);
    for (const [key, value] of Object.entries(params)) {
      for (const one of typeof value === "string" ? [value] : (value ?? [])) {
        if (one) search.append(key, one);
      }
    }
    navigate(`/activity?${search.toString()}`);
  };

  const errorRate = totals && totals.request_count > 0 ? totals.error_count / totals.request_count : 0;

  // Provenance only earns UI (a group-by option, a breakdown tab) once the
  // window actually has more than one source: most gateways see only their own
  // traffic, and a single-option dimension is noise. Kept while it is the
  // active grouping so switching windows never strands the selection.
  const multiSource = (data?.by_source ?? []).filter((r) => !r.is_other).length > 1;
  const showSource = multiSource || groupBy === "source";

  // ---------- derived analytics ----------

  const series = data?.series ?? [];
  const hasTrend = series.length > 1;

  // Billed token view: input (incl. cache) + output, with the raw provider total
  // as the fallback when the composition fields are absent (an older gateway
  // behind `vite dev`). Cache hit rate = reads / billed input.
  const billedInput = totals?.billed_input_tokens;
  const billedTotal =
    totals === undefined
      ? null
      : billedInput !== undefined
        ? billedInput + (totals.billed_output_tokens ?? totals.completion_tokens)
        : totals.total_tokens;
  const prevBilledTotal =
    prevTotals === undefined
      ? null
      : prevTotals.billed_input_tokens !== undefined
        ? prevTotals.billed_input_tokens + (prevTotals.billed_output_tokens ?? prevTotals.completion_tokens)
        : prevTotals.total_tokens;
  // Cache sums from the series composition rather than the raw totals columns:
  // the raw sums follow each provider's reporting convention, while the series
  // is meter-normalized, and the tile's own sparkline reads the series. One
  // source keeps the headline, its trendline, and the hint in agreement.
  const cacheSums = (points: UsageSeriesPoint[]) => {
    let input = 0;
    let read = 0;
    let write = 0;
    for (const p of points) {
      input += p.input_tokens ?? 0;
      read += p.cache_read_tokens ?? 0;
      write += p.cache_write_tokens ?? 0;
    }
    return { input, read, write };
  };
  const cache = cacheSums(series);
  const cacheHitRate = cache.input > 0 ? cache.read / cache.input : null;
  const prevCache = cacheSums(previousFilters !== null ? (previous.data?.series ?? []) : []);
  const prevCacheHitRate = prevCache.input > 0 ? prevCache.read / prevCache.input : undefined;

  const pointBilled = (p: UsageSeriesPoint) =>
    p.input_tokens !== undefined ? p.input_tokens + (p.output_tokens ?? 0) : p.tokens;
  const hasComposition = series.some((p) => (p.input_tokens ?? 0) > 0);
  const hasErrors = series.some((p) => (p.errors ?? 0) > 0);

  // The main chart's series + data for the current metric × group-by. All the
  // pivoting happens here so the chart component stays dumb.
  const chart = useMemo((): { series: SeriesDef[]; data: StackedPoint[] } => {
    const buckets = series.map((p) => p.bucket_start);
    if (effectiveGroupBy) {
      const g = grouped.data;
      if (!g) return { series: [], data: [] };
      const defs = g.groups.map((row, index) => ({
        key: `g${index}`,
        label: row.is_other
          ? "Other"
          : row.key === null
            ? "(unknown)"
            : effectiveGroupBy === "api_key_id"
              ? (row.label ?? `${row.key.slice(0, 8)}…`)
              : row.key,
        color: row.is_other ? OTHER_COLOR : CAT_COLORS[index % CAT_COLORS.length],
      }));
      const seriesKey = new Map(g.groups.map((row, index) => [`${row.is_other}|${row.key}`, `g${index}`]));
      const byBucket = new Map<string, StackedPoint>(
        buckets.map((b) => [b, { x: b, ...Object.fromEntries(defs.map((d) => [d.key, 0])) }]),
      );
      for (const point of g.points) {
        const key = seriesKey.get(`${point.is_other}|${point.key}`);
        const row = byBucket.get(point.bucket_start);
        if (!key || !row) continue;
        row[key] = metric === "cost" ? point.cost : metric === "tokens" ? point.tokens : point.requests;
      }
      return { series: defs, data: [...byBucket.values()] };
    }
    if (metric === "tokens" && hasComposition) {
      return {
        series: COMPOSITION_SERIES,
        data: series.map((p) => {
          const input = p.input_tokens ?? 0;
          const read = p.cache_read_tokens ?? 0;
          const write = p.cache_write_tokens ?? 0;
          return {
            x: p.bucket_start,
            fresh: Math.max(0, input - read - write),
            cache_read: read,
            cache_write: write,
            output: p.output_tokens ?? 0,
          };
        }),
      };
    }
    if (metric === "requests" && hasErrors) {
      return {
        series: REQUEST_SERIES,
        data: series.map((p) => {
          const errors = Math.min(p.errors ?? 0, p.requests);
          return { x: p.bucket_start, success: p.requests - errors, errors };
        }),
      };
    }
    const single: SeriesDef = {
      key: metric,
      label: METRIC_TABS.find((t) => t.key === metric)?.label ?? metric,
      color: "var(--otari-brand)",
    };
    return {
      series: [single],
      data: series.map((p) => ({
        x: p.bucket_start,
        [metric]: metric === "cost" ? p.cost : metric === "tokens" ? pointBilled(p) : p.requests,
      })),
    };
    // Group labels now come from the server on `grouped.data`, so this memo has
    // no input outside its dependency list and needs no exhaustive-deps escape.
  }, [series, effectiveGroupBy, grouped.data, metric, hasComposition, hasErrors]);

  const formatValue = metricFormatter(metric);
  const chartLoading = summary.isLoading || (Boolean(effectiveGroupBy) && grouped.isLoading);
  const peak = chart.data.length
    ? Math.max(
        ...chart.data.map((row) => chart.series.reduce((sum, s) => sum + (typeof row[s.key] === "number" ? (row[s.key] as number) : 0), 0)),
      )
    : 0;

  // Dragging across the chart zooms into the selected buckets (the same
  // interaction as the Activity strip and every mainstream metrics tool).
  const chartBuckets = series.map((p) => p.bucket_start);
  const onChartSelect = (startIndex: number, endIndex: number) => {
    const range = rangeFromBuckets(chartBuckets, startIndex, endIndex, bucket);
    if (range) pickCustom(range.startIso, range.endIso);
  };

  // ---------- breakdown tabs ----------

  const dimensions: BreakdownDimensionDef[] = [
    {
      key: "model",
      label: "Model",
      rows: data?.by_model ?? [],
      drill: (key) => drillTo({ model: key, user_id: userFilters, api_key_id: apiKeyFilters }),
    },
    {
      key: "user",
      label: "User",
      rows: data?.by_user ?? [],
      drill: (key) =>
        drillTo({ user_id: key, model: modelFilters, api_key_id: apiKeyFilters }),
    },
  ];

  // The secondary breakdown answers "what ran", complementing the primary's
  // "on what / for whom". Sessions first: for agent traffic a few long-running
  // sessions carry most of the spend, so this is usually the row that explains
  // a bill.
  const secondaryDimensions: BreakdownDimensionDef[] = [
    {
      key: "source_label",
      label: "Session",
      rows: data?.by_source_label ?? [],
      unknownLabel: "(no session)",
      drill: (key) =>
        drillTo({ source_label: key, model: modelFilters, user_id: userFilters, api_key_id: apiKeyFilters }),
    },
    {
      key: "endpoint",
      label: "Endpoint",
      rows: data?.by_endpoint ?? [],
      drill: (key) =>
        drillTo({ endpoint: key, model: modelFilters, user_id: userFilters, api_key_id: apiKeyFilters }),
    },
    {
      key: "provider",
      label: "Provider",
      rows: data?.by_provider ?? [],
      drill: (key) =>
        drillTo({ provider: key, model: modelFilters, user_id: userFilters, api_key_id: apiKeyFilters }),
    },
    {
      key: "source",
      label: "Source",
      rows: data?.by_source ?? [],
      drill: (key) =>
        drillTo({
          source: key,
          model: modelFilters,
          user_id: userFilters,
          api_key_id: apiKeyFilters,
        }),
    },
  ];
  const toolRows = data?.by_tool ?? [];
  const [primaryDim, setPrimaryDim] = useState<SummaryDimension>("model");
  const [secondaryDim, setSecondaryDim] = useState<SummaryDimension>("source_label");
  const activePrimary = dimensions.find((d) => d.key === primaryDim) ?? dimensions[0];
  const visibleSecondary = secondaryDimensions.filter(
    (d) => d.key !== "source" || multiSource || secondaryDim === "source",
  );
  const activeSecondary = visibleSecondary.find((d) => d.key === secondaryDim) ?? visibleSecondary[0];

  return (
    <div className="flex flex-col gap-6">
      <PageHeader
        title="Usage & analytics"
        description="Spend, tokens, cache use, and request volume over time. Group the chart by model, user, key, or source, and click a breakdown row to drill into the request log."
      />

      <ErrorBanner error={summary.error ?? (groupBy !== "" && !groupingUnsupported ? grouped.error : null)} />

      {/* Window + filter row: presets anchor the rolling window (dragging on the
          chart below selects an explicit sub-window), the Add filter toggle and
          its chips share the same line, and the caption + refresh sit at the
          right edge. */}
      <FilterChips
        chips={filterChips}
        onClearAll={clearEntityFilters}
        start={USAGE_PRESETS.map((p) => (
          <Button
            key={p.key}
            size="sm"
            variant={!customMode && preset.key === p.key ? "primary" : "outline"}
            onPress={() => pickPreset(p)}
          >
            {p.label}
          </Button>
        ))}
        end={
          <>
            <span className="text-xs text-[var(--otari-muted)]">
              Showing {formatWindowLabel(effectiveStart, effectiveEnd)} · UTC
            </span>
            <RefreshButton onRefresh={refresh} isFetching={summary.isFetching} updatedAt={summary.dataUpdatedAt} />
          </>
        }
      >
        <FilterMultiComboBox
          label="User"
          values={userFilters}
          onChange={setUserFilters}
          options={userOptions}
          placeholder="All users"
        />
        <FilterMultiComboBox
          label="Model"
          values={modelFilters}
          onChange={setModelFilters}
          options={modelOptionList}
          placeholder="All models"
        />
        <FilterMultiComboBox
          label="API key"
          values={apiKeyFilters}
          onChange={setApiKeyFilters}
          options={keyOptions}
          placeholder="All keys"
        />
      </FilterChips>

      {isEmptyEver ? (
        <EmptyState
          title="No usage yet"
          description="Once the gateway serves requests, spend and volume appear here."
        />
      ) : (
        <>
          {/* KPI tiles. Cache tells one story (hit rate + volumes) instead of
              three raw counters; tokens are the billed total, matching the
              chart's composition and the Activity page. */}
          <div className="grid grid-cols-2 gap-4 sm:grid-cols-3 xl:grid-cols-5">
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
              label="Tokens (billed)"
              value={billedTotal !== null ? formatTokens(billedTotal) : "—"}
              hint={
                billedTotal !== null ? (
                  <DeltaHint fraction={deltaFraction(billedTotal, prevBilledTotal ?? undefined)} />
                ) : null
              }
              chart={
                hasTrend ? (
                  <Sparkline values={series.map(pointBilled)} ariaLabel="Billed token trend over the selected window" />
                ) : undefined
              }
            />
            <StatCard
              label="Cache hit rate"
              value={cacheHitRate !== null ? formatPct(cacheHitRate) : "—"}
              hint={
                totals ? (
                  <span className="text-[var(--otari-muted)]">
                    {cacheHitRate !== null && prevCacheHitRate !== undefined ? (
                      <>
                        <DeltaHint fraction={deltaFraction(cacheHitRate, prevCacheHitRate)} />
                        {" · "}
                      </>
                    ) : null}
                    {formatTokens(cache.read)} read · {formatTokens(cache.write)} written
                  </span>
                ) : null
              }
              chart={
                hasTrend && hasComposition ? (
                  <Sparkline
                    values={series.map((p) =>
                      (p.input_tokens ?? 0) > 0 ? (p.cache_read_tokens ?? 0) / (p.input_tokens ?? 1) : 0,
                    )}
                    ariaLabel="Cache hit rate trend over the selected window"
                  />
                ) : undefined
              }
            />
            <StatCard label="Avg latency" value={totals ? formatLatency(totals.avg_latency_ms) : "—"} />
          </div>

          {/* The analytics chart: metric × group-by, brushable. */}
          <div className="flex flex-col gap-3 rounded-xl border border-[var(--otari-line)] bg-[var(--otari-surface)] p-4">
            <div className="flex flex-wrap items-center justify-between gap-3">
              <div className="inline-flex gap-1.5">
                {METRIC_TABS.map((tab) => (
                  <Button
                    key={tab.key}
                    size="sm"
                    variant={metric === tab.key ? "primary" : "outline"}
                    aria-pressed={metric === tab.key}
                    onPress={() => setMetric(tab.key)}
                  >
                    {tab.label}
                  </Button>
                ))}
              </div>
              <div className="flex items-center gap-2">
                {customMode ? (
                  <Button size="sm" variant="ghost" onPress={() => pickPreset(preset)}>
                    Reset zoom
                  </Button>
                ) : null}
                {summary.isFetching || (effectiveGroupBy && grouped.isFetching) ? <Spinner size="sm" /> : null}
                <FilterSelect
                  ariaLabel="Group by"
                  value={groupBy}
                  onChange={(value) => setGroupBy(value as "" | UsageGroupBy)}
                  options={GROUP_OPTIONS.filter((o) => o.value !== "source" || showSource).map((o) => ({
                    value: o.value,
                    label: o.value ? `By ${o.label.toLowerCase()}` : "No grouping",
                  }))}
                />
              </div>
            </div>
            {groupingUnsupported ? (
              <div className="rounded-md border border-amber-200 bg-amber-50 px-3 py-2 text-xs text-amber-800">
                The running gateway predates grouped series, so the chart shows ungrouped totals. Restart the gateway
                on this build to enable grouping.
              </div>
            ) : null}
            <ChartLegend series={chart.series} />
            {chartLoading ? (
              <div className="flex h-64 items-center justify-center">
                <Spinner size="sm" />
              </div>
            ) : chart.data.length === 0 ? (
              <div className="flex h-64 items-center justify-center text-sm text-[var(--otari-muted)]">
                No data in this range.
              </div>
            ) : (
              <figure className="flex flex-col gap-2">
                <TrendChart
                  data={chart.data}
                  series={chart.series}
                  formatValue={formatValue}
                  formatXTick={(iso) => formatBucketLabel(iso, bucket)}
                  ariaLabel={`${metric} per ${bucket}${effectiveGroupBy ? `, grouped by ${effectiveGroupBy}` : ""}`}
                  height={260}
                  showYAxis
                  showTotal
                  onSelectRange={onChartSelect}
                />
                <figcaption className="text-xs text-[var(--otari-muted)]">
                  {formatValue(peak)} peak · {chart.data.length} {bucket === "hour" ? "hours" : "days"} (times in UTC) ·
                  drag across the chart to zoom
                </figcaption>
              </figure>
            )}
          </div>

          {/* Breakdowns: the answer to "where is my money going?". The primary
              table splits spend by who/what is billed (model, user); the
              secondary by what ran (session, endpoint, provider, source), with
              session first since it usually names the work behind a bill.
              Drilling keeps the other active filters, so the log stays scoped
              to them instead of showing every request for the group. */}
          <div className="grid gap-6 xl:grid-cols-2">
            <div className="flex flex-col gap-3">
              <div className="flex flex-wrap items-center justify-between gap-3">
                <h2 className="text-sm font-semibold text-[var(--otari-ink)]">Spend by {activePrimary.label.toLowerCase()}</h2>
                <div className="inline-flex gap-1.5">
                  {dimensions.map((d) => (
                    <Button
                      key={d.key}
                      size="sm"
                      variant={primaryDim === d.key ? "primary" : "outline"}
                      aria-pressed={primaryDim === d.key}
                      onPress={() => setPrimaryDim(d.key)}
                    >
                      {d.label}
                    </Button>
                  ))}
                </div>
              </div>
              <BreakdownTable
                dimensionLabel={activePrimary.label}
                rows={activePrimary.rows}
                totalCost={totals?.cost ?? 0}
                emptyLabel={anyFilter ? "No usage matches these filters." : "No usage recorded yet."}
                unknownLabel={activePrimary.unknownLabel}
                onDrill={activePrimary.drill}
                loading={summary.isLoading}
              />
            </div>
            <div className="flex flex-col gap-3">
              <div className="flex flex-wrap items-center justify-between gap-3">
                <h2 className="text-sm font-semibold text-[var(--otari-ink)]">Spend by {activeSecondary.label.toLowerCase()}</h2>
                <div className="inline-flex gap-1.5">
                  {visibleSecondary.map((d) => (
                    <Button
                      key={d.key}
                      size="sm"
                      variant={secondaryDim === d.key ? "primary" : "outline"}
                      aria-pressed={secondaryDim === d.key}
                      onPress={() => setSecondaryDim(d.key)}
                    >
                      {d.label}
                    </Button>
                  ))}
                </div>
              </div>
              <BreakdownTable
                dimensionLabel={activeSecondary.label}
                rows={activeSecondary.rows}
                totalCost={totals?.cost ?? 0}
                emptyLabel={anyFilter ? "No usage matches these filters." : "No usage recorded yet."}
                unknownLabel={activeSecondary.unknownLabel}
                onDrill={activeSecondary.drill}
                loading={summary.isLoading}
              />
            </div>
          </div>

          {/* Tools get their own card rather than a tab, and only when the window
              contains some: on a gateway that runs no tools it would be an empty
              table asking to be explained, and where it does appear it answers a
              question none of the other tables can ("what did search cost me"). */}
          {toolRows.length ? (
            <div className="rounded-2xl border border-[var(--otari-line)] bg-[var(--otari-surface)] p-4">
              <div className="mb-3 flex flex-col gap-1">
                <h2 className="text-sm font-semibold text-[var(--otari-ink)]">Gateway-run tools</h2>
                <p className="text-xs text-[var(--otari-muted)]">
                  Tools Otari ran itself, billed per call. MCP tools are not listed here: their names come from your
                  own server, so they appear on each request instead.
                </p>
              </div>
              <ToolBreakdownTable
                rows={toolRows}
                totalCost={totals?.cost ?? 0}
                onDrill={(tool) => drillTo({ tool: tool as NonNullable<UsageFilters["tool"]> })}
                loading={summary.isLoading}
              />
            </div>
          ) : null}

        </>
      )}
    </div>
  );
}
