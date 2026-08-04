import { Button } from "@heroui/react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { ReactNode } from "react";

import {
  NO_BREAKDOWNS,
  useDeleteUsage,
  useKeys,
  useSetUsagePrice,
  useUsageCount,
  useUsageLogs,
  useUsageSummary,
  useUsers,
} from "@/api/hooks";
import type { SummaryDimension, UsageEntry, UsageFilters, UsageMutationSelection } from "@/api/types";
import { ActivityTimeline } from "@/components/ActivityTimeline";
import { BulkActionBar } from "@/components/BulkActionBar";
import { ConfirmDialog } from "@/components/ConfirmDialog";
import { DataTable, type DataTableColumn } from "@/components/DataTable";
import { FilterChips, type FilterChip } from "@/components/FilterChips";
import { SetPriceDialog, type ManualRates } from "@/components/SetPriceDialog";
import { PAGE_SIZE_OPTIONS, TablePagination } from "@/components/TablePagination";
import { CopyableValue, ErrorBanner, FilterComboBox, FilterSelect, PageHeader, RefreshButton } from "@/components/ui";
import { resolveSelectedIds, useTableSelection } from "@/lib/tableSelection";
import {
  ACTIVITY_DEFAULT_KEY,
  ACTIVITY_PRESETS,
  bucketForWindow,
  CUSTOM_KEY,
  findPreset,
  isoAgo,
  type RangePreset,
  YEAR_SPAN_S,
} from "@/lib/timeRange";
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

// Stable row-key getter and row class so DataTable's per-row cache holds
// across re-renders (see the DataTable docstring); an inline arrow here would
// rebuild every row on each selection click.
const getActivityRowKey = (e: UsageEntry): string => e.id;

const activityRowClassName = (e: UsageEntry): string | undefined =>
  e.status === "error" ? "bg-red-50" : undefined;

// ---------- filter option sets ----------
//
// The time presets and window math are shared with the Usage page via
// `@/lib/timeRange` (see the ActivityTimeline selector). Activity keeps a
// truthful "All": its raw list endpoint applies no default and no clamp, so an
// omitted start really is all-time.

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

// The only breakdown this page reads: the in-window models behind the typeahead.
// The typeahead reads `by_model`; the source picker's option list piggybacks on
// the same query's `by_source` while no source is picked (see the source
// suggestion note below), so both breakdowns ride one request.
const MODEL_AND_SOURCE_BREAKDOWNS: SummaryDimension[] = ["model", "source"];
const SOURCE_BREAKDOWN: SummaryDimension[] = ["source"];

// All filter + pagination state, with defaults, kept in the URL.
const URL_DEFAULTS = {
  range: ACTIVITY_DEFAULT_KEY,
  start_date: "",
  end_date: "",
  status: "",
  model: "",
  user_id: "",
  api_key_id: "",
  priced: "",
  source: "",
  source_label: "",
  endpoint: "",
  provider: "",
  page: "0",
  size: String(DEFAULT_PAGE_SIZE),
} as const;

// Resolve the query window. Explicit start_date/end_date bounds (a custom range,
// or a drill-down from the Usage page) take precedence; otherwise a preset anchors
// `start` to "now minus N", and "all" (or an empty custom range) leaves it open.
function resolveWindow(range: string, start: string, end: string): { start?: string; end?: string } {
  if (start || end) {
    return { start: start || undefined, end: end || undefined };
  }
  if (range === CUSTOM_KEY) {
    return {};
  }
  const preset = findPreset(ACTIVITY_PRESETS, range) ?? findPreset(ACTIVITY_PRESETS, ACTIVITY_DEFAULT_KEY);
  const seconds = preset?.seconds ?? null;
  return { start: seconds == null ? undefined : isoAgo(seconds), end: undefined };
}

// The histogram extent (what the bars span), which is *not* always the list
// window. For bounded presets it matches `resolveWindow`. Any range with no
// rolling start of its own (the unbounded "All", or the `custom` sentinel) gets an
// explicit year-long start instead: the list genuinely omits its start there, but
// the summary endpoint would then apply a hidden 30-day default, so the bars would
// silently show a rolling month while the caption reads "All time". The explicit
// start gives a deterministic, draggable span (the axis shows exactly what it
// covers) while the list stays all-time.
function resolveExtentWindow(range: string): { start?: string; end?: string } {
  const win = resolveWindow(range, "", "");
  if (win.start) return win;
  const preset = findPreset(ACTIVITY_PRESETS, range);
  if (preset?.seconds == null) return { start: isoAgo(YEAR_SPAN_S) };
  return win;
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

// ---------- token composition ----------
//
// One total is the least useful number on the row: on a cached agent workload it
// is ~98% cache read, so every row shows a large, similar-looking figure. The
// composition is what varies, so the column renders the split.

interface TokenComposition {
  // Input tokens billed at the full input rate (the prompt minus whatever was
  // served from, or written to, the cache).
  fresh: number;
  cacheRead: number;
  cacheWrite: number;
  output: number;
  total: number;
}

function positive(value: unknown): number {
  return typeof value === "number" && Number.isFinite(value) && value > 0 ? value : 0;
}

// Split a row's tokens into fresh input / cache read / cache write / output, or
// null when the row carries no usage at all (an error before the provider replied).
//
// `billing_meters` is preferred because it is the *normalized* view: providers
// disagree on whether cache reads and writes are counted inside `prompt_tokens`
// (OpenAI: yes, Anthropic: no), and the row does not record which convention its
// numbers follow, so the raw columns alone cannot be split reliably. The writers
// resolve that when they price a row, so meters are present for any priced row.
// Failing that, assume the subset convention and clamp, which is exact whenever
// there is no cache usage to misattribute.
function tokenComposition(entry: UsageEntry): TokenComposition | null {
  const meters = entry.billing_meters ?? null;
  const totalInput = meters ? positive(meters.total_input_tokens) : positive(entry.prompt_tokens);
  const cacheRead = meters ? positive(meters.cache_read_tokens) : positive(entry.cache_read_tokens);
  const cacheWrite = meters ? positive(meters.cache_write_tokens) : positive(entry.cache_write_tokens);
  const output = meters ? positive(meters.completion_tokens) : positive(entry.completion_tokens);
  const fresh = Math.max(0, totalInput - cacheRead - cacheWrite);
  const total = fresh + cacheRead + cacheWrite + output;
  return total > 0 ? { fresh, cacheRead, cacheWrite, output, total } : null;
}

// Segment order runs input side first (fresh, then the two cache buckets), then
// output. Shading is one hue at four lightnesses, assigned for legibility rather
// than for price: every fill clears the track it sits on, adjacent fills differ
// enough to show their boundary, and the bucket that is usually the bulk (cache
// read) takes a mid tone instead of the palest step, so a cache-heavy row reads
// as a filled bar and a fresh-input row as a dark one. Nothing is encoded by hue,
// and the tooltip / accessible name carry every number, so the bar adds a shape
// to scan and removes no information.
const TOKEN_SEGMENTS: { key: keyof Omit<TokenComposition, "total">; label: string; fill: string }[] = [
  { key: "fresh", label: "Fresh input", fill: "var(--otari-ink)" },
  { key: "cacheRead", label: "Cache read", fill: "var(--otari-brand)" },
  { key: "cacheWrite", label: "Cache write", fill: "var(--otari-brand-soft)" },
  { key: "output", label: "Output", fill: "var(--otari-brand-dark)" },
];

// The total plus a thin stacked bar of its composition. Widths are SVG rect
// attributes in a 100-unit viewBox (a dynamic Tailwind `w-[n%]` would not survive
// the JIT scanner, and inline styles are out). Proportions are exact, so a segment
// under a percent or so lands sub-pixel; the tooltip carries the real numbers.
//
// The number shown is the sum of the segments, so the cell is internally
// consistent and reads as "tokens billed". For an additive-convention row that is
// higher than the raw `total_tokens` column (which excludes the cache buckets);
// the raw fields stay visible, unchanged, in the detail panel.
function TokenBar({ entry }: { entry: UsageEntry }) {
  const composition = tokenComposition(entry);
  if (composition === null) {
    return <span className="tabular-nums">{formatTokens(entry.total_tokens)}</span>;
  }
  const parts = TOKEN_SEGMENTS.map((segment) => ({ ...segment, value: composition[segment.key] }));
  const summary = parts
    .filter((part) => part.value > 0)
    .map((part) => `${part.label} ${part.value.toLocaleString()}`)
    .join(", ");

  let offset = 0;
  const rects = parts.map((part) => {
    const width = (part.value / composition.total) * 100;
    const rect = { ...part, x: offset, width };
    offset += width;
    return rect;
  });

  return (
    <span className="inline-flex flex-col items-end gap-1" title={summary}>
      <span className="tabular-nums">{composition.total.toLocaleString()}</span>
      <svg
        viewBox="0 0 100 4"
        preserveAspectRatio="none"
        role="img"
        aria-label={`Token composition: ${summary}`}
        className="h-1.5 w-20 overflow-hidden rounded-full bg-[var(--otari-brand-tint)]"
      >
        {rects
          .filter((rect) => rect.width > 0)
          .map((rect) => (
            <rect key={rect.key} x={rect.x} y={0} width={rect.width} height={4} fill={rect.fill} />
          ))}
      </svg>
    </span>
  );
}

// `copyValue` adds a copy control for the fields that hold an opaque identifier
// (a request id, an api key id): they are what an operator pastes into a log
// search or a support thread, and a mistyped character makes them useless.
function DetailField({
  label,
  copyValue,
  children,
}: {
  label: string;
  copyValue?: string | null;
  children: ReactNode;
}) {
  return (
    <div className="flex flex-col gap-0.5">
      <span className="text-[11px] font-medium uppercase tracking-wide text-[var(--otari-muted)]">{label}</span>
      {copyValue ? (
        <CopyableValue value={copyValue} label={label.toLowerCase()} className="text-sm text-[var(--otari-ink)] break-all">
          {children}
        </CopyableValue>
      ) : (
        <span className="text-sm text-[var(--otari-ink)] break-all">{children}</span>
      )}
    </div>
  );
}

// The detail panel for one request: a safe error summary plus the metadata that
// does not fit the row. Provider diagnostics stay server-side. The summary is
// deliberately source-neutral: failures here include requests the gateway itself
// refused (e.g. a model with no pricing under `require_pricing`), so it must not
// attribute every one of them to the provider.
function RequestDetail({ entry }: { entry: UsageEntry }) {
  return (
    <div className="flex flex-col gap-4 px-4 py-4">
      {entry.error_message ? (
        <div className="flex flex-col gap-1.5">
          <span className="text-[11px] font-medium uppercase tracking-wide text-[var(--otari-muted)]">Error</span>
          <pre className="max-h-48 overflow-auto rounded-lg border border-red-200 bg-red-50 p-3 text-xs whitespace-pre-wrap break-all text-red-700">
            This request failed. Inspect gateway logs for details.
          </pre>
        </div>
      ) : null}
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <DetailField label="Provider">{entry.provider ?? "—"}</DetailField>
        <DetailField label="Endpoint">{entry.endpoint}</DetailField>
        <DetailField label="Source">{sourceLabel(entry.source)}</DetailField>
        {entry.source_label ? <DetailField label="Session">{entry.source_label}</DetailField> : null}
        <DetailField label="User" copyValue={entry.user_id}>{entry.user_id ?? "—"}</DetailField>
        <DetailField label="API key" copyValue={entry.api_key_id}>{entry.api_key_id ?? "—"}</DetailField>
        <DetailField label="Prompt tokens">{formatTokens(entry.prompt_tokens)}</DetailField>
        <DetailField label="Completion tokens">{formatTokens(entry.completion_tokens)}</DetailField>
        <DetailField label="Total tokens">{formatTokens(entry.total_tokens)}</DetailField>
        {/* The Tokens column's number, spelled out here because it can exceed the
            provider-reported total above: the row's composition counts the cache
            buckets, which an additive-convention provider reports outside the prompt. */}
        <DetailField label="Billed tokens">
          <span title="Fresh input, cache reads and writes, and output: the tokens this request was priced on, and the total the activity row's bar splits.">
            {formatTokens(tokenComposition(entry)?.total ?? null)}
          </span>
        </DetailField>
        <DetailField label="Cost">{formatUSD(entry.cost)}</DetailField>
        <DetailField label="Cache read tokens">{formatTokens(entry.cache_read_tokens)}</DetailField>
        <DetailField label="Cache write tokens">{formatTokens(entry.cache_write_tokens)}</DetailField>
        <DetailField label="1h cache writes">{formatTokens(entry.cache_write_1h_tokens ?? null)}</DetailField>
        <DetailField label="Total time">{formatLatency(entry.latency_ms)}</DetailField>
        <DetailField label="Request ID" copyValue={entry.id}>{entry.id}</DetailField>
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
  // Provenance: gateway traffic vs an imported agent source. Set by its own select,
  // or by a drill-down (the pricing alarm links here scoped to gateway traffic).
  const sourceFilter = url.get("source");
  // The Usage-page secondary breakdowns (session / endpoint / provider) drill in
  // the same way: no select of their own, carried as a chip so the scoping is
  // visible and one click removes it.
  const sessionFilter = url.get("source_label");
  const endpointFilter = url.get("endpoint");
  const providerFilter = url.get("provider");
  const page = Math.max(0, url.getNumber("page"));
  // Snap URL-supplied sizes to the nearest offered option: selection latency
  // grows linearly with rows on the page, so an old bookmark with size=500
  // must not resurrect second-long checkbox clicks, and a hand-edited size=0
  // or size=-5 must not reach the API as an invalid limit (or leave the
  // rows-per-page select showing a value it does not offer).
  const rawPageSize = url.getNumber("size");
  const pageSize = PAGE_SIZE_OPTIONS.reduce((best, option) =>
    Math.abs(option - rawPageSize) < Math.abs(best - rawPageSize) ? option : best,
  );

  // Snapshot the window so a rolling preset does not recompute "now" every render
  // (which would churn the query key). Re-anchored when the range selection changes,
  // and by re-picking the active preset (see `pickPreset`).
  //
  // Both re-anchor effects skip their first run. The `useState` initializers have
  // already snapshotted the window, so re-anchoring on mount only moves `start` by
  // the milliseconds since, which changes `filters`, which trips the page reset
  // below: a bookmarked or shared `?page=3` URL would silently open on page 1.
  const selectionKey = `${range}|${startParam}|${endParam}`;
  const [win, setWin] = useState(() => resolveWindow(range, startParam, endParam));
  const prevSelectionKey = useRef(selectionKey);
  useEffect(() => {
    if (prevSelectionKey.current === selectionKey) return;
    prevSelectionKey.current = selectionKey;
    setWin(resolveWindow(range, startParam, endParam));
  }, [selectionKey, range, startParam, endParam]);

  // The preset extent (ignoring any brushed bounds) that the timeline histogram
  // spans. Snapshotted like `win`, re-anchored when the preset changes: a brushed
  // sub-window must leave the extent alone, or zooming in would drag the frame
  // (and refetch the histogram) along with it.
  const [extentWin, setExtentWin] = useState(() => resolveExtentWindow(range));
  const prevRange = useRef(range);
  useEffect(() => {
    if (prevRange.current === range) return;
    prevRange.current = range;
    setExtentWin(resolveExtentWindow(range));
  }, [range]);

  const priced = pricedFilter === "true" ? true : pricedFilter === "false" ? false : undefined;

  const filters: UsageFilters = useMemo(
    () => ({
      start_date: win.start,
      end_date: win.end,
      status: statusFilter || undefined,
      model: modelFilter.trim() || undefined,
      user_id: userFilter || undefined,
      api_key_id: apiKeyFilter || undefined,
      source: sourceFilter || undefined,
      source_label: sessionFilter || undefined,
      endpoint: endpointFilter || undefined,
      provider: providerFilter || undefined,
      priced,
    }),
    [
      win,
      statusFilter,
      modelFilter,
      userFilter,
      apiKeyFilter,
      sourceFilter,
      sessionFilter,
      endpointFilter,
      providerFilter,
      priced,
    ],
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
      source: sourceFilter || undefined,
      source_label: sessionFilter || undefined,
      endpoint: endpointFilter || undefined,
      provider: providerFilter || undefined,
    }),
    [win, statusFilter, userFilter, apiKeyFilter, sourceFilter, sessionFilter, endpointFilter, providerFilter],
  );
  // Only `by_model` and `by_source` are read (typeahead + source picker), so
  // only those breakdowns are requested: no use for the other five GROUP BYs.
  const modelSummary = useUsageSummary(modelSuggestFilters, "day", MODEL_AND_SOURCE_BREAKDOWNS);
  const modelOptions =
    modelSummary.data?.by_model?.filter((r) => !r.is_other && r.key !== null).map((r) => r.key as string) ?? [];
  const keyOptions = (keys.data ?? []).map((k) => ({ value: k.id, label: k.key_name ?? `${k.id.slice(0, 8)}…` }));

  // Source options: the sources with usage in the window. Like the model
  // suggestions, this must ignore the source filter itself, or picking Claude Code
  // would hide every other source and switching would mean clearing first.
  //
  // While no source is picked the model-suggestion summary is already computed
  // without one, so its provenance breakdown is that full list and no second query
  // is needed. Only a picked source needs its own, so it is fetched then and only
  // then (each summary runs four grouped aggregations plus the series).
  // Deliberately broader than modelSuggestFilters: the session/endpoint/provider
  // drill-down chips are not applied here, so the source list stays complete
  // (and the picker stays useful) while a drill-down narrows everything else.
  const sourceSuggestFilters: UsageFilters = useMemo(
    () => ({
      start_date: win.start,
      end_date: win.end,
      status: statusFilter || undefined,
      model: modelFilter.trim() || undefined,
      user_id: userFilter || undefined,
      api_key_id: apiKeyFilter || undefined,
    }),
    [win, statusFilter, modelFilter, userFilter, apiKeyFilter],
  );
  const sourceSummary = useUsageSummary(sourceSuggestFilters, "day", SOURCE_BREAKDOWN, Boolean(sourceFilter));
  const sourceBreakdown = (sourceFilter ? sourceSummary.data : modelSummary.data)?.by_source;
  // A drill-down can name a source with no rows in the window; keep it listed so
  // the select shows the filter that is actually applied.
  const sourceOptions = useMemo(() => {
    const seen = (sourceBreakdown ?? []).filter((r) => !r.is_other && r.key !== null).map((r) => r.key as string);
    return sourceFilter && !seen.includes(sourceFilter) ? [sourceFilter, ...seen] : seen;
  }, [sourceBreakdown, sourceFilter]);

  // The timeline histogram spans the whole preset *extent* (the rolling preset
  // window, independent of any brushed sub-window), so the brush always has
  // context to zoom back out into. For the unbounded "All", `extentWin` carries an
  // explicit year-long start (see `resolveExtentWindow`) so the bars span a
  // deterministic window instead of the summary endpoint's hidden 30-day default;
  // the list stays all-time and the caption reflects the true list window, and the
  // brush still narrows it. Entity filters carry over so the bars match what's shown.
  const extentPreset = findPreset(ACTIVITY_PRESETS, range) ?? findPreset(ACTIVITY_PRESETS, ACTIVITY_DEFAULT_KEY);
  // A window reaching outside the preset extent (a drill-down from the Usage
  // page carries its own bounds while the URL's `range` still holds a default)
  // cannot be framed by that extent: the histogram would show unrelated bars
  // and the preset would read as active. Frame the window itself instead, with
  // no preset highlighted; zoom-out falls back to the smallest broader preset.
  const winOutsideExtent = Boolean(
    win.start && extentWin.start && new Date(win.start).getTime() < new Date(extentWin.start).getTime(),
  );
  const extentKey = winOutsideExtent ? CUSTOM_KEY : range;
  const extentBucket = winOutsideExtent
    ? bucketForWindow(win.start as string, win.end)
    : (extentPreset?.bucket ?? "day");
  const contextFilters: UsageFilters = useMemo(
    () => ({
      start_date: winOutsideExtent ? win.start : extentWin.start,
      end_date: winOutsideExtent ? win.end : undefined,
      status: statusFilter || undefined,
      model: modelFilter.trim() || undefined,
      user_id: userFilter || undefined,
      api_key_id: apiKeyFilter || undefined,
      source: sourceFilter || undefined,
      source_label: sessionFilter || undefined,
      endpoint: endpointFilter || undefined,
      provider: providerFilter || undefined,
      priced,
    }),
    [
      winOutsideExtent,
      win,
      extentWin,
      statusFilter,
      modelFilter,
      userFilter,
      apiKeyFilter,
      sourceFilter,
      sessionFilter,
      endpointFilter,
      providerFilter,
      priced,
    ],
  );
  // The timeline reads `series` only.
  const contextSummary = useUsageSummary(contextFilters, extentBucket, NO_BREAKDOWNS);
  const timelineSeries = (contextSummary.data?.series ?? []).map((p) => ({
    bucketStart: p.bucket_start,
    requests: p.requests,
    // Failed requests render as a red segment on the strip, so dropped traffic
    // shows up while browsing, not only after filtering to status=error.
    errors: p.errors ?? 0,
  }));

  const rows = usage.data ?? [];
  const totalIsExact = count.isSuccess && !count.isPlaceholderData;
  const total = totalIsExact ? (count.data?.total ?? 0) : null;
  // Neither the default preset nor the unbounded "All" is itself a filter: only an
  // explicit sub-window or a bounded non-default preset narrows the window, so a
  // brand-new gateway reads "never used" on both its 24h default and on "All",
  // and only a real narrowing reads "filtered empty". (UsagePage can mirror this
  // with a bare `!== default` because it has no unbounded preset; Activity does.)
  const rangePreset = findPreset(ACTIVITY_PRESETS, range);
  const timeFiltered =
    Boolean(startParam || endParam) || (range !== ACTIVITY_DEFAULT_KEY && rangePreset?.seconds != null);
  const anyFilter = Boolean(
    statusFilter ||
      modelFilter.trim() ||
      userFilter ||
      apiKeyFilter ||
      pricedFilter ||
      sourceFilter ||
      sessionFilter ||
      endpointFilter ||
      providerFilter ||
      timeFiltered,
  );

  // Active entity filters as removable chips (time is driven by the timeline, so
  // it is not a chip). Values show the human label where one exists.
  const labelFrom = (options: { value: string; label: string }[], value: string) =>
    options.find((o) => o.value === value)?.label ?? value;
  const userOptionsList = (users.data ?? []).map((u) => ({
    value: u.user_id,
    label: u.alias ? `${u.alias} (${u.user_id})` : u.user_id,
  }));
  const clearEntityFilters = () =>
    url.patch({
      status: "",
      priced: "",
      model: "",
      user_id: "",
      api_key_id: "",
      source: "",
      source_label: "",
      endpoint: "",
      provider: "",
    });
  const filterChips: FilterChip[] = [
    ...(statusFilter ? [{ key: "status", label: "Status", value: labelFrom(STATUS_OPTIONS, statusFilter), onClear: () => url.patch({ status: "" }) }] : []),
    ...(pricedFilter ? [{ key: "priced", label: "Priced", value: labelFrom(PRICED_OPTIONS, pricedFilter), onClear: () => url.patch({ priced: "" }) }] : []),
    ...(userFilter ? [{ key: "user", label: "User", value: labelFrom(userOptionsList, userFilter), onClear: () => url.patch({ user_id: "" }) }] : []),
    ...(modelFilter.trim() ? [{ key: "model", label: "Model", value: modelFilter.trim(), onClear: () => url.patch({ model: "" }) }] : []),
    ...(apiKeyFilter ? [{ key: "key", label: "API key", value: labelFrom(keyOptions, apiKeyFilter), onClear: () => url.patch({ api_key_id: "" }) }] : []),
    ...(sourceFilter ? [{ key: "source", label: "Source", value: sourceLabel(sourceFilter), onClear: () => url.patch({ source: "" }) }] : []),
    ...(sessionFilter ? [{ key: "session", label: "Session", value: sessionFilter, onClear: () => url.patch({ source_label: "" }) }] : []),
    ...(endpointFilter ? [{ key: "endpoint", label: "Endpoint", value: endpointFilter, onClear: () => url.patch({ endpoint: "" }) }] : []),
    ...(providerFilter ? [{ key: "provider", label: "Provider", value: providerFilter, onClear: () => url.patch({ provider: "" }) }] : []),
  ];

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

  // Inline accordion panel under the clicked row (DataTable renderDetail).
  // Stable (setter-only closure) so the row cache holds; see the DataTable
  // docstring.
  const renderDetail = useCallback(
    (entry: UsageEntry) => (
      <div>
        <div className="flex items-center justify-between border-b border-[var(--otari-line)] px-4 py-2">
          <span className="text-sm font-medium text-[var(--otari-ink)]">Request detail</span>
          <Button size="sm" variant="ghost" onPress={() => setExpandedId(null)}>
            Close
          </Button>
        </div>
        <RequestDetail entry={entry} />
      </div>
    ),
    [],
  );

  // A bulk op targets either the current page selection (ids) or, once the operator
  // opted into "all matching", the filter itself (by_filter). The server scopes
  // either to imported rows.
  //
  // Every filter that scopes the table has to be forwarded here. "All matching" is
  // counted client-side from the full filter set but re-derived server-side from
  // this body, so any filter left out widens the delete/reprice past the rows the
  // operator was shown (a session drill-down would wipe every other session).
  const selectionBody = (): UsageMutationSelection =>
    selection.allMatching
      ? {
          by_filter: true,
          model: filters.model,
          user_id: filters.user_id,
          api_key_id: filters.api_key_id,
          status: filters.status,
          source: filters.source,
          source_label: filters.source_label,
          endpoint: filters.endpoint,
          provider: filters.provider,
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

  // Refresh means "same view, newer rows", so it deliberately does *not* re-anchor
  // a rolling preset's window. Re-anchoring recomputed "now", which changed
  // `filters`, which tripped the page reset below: pressing refresh on page 12
  // dropped the operator back onto page 1, making deep browsing unusable. The
  // window is instead re-anchored when the range selection changes (see
  // `pickPreset`). Every window-scoped query is refetched explicitly (see the
  // UsagePage refresh note), since the keys are unchanged by design here.
  const refresh = () => {
    void usage.refetch();
    void count.refetch();
    void contextSummary.refetch();
    void modelSummary.refetch();
    // Guarded because refetch() ignores `enabled`: without a picked source the
    // query is disabled by design and refetching it would fire a pointless
    // extra summary request.
    if (sourceFilter) {
      void sourceSummary.refetch();
    }
  };

  // A rolling preset clears any explicit bounds; a timeline selection sets them
  // (the preset key is left as-is, since it still names the extent).
  const pickPreset = (preset: RangePreset) => {
    // Re-picking the preset that is already active is the explicit "re-anchor to
    // now" gesture. It leaves the URL untouched, so the effect that snapshots the
    // window never fires; do it here instead. (Going *to* a different preset, or
    // off an explicit range, changes the URL and that effect handles it, so
    // re-anchoring here as well would fire a second query for the same view.)
    if (preset.key === range && !startParam && !endParam) {
      setWin(resolveWindow(preset.key, "", ""));
      setExtentWin(resolveExtentWindow(preset.key));
      return;
    }
    url.patch({ range: preset.key, start_date: "", end_date: "" });
  };
  const pickCustom = (startIso: string, endIso: string) => url.patch({ start_date: startIso, end_date: endIso });

  // Memoized on keyLabels (the only per-render input) so DataTable's per-row
  // cache holds: a fresh array every render would rebuild all rows per click.
  const columns = useMemo<DataTableColumn<UsageEntry>[]>(() => {
    const apiKeyLabel = (id: string | null): string =>
      id === null ? "—" : (keyLabels.get(id) ?? `${id.slice(0, 8)}…`);
    return [
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
      { id: "tokens", header: "Tokens", align: "end", cell: (e) => <TokenBar entry={e} /> },
      { id: "cost", header: "Cost", align: "end", cell: (e) => formatUSD(e.cost) },
      { id: "latency", header: "Total time", align: "end", cell: (e) => formatLatency(e.latency_ms) },
      { id: "status", header: "Status", cell: (e) => <StatusPill status={e.status} /> },
    ];
  }, [keyLabels]);

  return (
    <div className="flex flex-col gap-6">
      <PageHeader
        title="Activity"
        description="A per-request log of what the gateway served: tokens, cost, latency, and failures. No request or response content is stored."
      />

      {/* The timeline's summary error is included so a failed series request
          reads as a failure, not as an empty "No activity in this range" strip. */}
      <ErrorBanner error={usage.error ?? count.error ?? contextSummary.error} />

      <div className="flex flex-col gap-3">
        <ActivityTimeline
          presets={ACTIVITY_PRESETS}
          extentKey={extentKey}
          onPreset={pickPreset}
          onSelectRange={pickCustom}
          onSelectFull={() => (extentPreset ? pickPreset(extentPreset) : undefined)}
          series={timelineSeries}
          bucket={extentBucket}
          windowStart={win.start}
          windowEnd={win.end}
          loading={contextSummary.isLoading}
          ariaLabel="Activity request volume over the selected window"
          action={<RefreshButton onRefresh={refresh} isFetching={usage.isFetching} updatedAt={usage.dataUpdatedAt} />}
        />
        <FilterChips chips={filterChips} onClearAll={clearEntityFilters}>
          <FilterSelect id="filter-status" label="Status" value={statusFilter} onChange={(value) => url.patch({ status: value })}>
            {STATUS_OPTIONS.map((opt) => (
              <option key={opt.value} value={opt.value}>
                {opt.label}
              </option>
            ))}
          </FilterSelect>
          <FilterSelect id="filter-priced" label="Priced?" value={pricedFilter} onChange={(value) => url.patch({ priced: value })}>
            {PRICED_OPTIONS.map((opt) => (
              <option key={opt.value} value={opt.value}>
                {opt.label}
              </option>
            ))}
          </FilterSelect>
          {/* Provenance only earns a select once there is more than one source
              to choose between: most gateways see only their own traffic, and a
              filter with a single option is noise. A drill-down that arrives
              with a source applied keeps the select so it stays clearable. */}
          {sourceOptions.length > 1 || sourceFilter ? (
            <FilterSelect id="filter-source" label="Source" value={sourceFilter} onChange={(value) => url.patch({ source: value })}>
              <option value="">All</option>
              {sourceOptions.map((source) => (
                <option key={source} value={source}>
                  {sourceLabel(source)}
                </option>
              ))}
            </FilterSelect>
          ) : null}
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
            options={userOptionsList}
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
        </FilterChips>
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
        getRowKey={getActivityRowKey}
        isLoading={usage.isLoading}
        emptyContent={anyFilter ? "No requests match these filters." : "No requests recorded yet."}
        selectionMode="multiple"
        selectedKeys={selection.selectedKeys}
        onSelectionChange={selection.onSelectionChange}
        disabledKeys={disabledKeys}
        onRowAction={(key) => setExpandedId((current) => (current === key ? null : key))}
        rowClassName={activityRowClassName}
        detailKey={expandedId}
        renderDetail={renderDetail}
      />

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
