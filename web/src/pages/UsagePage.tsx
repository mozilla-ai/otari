import { Button, Spinner } from "@heroui/react";
import { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";

import { apiFetchBlob } from "@/api/client";
import { useUsageSummary, usageSummaryCsvUrl, useUsers } from "@/api/hooks";
import type { UsageBucket, UsageFilters, UsageGroupRow, UsageSeriesPoint } from "@/api/types";
import { LoadingRow, Table, TableMessage, Td, Th, THead, Tr } from "@/components/Table";
import { ErrorBanner, FilterComboBox, PageHeader, StatCard } from "@/components/ui";

// ---------- formatting ----------

const usdCompact = new Intl.NumberFormat(undefined, { style: "currency", currency: "USD", maximumFractionDigits: 2 });

// Aggregate totals are four+ figures, so cents (not the per-request 4 decimals the
// Activity page uses) keeps the tiles readable.
function formatUSD(value: number): string {
  return usdCompact.format(value);
}

function formatCount(value: number): string {
  return value.toLocaleString();
}

// Compact token counts: 12.4M / 84.2k / 512.
function formatTokens(value: number): string {
  if (value >= 1_000_000) return `${(value / 1_000_000).toFixed(1)}M`;
  if (value >= 1_000) return `${(value / 1_000).toFixed(1)}k`;
  return String(value);
}

function formatLatency(ms: number | null): string {
  if (ms === null) return "—";
  if (ms < 1000) return `${Math.round(ms)} ms`;
  return `${(ms / 1000).toFixed(2)} s`;
}

function formatPct(fraction: number): string {
  return `${(fraction * 100).toFixed(1)}%`;
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

// ---------- delta ----------

// Period-over-period change. null when there is no comparable previous value
// (unbounded range, or a previous value of zero which would divide by zero).
function deltaFraction(current: number, previous: number | undefined): number | null {
  if (previous === undefined || previous === 0) return null;
  return (current - previous) / previous;
}

function DeltaHint({ fraction }: { fraction: number | null }) {
  if (fraction === null) return null;
  const arrow = fraction > 0 ? "▲" : fraction < 0 ? "▼" : "•";
  return (
    <span className="text-[var(--otari-muted)]">
      {arrow} {formatPct(Math.abs(fraction))} vs prev
    </span>
  );
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
function BreakdownTable({ title, rows, totalCost, emptyLabel, onDrill, loading }: BreakdownProps) {
  const [showAll, setShowAll] = useState(false);
  const visible = showAll ? rows : rows.slice(0, TABLE_TOP_N);
  const hidden = rows.length - visible.length;

  return (
    <div className="flex flex-col gap-2">
      <h2 className="text-sm font-semibold text-[var(--otari-ink)]">{title}</h2>
      <Table>
        <THead>
          <tr>
            <Th>{title.replace("Spend by ", "")}</Th>
            <Th className="text-right">Requests</Th>
            <Th className="text-right">Spend</Th>
          </tr>
        </THead>
        <tbody>
          {loading ? (
            <LoadingRow colSpan={3} />
          ) : rows.length === 0 ? (
            <TableMessage colSpan={3}>{emptyLabel}</TableMessage>
          ) : (
            visible.map((row, index) => {
              const isOther = row.key === null;
              const share = totalCost > 0 ? row.cost / totalCost : 0;
              return (
                <Tr
                  // "other" has a null key; index keeps the row key stable and unique.
                  key={row.key ?? `__other_${index}`}
                  onClick={isOther ? undefined : () => onDrill(row.key as string)}
                >
                  <Td className="text-[var(--otari-ink)]">
                    <div className="flex flex-col gap-1">
                      <span className="truncate">
                        {isOther ? `Other (${row.requests.toLocaleString()} req)` : row.key}
                      </span>
                      {/* Share-of-total bar. Width is data-driven, so it rides an
                          inline style like the Table's computed column widths. */}
                      <span className="h-1 w-full overflow-hidden rounded-full bg-[var(--otari-line)]">
                        <span
                          className="block h-full rounded-full bg-[var(--otari-brand)]"
                          style={{ width: `${Math.min(100, share * 100)}%` }}
                        />
                      </span>
                    </div>
                  </Td>
                  <Td className="text-right tabular-nums text-[var(--otari-muted)]">{formatCount(row.requests)}</Td>
                  <Td className="text-right tabular-nums text-[var(--otari-ink)]">{formatUSD(row.cost)}</Td>
                </Tr>
              );
            })
          )}
        </tbody>
      </Table>
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
  return metric === "cost" ? formatUSD(value) : metric === "tokens" ? formatTokens(value) : formatCount(value);
}

function formatBucketLabel(iso: string, bucket: UsageBucket): string {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  if (bucket === "hour") {
    return d.toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit", timeZone: "UTC" });
  }
  return d.toLocaleDateString(undefined, { month: "short", day: "numeric", timeZone: "UTC" });
}

// A single-series bar chart of the selected metric. Hand-rolled SVG (the dashboard
// keeps zero chart dependencies); the breakdown tables above are the accessible
// data table, and each bar carries a native <title> for hover/screen-reader. One
// series means one color, so nothing is encoded by hue alone.
function UsageChart({
  series,
  metric,
  bucket,
}: {
  series: UsageSeriesPoint[];
  metric: ChartMetric;
  bucket: UsageBucket;
}) {
  const width = 720;
  const height = 200;
  const pad = 8;
  const max = Math.max(1, ...series.map((p) => metricValue(p, metric)));
  const n = series.length;
  const slot = n > 0 ? (width - pad * 2) / n : 0;
  const barW = Math.max(1, slot * 0.7);

  // Label a handful of x positions (first / middle / last) so ticks never collide.
  // Dedupe so a short series (where middle == an endpoint) doesn't repeat a key.
  const labelIdx = n <= 1 ? [0] : [...new Set([0, Math.floor(n / 2), n - 1])];

  return (
    <figure className="flex flex-col gap-2">
      <svg
        viewBox={`0 0 ${width} ${height + 24}`}
        preserveAspectRatio="none"
        role="img"
        className="w-full"
        aria-label={`${metric} over time`}
      >
        <title>{`${metric} per ${bucket}`}</title>
        {series.map((point, i) => {
          const value = metricValue(point, metric);
          const h = (value / max) * (height - pad);
          const x = pad + i * slot + (slot - barW) / 2;
          const y = height - h;
          return (
            <rect key={point.bucket_start} x={x} y={y} width={barW} height={h} rx={1.5} className="fill-[var(--otari-brand)]">
              <title>{`${formatBucketLabel(point.bucket_start, bucket)}: ${formatMetric(value, metric)}`}</title>
            </rect>
          );
        })}
        {labelIdx.map((i) =>
          series[i] ? (
            <text
              key={`lbl_${i}`}
              x={pad + i * slot + slot / 2}
              y={height + 16}
              textAnchor="middle"
              className="fill-[var(--otari-muted)] text-[10px]"
            >
              {formatBucketLabel(series[i].bucket_start, bucket)}
            </text>
          ) : null,
        )}
      </svg>
      <figcaption className="text-xs text-[var(--otari-muted)]">
        {formatMetric(max, metric)} peak · {n} {bucket === "hour" ? "hours" : "days"} (times in UTC)
      </figcaption>
    </figure>
  );
}

// ---------- page ----------

export function UsagePage() {
  const navigate = useNavigate();
  const users = useUsers();

  const [preset, setPreset] = useState<Preset>(DEFAULT_PRESET);
  const [startDate, setStartDate] = useState<string | undefined>(() =>
    DEFAULT_PRESET.seconds === null ? undefined : isoAgo(DEFAULT_PRESET.seconds),
  );
  const [modelFilter, setModelFilter] = useState("");
  const [userFilter, setUserFilter] = useState("");
  const [metric, setMetric] = useState<ChartMetric>("cost");
  const [exporting, setExporting] = useState(false);
  const [exportError, setExportError] = useState<string | null>(null);

  const filters: UsageFilters = useMemo(
    () => ({
      start_date: startDate,
      model: modelFilter.trim() || undefined,
      user_id: userFilter || undefined,
    }),
    [startDate, modelFilter, userFilter],
  );

  // The immediately-preceding window of equal length, for period-over-period
  // deltas. Only meaningful for a bounded range.
  const previousFilters: UsageFilters | null = useMemo(() => {
    if (preset.seconds === null || !startDate) return null;
    return {
      ...filters,
      start_date: new Date(new Date(startDate).getTime() - preset.seconds * 1000).toISOString(),
      // Cap the previous window at the current window's start.
      end_date: startDate,
    };
  }, [filters, preset.seconds, startDate]);

  const summary = useUsageSummary(filters, preset.bucket);
  const previous = useUsageSummary(previousFilters ?? filters, preset.bucket, previousFilters !== null);

  const data = summary.data;
  const totals = data?.totals;
  const prevTotals = previousFilters !== null ? previous.data?.totals : undefined;

  // Options for the model typeahead: the models that actually have usage in the
  // current window. Captured only while no model filter is applied (a filtered
  // summary collapses by_model to the single selected model); the last full list
  // is retained so you can switch models without clearing first.
  const [modelOptions, setModelOptions] = useState<string[]>([]);
  useEffect(() => {
    if (!modelFilter && data?.by_model) {
      setModelOptions(data.by_model.filter((r) => r.key !== null).map((r) => r.key as string));
    }
  }, [modelFilter, data]);

  const userOptions = (users.data ?? []).map((u) => ({
    value: u.user_id,
    label: u.alias ? `${u.alias} (${u.user_id})` : u.user_id,
  }));
  // Keep a selected-but-not-in-list model visible (e.g. seeded from elsewhere).
  const modelOptionList = (
    modelFilter && !modelOptions.includes(modelFilter) ? [modelFilter, ...modelOptions] : modelOptions
  ).map((m) => ({ value: m, label: m }));

  const anyFilter = Boolean(modelFilter.trim() || userFilter || preset.seconds !== null);
  // Distinguish "this gateway has never served a request" from "no rows match
  // these filters": the first is an onboarding state, the second is a filter hint.
  const isEmptyEver = Boolean(data && totals && totals.request_count === 0 && !anyFilter);

  useEffect(() => {
    setExportError(null);
  }, [filters]);

  const pickPreset = (next: Preset) => {
    setPreset(next);
    setStartDate(next.seconds === null ? undefined : isoAgo(next.seconds));
  };

  const clearFilters = () => {
    pickPreset(TIME_PRESETS[TIME_PRESETS.length - 1]); // All
    setModelFilter("");
    setUserFilter("");
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

  const exportCsv = async () => {
    setExporting(true);
    setExportError(null);
    try {
      const blob = await apiFetchBlob(usageSummaryCsvUrl(filters));
      const url = URL.createObjectURL(blob);
      const anchor = document.createElement("a");
      anchor.href = url;
      anchor.download = "usage-summary.csv";
      anchor.click();
      URL.revokeObjectURL(url);
    } catch (error) {
      setExportError(error instanceof Error ? error.message : "Export failed.");
    } finally {
      setExporting(false);
    }
  };

  const errorRate = totals && totals.request_count > 0 ? totals.error_count / totals.request_count : 0;

  return (
    <div className="flex flex-col gap-6">
      <PageHeader
        title="Usage & analytics"
        description="Aggregate spend, tokens, and request volume over time. Click a model or user to drill into the request log."
        action={
          <>
            <Button variant="outline" onPress={refresh} isDisabled={summary.isFetching}>
              Refresh
            </Button>
            <Button
              variant="outline"
              onPress={() => void exportCsv()}
              isDisabled={exporting || !totals || totals.request_count === 0}
            >
              {exporting ? "Exporting…" : "Export CSV"}
            </Button>
          </>
        }
      />

      <ErrorBanner error={summary.error ?? (exportError ? new Error(exportError) : null)} />

      {/* Filters */}
      <div className="flex flex-wrap items-end gap-3">
        <div className="flex flex-wrap gap-2">
          {TIME_PRESETS.map((option) => (
            <Button
              key={option.label}
              size="sm"
              variant={preset.label === option.label ? "primary" : "outline"}
              onPress={() => pickPreset(option)}
            >
              {option.label}
            </Button>
          ))}
        </div>
        <div className="flex flex-wrap items-end gap-3">
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
          {/* Tiles */}
          <div className="flex flex-wrap gap-4">
            <StatCard
              label="Spend"
              value={totals ? formatUSD(totals.cost) : "—"}
              hint={<DeltaHint fraction={deltaFraction(totals?.cost ?? 0, prevTotals?.cost)} />}
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
            />
            <StatCard
              label="Tokens"
              value={totals ? formatTokens(totals.total_tokens) : "—"}
              hint={<DeltaHint fraction={deltaFraction(totals?.total_tokens ?? 0, prevTotals?.total_tokens)} />}
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
              onDrill={(key) => drillTo({ model: key })}
              loading={summary.isLoading}
            />
            <BreakdownTable
              title="Spend by user"
              rows={data?.by_user ?? []}
              totalCost={totals?.cost ?? 0}
              emptyLabel={anyFilter ? "No usage matches these filters." : "No usage recorded yet."}
              onDrill={(key) => drillTo({ user_id: key })}
              loading={summary.isLoading}
            />
          </div>

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
            ) : (data?.series.length ?? 0) === 0 ? (
              <div className="flex h-48 items-center justify-center text-sm text-[var(--otari-muted)]">
                No data in this range.
              </div>
            ) : (
              <UsageChart series={data?.series ?? []} metric={metric} bucket={preset.bucket} />
            )}
          </div>
        </>
      )}
    </div>
  );
}
