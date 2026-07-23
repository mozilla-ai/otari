import { Button, Spinner } from "@heroui/react";
import { Fragment, useEffect, useMemo, useState } from "react";
import type { ReactNode } from "react";
import { useSearchParams } from "react-router-dom";

import { useKeys, useUsageCount, useUsageLogs, useUsageSummary, useUsers } from "@/api/hooks";
import type { UsageEntry, UsageFilters } from "@/api/types";
import { LoadingRow, Table, TableMessage, Td, Th, THead, Tr } from "@/components/Table";
import { ErrorBanner, FilterComboBox, FilterSelect, PageHeader } from "@/components/ui";

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

const TIME_PRESETS: { label: string; seconds: number | null }[] = [
  { label: "Last hour", seconds: HOUR_S },
  { label: "24h", seconds: DAY_S },
  { label: "7d", seconds: 7 * DAY_S },
  { label: "All", seconds: null },
];

const STATUS_OPTIONS: { label: string; value: string }[] = [
  { label: "All", value: "" },
  { label: "Success", value: "success" },
  { label: "Error", value: "error" },
];

const PAGE_SIZE = 50;

function isoAgo(seconds: number): string {
  return new Date(Date.now() - seconds * 1000).toISOString();
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

// The expandable detail panel for one request: a safe error summary plus the
// metadata that does not fit the row. Provider diagnostics stay server-side.
function RequestDetail({ entry }: { entry: UsageEntry }) {
  return (
    <div className="flex flex-col gap-4 px-4 py-4">
      {entry.error_message ? (
        <div className="flex flex-col gap-1.5">
          <div className="flex items-center justify-between">
            <span className="text-[11px] font-medium uppercase tracking-wide text-[var(--otari-muted)]">
              Error
            </span>
          </div>
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
          <span className="text-[11px] font-medium uppercase tracking-wide text-[var(--otari-muted)]">Billed meters</span>
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

const COLS = 8;

export function ActivityPage() {
  const users = useUsers();
  const keys = useKeys();
  // Map an api_key_id to a human label (the key's name, else a short id). Imported
  // usage carries the importing key; a null id is a master-key or historical row.
  const keyLabels = useMemo(() => {
    const map = new Map<string, string>();
    for (const k of keys.data ?? []) map.set(k.id, k.key_name ?? `${k.id.slice(0, 8)}…`);
    return map;
  }, [keys.data]);
  const apiKeyLabel = (id: string | null): string => (id === null ? "—" : (keyLabels.get(id) ?? `${id.slice(0, 8)}…`));
  // Drill-down from the Usage page arrives with model / user_id / start_date /
  // status in the query string; seed the initial filter state from it (once, on
  // mount) so the log opens pre-filtered on what the operator clicked.
  const [searchParams] = useSearchParams();
  const initialStart = searchParams.get("start_date") ?? undefined;

  const [rangeSeconds, setRangeSeconds] = useState<number | null>(initialStart ? null : DAY_S);
  // Snapshotted when a range is picked (and on refresh), so the query key is
  // stable across renders instead of recomputing "now" every render.
  const [startDate, setStartDate] = useState<string | undefined>(() => initialStart ?? isoAgo(DAY_S));
  const [statusFilter, setStatusFilter] = useState(() => searchParams.get("status") ?? "");
  const [modelFilter, setModelFilter] = useState(() => searchParams.get("model") ?? "");
  const [userFilter, setUserFilter] = useState(() => searchParams.get("user_id") ?? "");
  const [apiKeyFilter, setApiKeyFilter] = useState(() => searchParams.get("api_key_id") ?? "");
  const [page, setPage] = useState(0);
  const [expanded, setExpanded] = useState<string | null>(null);

  const filters: UsageFilters = useMemo(
    () => ({
      start_date: startDate,
      status: statusFilter || undefined,
      model: modelFilter.trim() || undefined,
      user_id: userFilter || undefined,
      api_key_id: apiKeyFilter || undefined,
    }),
    [startDate, statusFilter, modelFilter, userFilter, apiKeyFilter],
  );

  // Any change to the filter set returns to the first page.
  useEffect(() => {
    setPage(0);
  }, [filters]);

  const usage = useUsageLogs(filters, page, PAGE_SIZE);
  const count = useUsageCount(filters);

  // Model suggestions: models with usage in the current window (the other filters
  // applied, the model filter itself omitted so the full list stays offered). The
  // picker still accepts any typed model, so one outside this window is reachable.
  const modelSuggestFilters: UsageFilters = useMemo(
    () => ({
      start_date: startDate,
      status: statusFilter || undefined,
      user_id: userFilter || undefined,
      api_key_id: apiKeyFilter || undefined,
    }),
    [startDate, statusFilter, userFilter, apiKeyFilter],
  );
  const modelSummary = useUsageSummary(modelSuggestFilters, "day");
  // Derived from query data, not mirrored into state. modelSuggestFilters omits
  // the model filter, so the list is always the full set of in-window models.
  const modelOptions =
    modelSummary.data?.by_model?.filter((r) => !r.is_other && r.key !== null).map((r) => r.key as string) ?? [];
  // API-key options for the filter: every key's human name (or a short id), so an
  // operator can scope the log to one key. Imported OTLP usage carries its
  // importer key, so this reaches those rows too.
  const keyOptions = (keys.data ?? []).map((k) => ({
    value: k.id,
    label: k.key_name ?? `${k.id.slice(0, 8)}…`,
  }));

  const rows = usage.data ?? [];
  const total = count.data?.total ?? 0;
  const anyFilter = Boolean(
    statusFilter || modelFilter.trim() || userFilter || apiKeyFilter || rangeSeconds !== null,
  );

  const pickRange = (seconds: number | null) => {
    setRangeSeconds(seconds);
    setStartDate(seconds === null ? undefined : isoAgo(seconds));
  };

  const clearFilters = () => {
    setRangeSeconds(null);
    setStartDate(undefined);
    setStatusFilter("");
    setModelFilter("");
    setUserFilter("");
    setApiKeyFilter("");
  };

  const refresh = () => {
    // Re-anchor a rolling window to "now" before refetching.
    if (rangeSeconds !== null) {
      setStartDate(isoAgo(rangeSeconds));
    }
    void usage.refetch();
    void count.refetch();
  };

  // Anchored on the rows actually on screen, not the total, so the range stays
  // truthful when the count request failed and `total` is therefore 0.
  const hasRows = rows.length > 0;
  const rangeStart = hasRows ? page * PAGE_SIZE + 1 : 0;
  const rangeEnd = page * PAGE_SIZE + rows.length;
  // The total is only exact when the newest count for *these* filters succeeded.
  // `count.data` alone is not that test: TanStack keeps the last successful data
  // when a refetch errors, and keepPreviousData serves the previous filters'
  // total while a new one loads. Either would drive the pager off a stale number.
  const totalIsExact = count.isSuccess && !count.isPlaceholderData;
  // Without an exact total, show just the visible range: "0 of 0" would
  // contradict the rows on screen.
  const pagerLabel = !totalIsExact
    ? hasRows
      ? `${rangeStart}–${rangeEnd}`
      : "0"
    : total === 0
      ? "0 of 0"
      : `${rangeStart}–${rangeEnd} of ${total.toLocaleString()}`;
  // Fall back to "a full page came back, so there is probably more" when the
  // total is not exact. Otherwise a failed count would strand the operator on
  // page 1 with rows they cannot reach.
  const hasNext = totalIsExact ? (page + 1) * PAGE_SIZE < total : rows.length === PAGE_SIZE;

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

      {/* Filters. The time-range presets share a row with the filter boxes when the
          window is wide enough, and wrap onto their own line when it is not. */}
      <div className="flex flex-wrap items-end gap-3">
        <div className="flex flex-wrap gap-2">
          {TIME_PRESETS.map((preset) => (
            <Button
              key={preset.label}
              size="sm"
              variant={rangeSeconds === preset.seconds ? "primary" : "outline"}
              onPress={() => pickRange(preset.seconds)}
            >
              {preset.label}
            </Button>
          ))}
        </div>
        <div className="flex flex-wrap items-end gap-3">
          <FilterSelect id="filter-status" label="Status" value={statusFilter} onChange={setStatusFilter}>
            {STATUS_OPTIONS.map((opt) => (
              <option key={opt.value} value={opt.value}>
                {opt.label}
              </option>
            ))}
          </FilterSelect>
          <FilterComboBox
            label="API key"
            value={apiKeyFilter}
            onChange={setApiKeyFilter}
            placeholder="All keys"
            options={keyOptions}
          />
          <FilterComboBox
            label="User"
            value={userFilter}
            onChange={setUserFilter}
            placeholder="All users"
            options={(users.data ?? []).map((u) => ({
              value: u.user_id,
              label: u.alias ? `${u.alias} (${u.user_id})` : u.user_id,
            }))}
          />
          <FilterComboBox
            label="Model"
            value={modelFilter}
            onChange={setModelFilter}
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

      <Table>
        <THead>
          <tr>
            <Th>Time</Th>
            <Th>User</Th>
            <Th>Model</Th>
            <Th>API key</Th>
            <Th className="text-right">Tokens</Th>
            <Th className="text-right">Cost</Th>
            <Th className="text-right">Total time</Th>
            <Th>Status</Th>
          </tr>
        </THead>
        <tbody>
          {usage.isLoading ? (
            <LoadingRow colSpan={COLS} />
          ) : rows.length === 0 ? (
            <TableMessage colSpan={COLS}>
              {anyFilter ? "No requests match these filters." : "No requests recorded yet."}
            </TableMessage>
          ) : (
            rows.map((entry) => {
              const isError = entry.status === "error";
              const isOpen = expanded === entry.id;
              return (
                <Fragment key={entry.id}>
                  <Tr
                    selected={isOpen}
                    className={isError ? "bg-red-50" : ""}
                    onClick={() => setExpanded((current) => (current === entry.id ? null : entry.id))}
                  >
                    <Td className="text-[var(--otari-muted)]">
                      <span title={absolute(entry.timestamp)}>{timeAgo(entry.timestamp)}</span>
                    </Td>
                    <Td className="text-[var(--otari-ink)]">{entry.user_id ?? "—"}</Td>
                    <Td className="text-[var(--otari-ink)]">{entry.model}</Td>
                    <Td className="text-[var(--otari-muted)]">{apiKeyLabel(entry.api_key_id)}</Td>
                    <Td className="text-right tabular-nums">{formatTokens(entry.total_tokens)}</Td>
                    <Td className="text-right tabular-nums">{formatUSD(entry.cost)}</Td>
                    <Td className="text-right tabular-nums">{formatLatency(entry.latency_ms)}</Td>
                    <Td>
                      <StatusPill status={entry.status} />
                    </Td>
                  </Tr>
                  {isOpen ? (
                    <tr className="border-b border-[var(--otari-line)] bg-[var(--otari-bg)]">
                      <td colSpan={COLS}>
                        <RequestDetail entry={entry} />
                      </td>
                    </tr>
                  ) : null}
                </Fragment>
              );
            })
          )}
        </tbody>
      </Table>

      {/* Pager */}
      <div className="flex items-center justify-between gap-3">
        <span className="inline-flex items-center gap-2 text-sm text-[var(--otari-muted)]">
          {pagerLabel}
          {usage.isFetching ? <Spinner size="sm" /> : null}
        </span>
        <span className="inline-flex gap-1.5">
          <Button size="sm" variant="outline" isDisabled={page === 0} onPress={() => setPage((p) => Math.max(0, p - 1))}>
            Previous
          </Button>
          <Button size="sm" variant="outline" isDisabled={!hasNext} onPress={() => setPage((p) => p + 1)}>
            Next
          </Button>
        </span>
      </div>
    </div>
  );
}
