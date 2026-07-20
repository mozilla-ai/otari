import { Button, Spinner } from "@heroui/react";
import { Fragment, useEffect, useMemo, useState } from "react";
import type { ReactNode } from "react";

import { useUsageCount, useUsageLogs, useUsers } from "@/api/hooks";
import type { UsageEntry, UsageFilters } from "@/api/types";
import { LoadingRow, Table, TableMessage, Td, Th, THead, Tr } from "@/components/Table";
import { ErrorBanner, PageHeader } from "@/components/ui";

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

// Every endpoint that writes a usage_logs row, so the filter can reach all of
// them. A curated list keeps this a simple select without a separate "distinct
// endpoints" query; it must be extended when a new billable route is added.
const ENDPOINT_OPTIONS = [
  "/v1/chat/completions",
  "/v1/messages",
  "/v1/responses",
  "/v1/embeddings",
  "/v1/moderations",
  "/v1/audio/transcriptions",
  "/v1/audio/speech",
  "/v1/images/generations",
  "/v1/rerank",
  "/v1/batches",
  "/v1/batches/results",
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

function CopyButton({ text }: { text: string }) {
  const [copied, setCopied] = useState(false);
  return (
    <Button
      size="sm"
      variant="ghost"
      onPress={() => {
        void navigator.clipboard?.writeText(text);
        setCopied(true);
        window.setTimeout(() => setCopied(false), 1500);
      }}
    >
      {copied ? "Copied" : "Copy"}
    </Button>
  );
}

function DetailField({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div className="flex flex-col gap-0.5">
      <span className="text-[11px] font-medium uppercase tracking-wide text-[var(--otari-muted)]">{label}</span>
      <span className="text-sm text-[var(--otari-ink)] break-all">{children}</span>
    </div>
  );
}

// The expandable detail panel for one request: the full error text (copyable)
// plus the metadata that does not fit the row.
function RequestDetail({ entry }: { entry: UsageEntry }) {
  return (
    <div className="flex flex-col gap-4 px-4 py-4">
      {entry.error_message ? (
        <div className="flex flex-col gap-1.5">
          <div className="flex items-center justify-between">
            <span className="text-[11px] font-medium uppercase tracking-wide text-[var(--otari-muted)]">
              Error (provider-reported)
            </span>
            <CopyButton text={entry.error_message} />
          </div>
          <pre className="max-h-48 overflow-auto rounded-lg border border-red-200 bg-red-50 p-3 text-xs whitespace-pre-wrap break-all text-red-700">
            {entry.error_message}
          </pre>
        </div>
      ) : null}
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <DetailField label="Provider">{entry.provider ?? "—"}</DetailField>
        <DetailField label="Endpoint">{entry.endpoint}</DetailField>
        <DetailField label="User">{entry.user_id ?? "—"}</DetailField>
        <DetailField label="API key">{entry.api_key_id ?? "—"}</DetailField>
        <DetailField label="Prompt tokens">{formatTokens(entry.prompt_tokens)}</DetailField>
        <DetailField label="Completion tokens">{formatTokens(entry.completion_tokens)}</DetailField>
        <DetailField label="Total tokens">{formatTokens(entry.total_tokens)}</DetailField>
        <DetailField label="Cost">{formatUSD(entry.cost)}</DetailField>
        <DetailField label="Cache read tokens">{formatTokens(entry.cache_read_tokens)}</DetailField>
        <DetailField label="Cache write tokens">{formatTokens(entry.cache_write_tokens)}</DetailField>
        <DetailField label="Total time">{formatLatency(entry.latency_ms)}</DetailField>
        <DetailField label="Request ID">{entry.id}</DetailField>
      </div>
    </div>
  );
}

// A token-styled select matching the app's other filter controls.
function FilterSelect({
  id,
  label,
  value,
  onChange,
  children,
}: {
  id: string;
  label: string;
  value: string;
  onChange: (value: string) => void;
  children: ReactNode;
}) {
  return (
    <div className="flex flex-col gap-1">
      <label htmlFor={id} className="text-xs font-medium text-[var(--otari-muted)]">
        {label}
      </label>
      <select
        id={id}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="rounded-lg border border-[var(--otari-line)] bg-[var(--otari-bg)] px-3 py-2 text-sm text-[var(--otari-ink)]"
      >
        {children}
      </select>
    </div>
  );
}

// ---------- page ----------

const COLS = 7;

export function ActivityPage() {
  const users = useUsers();

  const [rangeSeconds, setRangeSeconds] = useState<number | null>(DAY_S);
  // Snapshotted when a range is picked (and on refresh), so the query key is
  // stable across renders instead of recomputing "now" every render.
  const [startDate, setStartDate] = useState<string | undefined>(() => isoAgo(DAY_S));
  const [statusFilter, setStatusFilter] = useState("");
  const [modelFilter, setModelFilter] = useState("");
  const [endpointFilter, setEndpointFilter] = useState("");
  const [userFilter, setUserFilter] = useState("");
  const [page, setPage] = useState(0);
  const [expanded, setExpanded] = useState<string | null>(null);

  const filters: UsageFilters = useMemo(
    () => ({
      start_date: startDate,
      status: statusFilter || undefined,
      model: modelFilter.trim() || undefined,
      endpoint: endpointFilter || undefined,
      user_id: userFilter || undefined,
    }),
    [startDate, statusFilter, modelFilter, endpointFilter, userFilter],
  );

  // Any change to the filter set returns to the first page.
  useEffect(() => {
    setPage(0);
  }, [filters]);

  const usage = useUsageLogs(filters, page, PAGE_SIZE);
  const count = useUsageCount(filters);

  const rows = usage.data ?? [];
  const total = count.data?.total ?? 0;
  const anyFilter = Boolean(
    statusFilter || modelFilter.trim() || endpointFilter || userFilter || rangeSeconds !== null,
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
    setEndpointFilter("");
    setUserFilter("");
  };

  const refresh = () => {
    // Re-anchor a rolling window to "now" before refetching.
    if (rangeSeconds !== null) {
      setStartDate(isoAgo(rangeSeconds));
    }
    void usage.refetch();
    void count.refetch();
  };

  const rangeStart = total === 0 ? 0 : page * PAGE_SIZE + 1;
  const rangeEnd = page * PAGE_SIZE + rows.length;
  // Prefer the exact total, but fall back to "a full page came back, so there is
  // probably more" when the count request failed. Otherwise a failed count would
  // strand the operator on page 1 with rows they cannot reach.
  const hasNext = count.data ? (page + 1) * PAGE_SIZE < total : rows.length === PAGE_SIZE;

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

      {/* Filters */}
      <div className="flex flex-col gap-3">
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
          <FilterSelect id="filter-endpoint" label="Endpoint" value={endpointFilter} onChange={setEndpointFilter}>
            <option value="">All endpoints</option>
            {ENDPOINT_OPTIONS.map((ep) => (
              <option key={ep} value={ep}>
                {ep}
              </option>
            ))}
          </FilterSelect>
          <FilterSelect id="filter-user" label="User" value={userFilter} onChange={setUserFilter}>
            <option value="">All users</option>
            {(users.data ?? []).map((u) => (
              <option key={u.user_id} value={u.user_id}>
                {u.alias ?? u.user_id}
              </option>
            ))}
          </FilterSelect>
          <div className="flex flex-col gap-1">
            <label htmlFor="filter-model" className="text-xs font-medium text-[var(--otari-muted)]">
              Model
            </label>
            <input
              id="filter-model"
              value={modelFilter}
              onChange={(e) => setModelFilter(e.target.value)}
              placeholder="e.g. gpt-4o"
              className="rounded-lg border border-[var(--otari-line)] bg-[var(--otari-bg)] px-3 py-2 text-sm text-[var(--otari-ink)]"
            />
          </div>
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
          {total === 0 ? "0 of 0" : `${rangeStart}–${rangeEnd} of ${total.toLocaleString()}`}
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
