import { Button } from "@heroui/react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { ReactNode } from "react";

import {
  useDeleteUsage,
  useInFlightRequests,
  useSetPricing,
  useSetUsagePrice,
  useRequestGroups,
  useUsageCount,
  useUsageLogs,
  useUsageSummary,
} from "@/api/hooks";
import type {
  InFlightRequest,
  SummaryDimension,
  UsageEntry,
  UsageFilters,
  UsageGroupRow,
  UsageMutationSelection,
} from "@/api/types";
import { ActivityTimeline } from "@/components/ActivityTimeline";
import { BulkActionBar } from "@/components/BulkActionBar";
import { ConfirmDialog } from "@/components/ConfirmDialog";
import { DataTable, type DataTableColumn } from "@/components/DataTable";
import { FilterChips, type FilterChip } from "@/components/FilterChips";
import { SetPriceDialog, type ManualRates } from "@/components/SetPriceDialog";
import { PAGE_SIZE_OPTIONS, TablePagination } from "@/components/TablePagination";
import {
  CopyableValue,
  ErrorBanner,
  FilterMultiComboBox,
  FilterSelect,
  PageHeader,
  RefreshButton,
} from "@/components/ui";
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

// A per-call rate, unlike a per-million-token one, is routinely smaller than the
// 4 decimal places `usd` keeps: $0.00002 per search would render as "$0.0000" and
// read as free. Fall back to significant digits once the value is below what the
// currency format can show, so a real rate is never displayed as zero.
const usdPrecise = new Intl.NumberFormat(undefined, {
  style: "currency",
  currency: "USD",
  maximumSignificantDigits: 3,
});

function formatUnitRate(value: number): string {
  if (value === 0) return usd.format(0);
  return value < 0.0001 ? usdPrecise.format(value) : usd.format(value);
}

// A charge line is discriminated by which rate it carries: token meters price per
// million, gateway-run tool meters price per call.
type ChargeLine = NonNullable<UsageEntry["pricing_breakdown"]>[number];

// Token lines first, tool lines after, each group keeping the order the writers
// emitted. "Billed meters" otherwise reads as an unordered mix once a row has both.
function sortedBreakdown(lines: readonly ChargeLine[]): ChargeLine[] {
  return [...lines].sort((a, b) => Number("unit_rate" in a) - Number("unit_rate" in b));
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

// ---------- requests in flight ----------
//
// A usage row is written when a request settles, so the log alone can only
// describe the past: on a slow backend a 30-second call is invisible for its whole
// duration. A tracked request is therefore rendered as a row in this same table,
// pinned above the settled rows and carrying the status below, so that when it
// lands the row resolves in place into the success or error row it became. The
// synthetic row is shaped as a `UsageEntry` so the table's columns, widths,
// selection, and detail machinery need no separate code path: the outcome fields
// are null, which the shared formatters already render as their own placeholder.
const IN_FLIGHT_STATUS = "in progress";

// Browser-local timestamp the wait counts up from, present only on a synthetic
// row. Derived from the poll's `dataUpdatedAt` minus the server's own `elapsed_ms`,
// so the wait never depends on the browser clock agreeing with the gateway's.
type ActivityRow = UsageEntry & { inFlightStartedAtMs?: number };

const isInFlightRow = (row: ActivityRow): boolean => row.status === IN_FLIGHT_STATUS;

function inFlightRow(request: InFlightRequest, startedAtMs: number): ActivityRow {
  return {
    inFlightStartedAtMs: startedAtMs,
    id: request.id,
    user_id: request.user_id,
    api_key_id: request.api_key_id,
    timestamp: request.started_at,
    model: request.model,
    provider: request.provider,
    endpoint: request.endpoint,
    prompt_tokens: null,
    completion_tokens: null,
    total_tokens: null,
    cache_read_tokens: null,
    cache_write_tokens: null,
    cost: null,
    status: IN_FLIGHT_STATUS,
    error_message: null,
    status_code: null,
    policy_name: request.policy_name,
    latency_ms: null,
    source: "gateway",
    source_label: null,
    // Never selectable: bulk delete and reprice target logged rows, and this one
    // does not exist in the table yet. `true` is what the page's existing
    // disabled-key rule keys on, so no extra case is needed there.
    counts_toward_budget: true,
  };
}

// Coarser than the settled Total time column: this is a wall-clock wait an
// operator is watching rather than a measurement, so sub-second precision is
// noise. Minutes appear because a stuck local model is the case this exists for.
function formatElapsed(ms: number): string {
  const seconds = Math.floor(ms / 1000);
  if (seconds < 60) return `${seconds}s`;
  const minutes = Math.floor(seconds / 60);
  return `${minutes}m ${String(seconds % 60).padStart(2, "0")}s`;
}

// The wait, ticking between the 2s polls: on the one row whose whole point is that
// it has not finished, a number that only moved when a response landed would read
// as stalled. Each live row owns its interval so the `columns` array stays
// referentially stable and DataTable's per-row cache is not rebuilt every second
// (see the DataTable docstring).
function InFlightWait({ startedAtMs }: { startedAtMs: number }) {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    const timer = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(timer);
  }, []);
  // Never negative: a poll can resolve between the tick and the paint.
  return <span className="tabular-nums">{formatElapsed(Math.max(0, now - startedAtMs))}</span>;
}

// Stable row-key getter and row class so DataTable's per-row cache holds
// across re-renders (see the DataTable docstring); an inline arrow here would
// rebuild every row on each selection click.
const getActivityRowKey = (e: ActivityRow): string => e.id;

// An absorbed attempt is a failure a routing policy recovered from, so the
// request it belongs to succeeded. Styling it like an error would make a working
// fallback chain read as an outage, which is the same reason the server keeps it
// out of error_count. Amber says "something happened here" without saying "this
// request failed".
const activityRowClassName = (e: ActivityRow): string | undefined => {
  if (isInFlightRow(e)) return "bg-[var(--otari-brand-tint)]";
  if (e.status === "error") return "bg-red-50";
  if (e.status === "absorbed") return "bg-amber-50";
  return undefined;
};

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
  // An attempt a routing policy recovered from. Listed because the rows are
  // rendered and styled distinctly, so an operator who spots one has to be able
  // to filter to the rest of them.
  { label: "Absorbed", value: "absorbed" },
];

const PRICED_OPTIONS: { label: string; value: string }[] = [
  { label: "All", value: "" },
  { label: "Priced", value: "true" },
  { label: "Unpriced", value: "false" },
];

// Gateway-run tools an operator can filter on. "Any tool" also matches MCP tools,
// whose names come from the caller's own server and so cannot be enumerated here.
const TOOL_OPTIONS: { label: string; value: string }[] = [
  { label: "All", value: "" },
  { label: "Any tool", value: "any" },
  { label: "Web search", value: "web_search" },
  { label: "Code execution", value: "code_execution" },
];

// The only breakdown this page asks the summary for: whether the window contains
// gateway-run tool calls, which decides if the Tool filter is worth offering.
const TOOL_BREAKDOWN: SummaryDimension[] = ["tool"];

const DEFAULT_PAGE_SIZE = 50;

// Floor on how often a settled in-flight request may pull the log again. Matches
// the log's own `staleTime` in `useUsageLogs`, so this never delays a refresh the
// query would have served from cache anyway.
const SETTLED_REFETCH_MIN_MS = 10_000;

// The only breakdown this page reads: the in-window models behind the typeahead.
// The typeahead reads `by_model`; the source picker's option list piggybacks on
// the same query's `by_source` while no source is picked (see the source
// suggestion note below), so both breakdowns ride one request.
const MODEL_AND_SOURCE_BREAKDOWNS: SummaryDimension[] = ["model", "source"];

// The user and key pickers read these two. by_user and by_api_key carry each
// entity's display name, resolved server-side in the same GROUP BY, so naming an
// option costs nothing beyond the breakdown itself. The alternative, and what
// this replaced, was paging the whole users and api_keys tables on every visit.
const ENTITY_BREAKDOWNS: SummaryDimension[] = ["user", "api_key"];
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
  tool: "",
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
      : status === "absorbed"
        ? "border-amber-200 bg-amber-50 text-amber-700"
        : "border-[var(--otari-line)] bg-[var(--otari-brand-tint)] text-[var(--otari-brand-dark)]";
  return (
    <span className={`inline-flex items-center gap-1.5 rounded-full border px-2 py-0.5 text-xs font-medium ${cls}`}>
      {/* The pulse is decorative: "in progress" beside it carries the same meaning
          in text, so nothing here is encoded in motion alone. That is also why it
          can stop outright under `prefers-reduced-motion`: this is the one element
          on the page that would otherwise animate indefinitely. */}
      {status === IN_FLIGHT_STATUS ? (
        <span
          className="h-1.5 w-1.5 shrink-0 animate-pulse rounded-full bg-[var(--otari-brand-dark)] motion-reduce:animate-none"
          aria-hidden="true"
        />
      ) : null}
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
  // Per key, not per object: a row can carry meters for its gateway-run tool calls
  // while its tokens were never metered (an unpriced model still owes for the
  // searches it ran). Keying off the object's presence would then read every token
  // as 0 and the bar would vanish from a row that has real tokens.
  const meters = entry.billing_meters ?? null;
  const meter = (key: string, fallback: number | null): number =>
    meters && typeof meters[key] === "number" ? positive(meters[key]) : positive(fallback);
  const totalInput = meter("total_input_tokens", entry.prompt_tokens);
  const cacheRead = meter("cache_read_tokens", entry.cache_read_tokens);
  const cacheWrite = meter("cache_write_tokens", entry.cache_write_tokens);
  const output = meter("completion_tokens", entry.completion_tokens);
  const fresh = Math.max(0, totalInput - cacheRead - cacheWrite);
  const total = fresh + cacheRead + cacheWrite + output;
  return total > 0 ? { fresh, cacheRead, cacheWrite, output, total } : null;
}

// A row's gateway-run tool calls, read out of the reserved `tools` meter namespace.
// Nested under one key so a caller-named MCP tool can never collide with a token
// meter (which the billed-token SQL and the bar above both read by name).
type ToolUsage = { tool: string; billed: number; errors: number; unitRate: number | null };

function toolUsage(entry: UsageEntry): ToolUsage[] {
  const nested = entry.billing_meters?.tools;
  if (!nested || typeof nested !== "object") return [];
  return Object.entries(nested as Record<string, unknown>)
    .flatMap(([tool, counts]) => {
      if (!counts || typeof counts !== "object") return [];
      const record = counts as Record<string, unknown>;
      const billed = positive(record.billed);
      const errors = positive(record.errors);
      if (!billed && !errors) return [];
      const rate = record.unit_rate;
      return [{ tool, billed, errors, unitRate: typeof rate === "number" ? rate : null }];
    })
    .sort((a, b) => b.billed - a.billed || a.tool.localeCompare(b.tool));
}

// "web search x3" / "web search x3, 1 failed". The tool name is de-underscored for
// reading; failures are named rather than folded into the count, because a failed
// call is not billed and an operator chasing a cost needs that distinction.
function formatToolUsage(usage: ToolUsage): string {
  const label = usage.tool.replaceAll("_", " ");
  const parts = usage.billed ? [`${label} \u00d7${usage.billed}`] : [label];
  if (usage.errors) parts.push(`${usage.errors} failed`);
  return parts.join(", ");
}


// Cost attributable to gateway-run tools on this row, from the rate stored with the
// row rather than the live price, so a historical row reads as it was billed.
function toolCost(entry: UsageEntry): number | null {
  const usages = toolUsage(entry).filter((usage) => usage.unitRate !== null);
  if (!usages.length) return null;
  return usages.reduce((sum, usage) => sum + usage.billed * (usage.unitRate ?? 0), 0);
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

// ---------- routing ----------
//
// A routed request writes one usage row per attempt (all sharing a
// `request_group_id`), so a single row answers only half of what an operator
// wants: "attempt 1 of 2 failed" without saying what served the request. These
// helpers turn the stored attribution into the sentence the Routing column shows,
// and reassemble a plan from the rows of one group.

// Plain English for a compiled attempt's `selection_reason`. The stored values are
// the compiler's vocabulary (`static`, `default`, `on_failure`, `condition:<keys>`,
// `router:<name>`): precise, and meaningless to a reader who has not read the
// compiler. An unrecognized value is de-underscored rather than dropped, since the
// set is open by construction (a condition or router name comes from config).
function selectionReasonLabel(reason: string | null | undefined): string | null {
  if (!reason) return null;
  if (reason === "static") return "the policy's only target";
  if (reason === "default") return "the policy's default target";
  if (reason === "on_failure") return "a fallback candidate";
  if (reason.startsWith("condition:")) {
    const keys = reason
      .slice("condition:".length)
      .split(",")
      .filter(Boolean)
      .join(", ");
    return keys ? `matched on ${keys}` : "matched a condition";
  }
  if (reason.startsWith("router:")) {
    const name = reason.slice("router:".length);
    return name ? `chosen by router ${name}` : "chosen by a router";
  }
  return reason.replaceAll("_", " ");
}

// What a group's request ended up doing, read off its outcome row. Absorbed rows
// are attempts the policy recovered from, so exactly one row per finished group is
// the outcome: the attempt that served, or the terminal failure.
interface GroupOutcome {
  /** Qualified target of the attempt that served, or null when none did. */
  servedBy: string | null;
  servedPosition: number | null;
}

// Index the outcome of every group represented in `rows`. Built from rows the page
// already holds first, then filled in from a batched lookup, so the common case
// (a group's attempts are adjacent in a newest-first list) costs no extra request.
function indexGroupOutcomes(rows: readonly UsageEntry[]): Map<string, GroupOutcome> {
  const index = new Map<string, GroupOutcome>();
  for (const row of rows) {
    if (!row.request_group_id || row.status === "absorbed") continue;
    index.set(row.request_group_id, {
      servedBy: row.status === "success" ? pricingSelectorOf(row) : null,
      servedPosition: row.status === "success" ? (row.attempt_position ?? null) : null,
    });
  }
  return index;
}

// One line of prose for a row's place in its plan, replacing the "attempt 1/2 ·
// default" shorthand: that read as a fraction of something unnamed, said nothing
// about whether the attempt worked, and pointed at no other row. `outcome` is the
// group's outcome when it is known, which is what lets an absorbed row name the
// model that served in its place.
function attemptSentence(entry: UsageEntry, outcome: GroupOutcome | null): string | null {
  const reason = selectionReasonLabel(entry.selection_reason);
  const position = entry.attempt_position;
  const total = entry.attempt_count;
  // A policy with one candidate has no plan to place the row in, so the only thing
  // worth saying is why that candidate was picked.
  if (position == null || total == null || total <= 1) return reason;
  const attempt = `attempt ${position} of ${total}`;
  if (entry.status === "absorbed") {
    if (outcome?.servedBy) return `${attempt} failed, served by ${outcome.servedBy}`;
    // Not "and so did the rest": the group's outcome row is an error, but the walk
    // may have stopped on it (a non-retryable status, a lock-in) with later
    // candidates never called, which is what that row's own sentence says.
    if (outcome) return `${attempt} failed, and the request ended in an error`;
    return `${attempt} failed, fell back`;
  }
  if (entry.status === "error") {
    // The walk stops early on a non-retryable failure, a lock-in, or a
    // gateway-side refusal, so the later candidates were not necessarily tried.
    return position < total ? `${attempt} failed, no further candidate tried` : `${attempt} failed, plan exhausted`;
  }
  return reason ? `served on ${attempt} (${reason})` : `served on ${attempt}`;
}

// The Routing column: the policy the caller named, then where this row sits in its
// plan and how that turned out.
function RoutingCell({ entry, outcome }: { entry: UsageEntry; outcome: GroupOutcome | null }) {
  // Blank, not an em-dash, when the request named a plain model. This column is
  // sparse by nature (most rows are unrouted), and a placeholder on every one of
  // them would add noise to every scan while saying nothing.
  if (entry.policy_name == null) return null;
  const sentence = attemptSentence(entry, outcome);
  return (
    <span className="flex flex-col leading-tight">
      <span className="text-[var(--otari-ink)]">{entry.policy_name}</span>
      {/* Non-color signal: the outcome is spelled out, so the amber row tint is
          never the only thing carrying the meaning. */}
      {/* Wraps rather than truncates: the tail is the part that matters (it names
          the model that served), and a qualified target is routinely longer than
          the column. The Model column wraps for the same reason. */}
      {sentence ? <span className="text-xs break-words text-[var(--otari-muted)]">{sentence}</span> : null}
    </span>
  );
}

// Per-attempt outcome for the plan table. Terser than the row sentence, which has
// to stand alone; here the table's shape already says which attempt this is.
function attemptOutcome(entry: UsageEntry): string {
  if (entry.status === "absorbed") return entry.status_code === null ? "failed, fell back" : `failed ${entry.status_code}, fell back`;
  if (entry.status === "error") return entry.status_code === null ? "failed" : `failed ${entry.status_code}`;
  return "served the request";
}

// Attempts in plan order. `attempt_position` is authoritative; timestamp is the
// tiebreaker for a row written before the column existed, or a group whose rows
// share a position (which would be a writer bug, not something to hide).
function planOrder(rows: readonly UsageEntry[]): UsageEntry[] {
  return [...rows].sort(
    (a, b) => (a.attempt_position ?? 0) - (b.attempt_position ?? 0) || a.timestamp.localeCompare(b.timestamp),
  );
}

// The whole plan behind one routed request: every candidate that ran, in order,
// with the one that served marked. This is the answer to "a fallback fired, so
// what actually served me", which no single row can give.
function RoutingPlan({ entry }: { entry: UsageEntry }) {
  const groupId = entry.request_group_id;
  const groupIds = groupId ? [groupId] : [];
  const group = useRequestGroups(groupIds);
  // Only rows of this row's own group are the plan. The lookup keeps previous data
  // across a key change, so this is what stops another request's plan from ever
  // being narrated as this one's, whatever the detail panel does with mounting.
  const siblings = groupId ? (group.data ?? []).filter((row) => row.request_group_id === groupId) : [];
  // Falls back to the row itself while the lookup is in flight (and for a
  // pre-`request_group_id` row, which has no siblings to find), so the section
  // never flashes empty and never claims a one-attempt plan it did not read.
  const attempts = planOrder(siblings.length ? siblings : [entry]);
  const complete = siblings.length > 0;
  const served = attempts.find((attempt) => attempt.status === "success");
  const total = entry.attempt_count ?? attempts.length;

  // "Loading" only while a lookup is actually outstanding: a failed lookup, or a row
  // that carries no group to look up, would otherwise sit on that line forever.
  const summary = !complete
    ? group.isError
      ? "Could not load this request's other attempts."
      : entry.request_group_id
        ? "Loading the rest of this request's attempts…"
        : "This row carries no request group, so its other attempts cannot be found."
    : served
      ? `Served by attempt ${served.attempt_position ?? "?"} of ${total}: ${pricingSelectorOf(served)}`
      : attempts.some((attempt) => attempt.status === "error")
        ? "No candidate served this request."
        : "This request has no outcome row yet.";

  return (
    <div className="flex flex-col gap-2">
      <span className="text-[11px] font-medium uppercase tracking-wide text-[var(--otari-muted)]">
        Routing plan · {entry.policy_name}
      </span>
      <span className="text-sm text-[var(--otari-ink)]">{summary}</span>
      <div className="overflow-x-auto rounded-lg border border-[var(--otari-line)]">
        <table className="w-full text-xs" aria-label={`Routing plan for policy ${entry.policy_name}`}>
          <thead className="text-[var(--otari-muted)]">
            <tr className="border-b border-[var(--otari-line)]">
              <th scope="col" className="px-3 py-2 text-left font-medium">#</th>
              <th scope="col" className="px-3 py-2 text-left font-medium">Target</th>
              <th scope="col" className="px-3 py-2 text-left font-medium">Selected as</th>
              <th scope="col" className="px-3 py-2 text-left font-medium">Outcome</th>
              {/* Every attempt's `latency_ms` is measured from the start of the
                  request, not from the start of that attempt, so this is the same
                  "Total time" the row column shows, not a per-candidate duration. */}
              <th scope="col" className="px-3 py-2 text-right font-medium">Total time</th>
              <th scope="col" className="px-3 py-2 text-right font-medium">Cost</th>
            </tr>
          </thead>
          <tbody>
            {attempts.map((attempt) => (
              <tr
                key={attempt.id}
                className={`border-t border-[var(--otari-line)] first:border-t-0 ${
                  attempt.status === "success" ? "bg-[var(--otari-brand-tint)]" : ""
                }`}
              >
                <td className="px-3 py-2 tabular-nums">{attempt.attempt_position ?? "?"}</td>
                <td className="px-3 py-2 break-all text-[var(--otari-ink)]">
                  {pricingSelectorOf(attempt)}
                  {attempt.id === entry.id ? (
                    <span className="ml-2 rounded-full border border-[var(--otari-line)] px-1.5 py-0.5 text-[10px] text-[var(--otari-muted)]">
                      this row
                    </span>
                  ) : null}
                </td>
                <td className="px-3 py-2">{selectionReasonLabel(attempt.selection_reason) ?? "—"}</td>
                <td className={`px-3 py-2 ${attempt.status === "success" ? "" : "text-amber-700"}`}>
                  {attemptOutcome(attempt)}
                </td>
                <td className="px-3 py-2 text-right tabular-nums">{formatLatency(attempt.latency_ms)}</td>
                <td className="px-3 py-2 text-right tabular-nums">{formatUSD(attempt.cost)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <span className="text-xs text-[var(--otari-muted)]">
        Cost and tool charges settle on the attempt that served, so a failed attempt carries its tokens and no charge.
      </span>
    </div>
  );
}

// `copyValue` adds a copy control for the fields that hold an opaque identifier
// (a request id, an api key id): they are what an operator pastes into a log
// search or a support thread, and a mistyped character makes them useless.
function DetailField({
  label,
  copyValue,
  copyLabel,
  children,
}: {
  label: string;
  copyValue?: string | null;
  /** Overrides the copy control's name where the column heading would misname the
      value: "API key" holds a key id, never key material. */
  copyLabel?: string;
  children: ReactNode;
}) {
  return (
    <div className="flex flex-col gap-0.5">
      <span className="text-[11px] font-medium uppercase tracking-wide text-[var(--otari-muted)]">{label}</span>
      {copyValue ? (
        <CopyableValue value={copyValue} label={copyLabel ?? label.toLowerCase()} className="text-sm text-[var(--otari-ink)] break-all">
          {children}
        </CopyableValue>
      ) : (
        <span className="text-sm text-[var(--otari-ink)] break-all">{children}</span>
      )}
    </div>
  );
}

// The pricing key a usage row bills against. A row stores the instance and the
// bare model separately (`log_usage` is called with `provider=resolved.instance,
// model=resolved.model`, and a gateway rejection logs the same pair), while
// pricing is looked up as `instance:model` (`find_model_pricing`), so the key has
// to be rebuilt from both: `entry.model` alone is prefix-less and would store a
// price nothing ever reads. A row whose selector never resolved carries no
// provider and its model is the raw selector, so that is used as-is.
function pricingSelectorOf(entry: UsageEntry): string {
  if (!entry.provider) return entry.model;
  return entry.model.startsWith(`${entry.provider}:`) ? entry.model : `${entry.provider}:${entry.model}`;
}

// The detail panel for one request: the failure diagnostic plus the metadata
// that does not fit the row. The dashboard is master-key admin-only, so the
// stored `error_message` is shown verbatim; it is source-neutral by nature,
// carrying either a fixed gateway rejection string (e.g. a model with no pricing
// under `require_pricing`) or the raw upstream provider error, so the heading
// stays "Error" rather than blaming the provider for every failure.
function RequestDetail({ entry, onPriceModel }: { entry: UsageEntry; onPriceModel: (model: string) => void }) {
  // A row with no cost (cost IS NULL, the same test the "Priced?" filter uses)
  // is either a model the gateway has no price for or a request refused before
  // it could be billed. Both are the same fix, and the row holds what the
  // selector was, which a provider without model discovery would never have put
  // in the catalogue. A $0 cost is a real price, so it is deliberately not
  // treated as uncosted.
  const uncosted = entry.cost === null;
  const pricingKey = pricingSelectorOf(entry);
  return (
    <div className="flex flex-col gap-4 px-4 py-4">
      {entry.error_message ? (
        <div className="flex flex-col gap-1.5">
          <span className="text-[11px] font-medium uppercase tracking-wide text-[var(--otari-muted)]">
            Error{entry.status_code !== null ? ` (${entry.status_code})` : ""}
          </span>
          <pre className="max-h-48 overflow-auto rounded-lg border border-red-200 bg-red-50 p-3 text-xs whitespace-pre-wrap break-all text-red-700">
            {entry.error_message}
          </pre>
        </div>
      ) : null}
      {/* Routed requests only. Placed above the metadata grid, and above the
          per-row fields, because on a failed attempt it answers the first question
          the failure raises: what served the request instead. */}
      {entry.policy_name !== null && entry.policy_name !== undefined ? <RoutingPlan entry={entry} /> : null}
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <DetailField label="Provider">{entry.provider ?? "—"}</DetailField>
        <DetailField label="Endpoint">{entry.endpoint}</DetailField>
        <DetailField label="Source">{sourceLabel(entry.source)}</DetailField>
        {entry.source_label ? <DetailField label="Session">{entry.source_label}</DetailField> : null}
        <DetailField label="User" copyValue={entry.user_id} copyLabel="user id">{entry.user_id ?? "—"}</DetailField>
        <DetailField label="API key" copyValue={entry.api_key_id} copyLabel="api key id">{entry.api_key_id ?? "—"}</DetailField>
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
        {toolUsage(entry).length ? (
          <>
            <DetailField label="Tools">
              {toolUsage(entry).map(formatToolUsage).join(" \u00b7 ")}
            </DetailField>
            <DetailField label="Tool cost">
              {toolCost(entry) === null ? (
                <span
                  className="text-[var(--otari-warning-ink,var(--otari-muted))]"
                  title="No per-request price is configured for this tool, so its calls were recorded at zero cost. Set one on the Tools & Guardrails screen."
                >
                  unpriced
                </span>
              ) : (
                formatUSD(toolCost(entry))
              )}
            </DetailField>
          </>
        ) : null}
        <DetailField label="Cache read tokens">{formatTokens(entry.cache_read_tokens)}</DetailField>
        <DetailField label="Cache write tokens">{formatTokens(entry.cache_write_tokens)}</DetailField>
        <DetailField label="1h cache writes">{formatTokens(entry.cache_write_1h_tokens ?? null)}</DetailField>
        <DetailField label="Total time">{formatLatency(entry.latency_ms)}</DetailField>
        <DetailField label="Request ID" copyValue={entry.id}>{entry.id}</DetailField>
      </div>
      {uncosted ? (
        <div className="flex flex-wrap items-center gap-3">
          <Button size="sm" variant="outline" onPress={() => onPriceModel(pricingKey)}>
            Price this model
          </Button>
          <span className="text-xs text-[var(--otari-muted)]">
            This request carries no cost. Set a price for <code className="break-all">{pricingKey}</code> so later
            requests are metered and count against budgets. Rows already logged keep the cost they were served with.
          </span>
        </div>
      ) : null}
      {entry.pricing_breakdown?.length ? (
        <div className="flex flex-col gap-2">
          <span className="text-[11px] font-medium uppercase tracking-wide text-[var(--otari-muted)]">
            Billed meters
          </span>
          <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
            {sortedBreakdown(entry.pricing_breakdown).map((line) => (
              <DetailField key={line.meter} label={line.meter.replaceAll("_", " ")}>
                {"unit_rate" in line
                  ? `${formatTokens(line.units)} at ${formatUnitRate(line.unit_rate)} each, ${formatUSD(line.cost)}`
                  : `${formatTokens(line.units)} at ${formatUSD(line.rate_per_million)} / 1M, ${formatUSD(line.cost)}`}
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
  // Filter + pagination state lives in the URL, so a filtered view is shareable
  // and survives the back button. `patch` batches related changes into one entry.
  const url = useUrlState(URL_DEFAULTS);
  const range = url.get("range");
  const startParam = url.get("start_date");
  const endParam = url.get("end_date");
  const statusFilter = url.get("status");
  // The three entity filters hold sets: each is repeatable in the URL and on the
  // wire, so a comparison ("these two models") is one view rather than several, and
  // a drill-down from the analytics page can carry its whole selection across.
  const modelFilters = url.getAll("model");
  const userFilters = url.getAll("user_id");
  const apiKeyFilters = url.getAll("api_key_id");
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
  const toolFilter = url.get("tool");
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
      model: modelFilters.length > 0 ? modelFilters : undefined,
      user_id: userFilters.length > 0 ? userFilters : undefined,
      api_key_id: apiKeyFilters.length > 0 ? apiKeyFilters : undefined,
      source: sourceFilter || undefined,
      source_label: sessionFilter || undefined,
      endpoint: endpointFilter || undefined,
      provider: providerFilter || undefined,
      tool: (toolFilter || undefined) as UsageFilters["tool"],
      priced,
    }),
    [
      win,
      toolFilter,
      statusFilter,
      modelFilters,
      userFilters,
      apiKeyFilters,
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

  // Requests in progress, read unfiltered: the endpoint has no filters, because a
  // request that has not finished has no outcome, cost, or token count to filter
  // on. Which of them belong in the current view is decided below.
  const inFlight = useInFlightRequests();

  // A tracked request that has left the in-flight list just settled, so its row
  // exists in the log now. Pull it in, so the operator watches the live row turn
  // into its settled one rather than vanish until they press refresh. Keyed on
  // ids, not on the count: one request finishing while another starts leaves the
  // count unchanged.
  //
  // Rate-limited, because the registry is per-process: a deployment running
  // several otari processes behind a load balancer answers consecutive polls from
  // different processes, so a request that is still running reads as settled on
  // every poll. Unthrottled that refetches the log and its `COUNT(*)` every two
  // seconds for as long as the gateway has any traffic at all. The ceiling matches
  // the log's own `staleTime`, so a genuine settle still refreshes promptly and
  // the flapping case costs one extra pair of reads per interval instead of one
  // per poll.
  //
  // Deferred rather than dropped when the throttle bites: a settle is remembered
  // on `settlePending` and served by the first poll past the window. Discarding it
  // instead lost the row entirely, because the id is gone from the next poll and
  // nothing re-detects it: two requests landing inside one 10-second window left
  // the second one's row missing from the log until an unrelated later settle or a
  // manual refresh. The flapping case is unchanged (it re-flags every poll either
  // way, and still costs one refetch per interval).
  //
  // Keyed on `dataUpdatedAt` as well as the payload, so a deferred settle is
  // reconsidered on the next poll: an idle registry answers every poll with the
  // same `{requests: [], total: 0}`, which structural sharing hands back as the
  // same object, so `data` alone stops changing the moment the last request lands.
  const seenInFlightIds = useRef<Set<string>>(new Set());
  const lastSettledRefetch = useRef(0);
  const settlePending = useRef(false);
  useEffect(() => {
    if (!inFlight.data) return;
    const live = new Set((inFlight.data.requests ?? []).map((request) => request.id));
    if ([...seenInFlightIds.current].some((id) => !live.has(id))) settlePending.current = true;
    seenInFlightIds.current = live;
    if (!settlePending.current) return;
    const now = Date.now();
    if (now - lastSettledRefetch.current < SETTLED_REFETCH_MIN_MS) return;
    settlePending.current = false;
    lastSettledRefetch.current = now;
    void usage.refetch();
    void count.refetch();
  }, [inFlight.data, inFlight.dataUpdatedAt, usage.refetch, count.refetch]);

  // Model suggestions: models with usage in the window (other filters applied, the
  // model filter omitted so the full list stays offered).
  const modelSuggestFilters: UsageFilters = useMemo(
    () => ({
      start_date: win.start,
      end_date: win.end,
      status: statusFilter || undefined,
      user_id: userFilters.length > 0 ? userFilters : undefined,
      api_key_id: apiKeyFilters.length > 0 ? apiKeyFilters : undefined,
      source: sourceFilter || undefined,
      source_label: sessionFilter || undefined,
      endpoint: endpointFilter || undefined,
      provider: providerFilter || undefined,
      tool: (toolFilter || undefined) as UsageFilters["tool"],
    }),
    [
      win,
      statusFilter,
      userFilters,
      apiKeyFilters,
      sourceFilter,
      sessionFilter,
      endpointFilter,
      providerFilter,
      toolFilter,
    ],
  );
  // Two breakdowns are read here (model typeahead, source picker); the rest are
  // not requested.
  const modelSummary = useUsageSummary(modelSuggestFilters, "day", MODEL_AND_SOURCE_BREAKDOWNS);
  const realGroups = (rows: UsageGroupRow[] | undefined) =>
    (rows ?? []).filter((r) => !r.is_other && r.key !== null);
  const modelOptions = realGroups(modelSummary.data?.by_model).map((r) => r.key as string);

  // The user and key pickers need their own window: each must keep offering the
  // *other* values of its own dimension, so both entity filters come off. That
  // cannot share the model/source query above, which has to keep them applied,
  // or filtering Activity to one user would make the typeahead suggest only the
  // models other users called and picking one would return an empty table.
  const entitySuggestFilters: UsageFilters = useMemo(
    () => ({ ...filters, user_id: undefined, api_key_id: undefined }),
    [filters],
  );
  const entitySummary = useUsageSummary(entitySuggestFilters, "day", ENTITY_BREAKDOWNS);
  const keyOptions = realGroups(entitySummary.data?.by_api_key).map((r) => ({
    value: r.key as string,
    label: r.label ?? `${(r.key as string).slice(0, 8)}…`,
  }));

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
      model: modelFilters.length > 0 ? modelFilters : undefined,
      user_id: userFilters.length > 0 ? userFilters : undefined,
      api_key_id: apiKeyFilters.length > 0 ? apiKeyFilters : undefined,
    }),
    [win, statusFilter, modelFilters, userFilters, apiKeyFilters],
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
      model: modelFilters.length > 0 ? modelFilters : undefined,
      user_id: userFilters.length > 0 ? userFilters : undefined,
      api_key_id: apiKeyFilters.length > 0 ? apiKeyFilters : undefined,
      source: sourceFilter || undefined,
      source_label: sessionFilter || undefined,
      endpoint: endpointFilter || undefined,
      provider: providerFilter || undefined,
      tool: (toolFilter || undefined) as UsageFilters["tool"],
      priced,
    }),
    [
      winOutsideExtent,
      toolFilter,
      win,
      extentWin,
      statusFilter,
      modelFilters,
      userFilters,
      apiKeyFilters,
      sourceFilter,
      sessionFilter,
      endpointFilter,
      providerFilter,
      priced,
    ],
  );
  // The timeline reads `series`; the tool dimension is requested so the Tool filter
  // knows whether this window contains any gateway-run tool calls. With
  // NO_BREAKDOWNS the server returns `by_tool: []` by contract, which left the
  // selector permanently hidden unless a tool filter was already in the URL.
  const contextSummary = useUsageSummary(contextFilters, extentBucket, TOOL_BREAKDOWN);
  const timelineSeries = (contextSummary.data?.series ?? []).map((p) => ({
    bucketStart: p.bucket_start,
    requests: p.requests,
    // Failed requests render as a red segment on the strip, so dropped traffic
    // shows up while browsing, not only after filtering to status=error.
    errors: p.errors ?? 0,
  }));

  const rows = usage.data ?? [];

  // The live rows to pin above this page, and how many the endpoint left out.
  //
  // Kept separate from `rows` throughout: `rows` is what the paginator counts and
  // what the bulk actions target, and a request that has not been logged belongs
  // in neither. Newest-first like the rest of the table, so the row an operator
  // just triggered appears at the top and stays there as it resolves (the endpoint
  // serves the longest-running first, which is the order the cap applies in).
  //
  // Suppressed rather than shown wherever the current view cannot honestly include
  // them: page 2 onward (they are not part of any page's slice), a window that ends
  // in the past, and any filter on something a request has not got yet (outcome,
  // price, tools, provenance). The identity filters can be applied, so they are.
  const { inFlightRows, inFlightHidden } = useMemo(() => {
    const none = { inFlightRows: [] as ActivityRow[], inFlightHidden: 0 };
    if (!inFlight.data || page > 0) return none;
    if (filters.end_date || statusFilter || pricedFilter || toolFilter || sourceFilter || sessionFilter) return none;
    const matching = inFlight.data.requests.filter(
      (request) =>
        (modelFilters.length === 0 || modelFilters.includes(request.model)) &&
        (userFilters.length === 0 || (request.user_id !== null && userFilters.includes(request.user_id))) &&
        (apiKeyFilters.length === 0 || (request.api_key_id !== null && apiKeyFilters.includes(request.api_key_id))) &&
        (!endpointFilter || request.endpoint === endpointFilter) &&
        (!providerFilter || request.provider === providerFilter),
    );
    return {
      inFlightRows: matching
        .map((request) => inFlightRow(request, inFlight.dataUpdatedAt - request.elapsed_ms))
        .reverse(),
      // Only meaningful when nothing was filtered out locally; a filtered view
      // cannot say how many of the requests the endpoint dropped would have matched.
      inFlightHidden: matching.length === inFlight.data.requests.length ? inFlight.data.total - matching.length : 0,
    };
  }, [
    inFlight.data,
    inFlight.dataUpdatedAt,
    page,
    filters.end_date,
    statusFilter,
    pricedFilter,
    toolFilter,
    sourceFilter,
    sessionFilter,
    modelFilters,
    userFilters,
    apiKeyFilters,
    endpointFilter,
    providerFilter,
  ]);

  // What the table renders: live rows pinned above the logged page.
  const tableRows = useMemo<ActivityRow[]>(() => [...inFlightRows, ...rows], [inFlightRows, rows]);
  const inFlightIds = useMemo(() => new Set(inFlightRows.map((row) => row.id)), [inFlightRows]);

  // What served each routed request on this page. Built from the page itself where
  // possible (a group's attempts are written milliseconds apart, so they are
  // usually adjacent in a newest-first list), and looked up for the groups whose
  // outcome row is missing: a page boundary splits a group, and filtering to the
  // `absorbed` status (the way an operator investigates fallovers) hides every
  // outcome row by construction, which is precisely when the answer is wanted.
  const { pageOutcomes, unresolvedGroupIds } = useMemo(() => {
    const known = indexGroupOutcomes(rows);
    const missing = new Set<string>();
    for (const row of rows) {
      if (row.status === "absorbed" && row.request_group_id && !known.has(row.request_group_id)) {
        missing.add(row.request_group_id);
      }
    }
    return { pageOutcomes: known, unresolvedGroupIds: [...missing] };
  }, [rows]);
  const unresolvedGroups = useRequestGroups(unresolvedGroupIds);
  const groupOutcomes = useMemo(() => {
    if (!unresolvedGroups.data?.length) return pageOutcomes;
    return new Map([...pageOutcomes, ...indexGroupOutcomes(unresolvedGroups.data)]);
  }, [pageOutcomes, unresolvedGroups.data]);

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
      modelFilters.length ||
      userFilters.length ||
      apiKeyFilters.length ||
      pricedFilter ||
      sourceFilter ||
      sessionFilter ||
      endpointFilter ||
      providerFilter ||
      toolFilter ||
      timeFiltered,
  );

  // Active entity filters as removable chips (time is driven by the timeline, so
  // it is not a chip). Values show the human label where one exists.
  const labelFrom = (options: { value: string; label: string }[], value: string) =>
    options.find((o) => o.value === value)?.label ?? value;
  const userOptionsList = realGroups(entitySummary.data?.by_user).map((r) => ({
    value: r.key as string,
    label: r.label ? `${r.label} (${r.key})` : (r.key as string),
  }));
  const clearEntityFilters = () =>
    url.patch({
      status: "",
      priced: "",
      model: [],
      user_id: [],
      api_key_id: [],
      source: "",
      source_label: "",
      endpoint: "",
      provider: "",
      tool: "",
    });
  // One chip per picked value of a repeatable filter, each clearing only itself.
  const valueChips = (
    dimension: string,
    label: string,
    param: "model" | "user_id" | "api_key_id",
    values: string[],
    display: (value: string) => string,
  ): FilterChip[] =>
    values.map((value) => ({
      key: `${dimension}:${value}`,
      label,
      value: display(value),
      // Several chips share a dimension, so the value has to be part of the name.
      clearLabel: `Remove ${label} filter ${display(value)}`,
      onClear: () => url.patch({ [param]: values.filter((v) => v !== value) }),
    }));
  const filterChips: FilterChip[] = [
    ...(statusFilter ? [{ key: "status", label: "Status", value: labelFrom(STATUS_OPTIONS, statusFilter), onClear: () => url.patch({ status: "" }) }] : []),
    ...(pricedFilter ? [{ key: "priced", label: "Priced", value: labelFrom(PRICED_OPTIONS, pricedFilter), onClear: () => url.patch({ priced: "" }) }] : []),
    ...valueChips("user", "User", "user_id", userFilters, (v) => labelFrom(userOptionsList, v)),
    ...valueChips("model", "Model", "model", modelFilters, (v) => v),
    ...valueChips("key", "API key", "api_key_id", apiKeyFilters, (v) => labelFrom(keyOptions, v)),
    ...(sourceFilter ? [{ key: "source", label: "Source", value: sourceLabel(sourceFilter), onClear: () => url.patch({ source: "" }) }] : []),
    ...(sessionFilter ? [{ key: "session", label: "Session", value: sessionFilter, onClear: () => url.patch({ source_label: "" }) }] : []),
    ...(endpointFilter ? [{ key: "endpoint", label: "Endpoint", value: endpointFilter, onClear: () => url.patch({ endpoint: "" }) }] : []),
    ...(providerFilter ? [{ key: "provider", label: "Provider", value: providerFilter, onClear: () => url.patch({ provider: "" }) }] : []),
    ...(toolFilter ? [{ key: "tool", label: "Tool", value: labelFrom(TOOL_OPTIONS, toolFilter), onClear: () => url.patch({ tool: "" }) }] : []),
  ];

  // Selection targets imported rows only; enforced gateway rows are disabled so
  // bulk delete / set-price can never reach them. Over `tableRows`, so a live row
  // is disabled too: it carries `counts_toward_budget` true for exactly that.
  const selectableKeys = useMemo(() => rows.filter((r) => !r.counts_toward_budget).map((r) => r.id), [rows]);
  const disabledKeys = useMemo(() => tableRows.filter((r) => r.counts_toward_budget).map((r) => r.id), [tableRows]);
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
  // Only offer the selection column when something on the page can actually be
  // selected. A deployment with no imported usage has none: every checkbox would
  // render disabled, which reads as a broken control rather than as "these rows
  // are not eligible". Kept while "all matching" is live so the affordance does
  // not vanish under an operator mid-bulk-op.
  const showSelection = selectableKeys.length > 0 || selection.allMatching;

  const deleteUsage = useDeleteUsage();
  const setPrice = useSetUsagePrice();
  const setModelPrice = useSetPricing();
  const [deleteOpen, setDeleteOpen] = useState(false);
  const [priceOpen, setPriceOpen] = useState(false);
  // The model selector whose price is being set from a request detail, or null
  // when that dialog is closed. Distinct from `priceOpen` above, which reprices
  // already-logged imported rows rather than setting a model's price.
  const [modelPriceKey, setModelPriceKey] = useState<string | null>(null);
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
        <RequestDetail entry={entry} onPriceModel={setModelPriceKey} />
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
  //
  // The entity filters travel as the sets they are: the selection body takes the same
  // repeatable form as the read filters, so "all N matching" targets exactly the rows
  // the count was taken over (see UsageSelection).
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
          tool: filters.tool,
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

  // Sets the model's own price (a ModelPricing row), which is what future
  // requests are billed at. Deliberately does not touch the rows already logged:
  // a gateway row's cost is what it was served at, and rewriting history from an
  // activity view would move spend that budgets were already enforced against.
  const onSetModelPrice = (rates: ManualRates, modelKey: string) => {
    setModelPrice.mutate(
      {
        model_key: modelKey,
        input_price_per_million: rates.input_price_per_million,
        output_price_per_million: rates.output_price_per_million,
        cache_read_price_per_million: rates.cache_read_price_per_million ?? null,
        cache_write_price_per_million: rates.cache_write_price_per_million ?? null,
      },
      { onSuccess: () => setModelPriceKey(null) },
    );
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
    void inFlight.refetch();
    void contextSummary.refetch();
    void modelSummary.refetch();
    void entitySummary.refetch();
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

  // Memoized on its per-render inputs (the key labels and the routing outcomes,
  // both themselves memoized) so DataTable's per-row cache holds: a fresh array
  // every render would rebuild all rows per click.
  const columns = useMemo<DataTableColumn<ActivityRow>[]>(() => {
    const apiKeyLabel = (entry: UsageEntry): string =>
      entry.api_key_id === null ? "—" : (entry.api_key_name ?? `${entry.api_key_id.slice(0, 8)}…`);
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
      {
        id: "model",
        header: "Model",
        isRowHeader: true,
        // The tool marker lives here rather than in a column of its own: a ninth
        // column would compete with the token bar for the row's only graphic slot
        // and push the failure-forward Status pill off a narrow viewport. Text, not
        // colour alone, so it survives the same accessibility bar as TokenBar.
        cell: (e) => {
          const usage = toolUsage(e);
          if (!usage.length) return e.model;
          const calls = usage.reduce((sum, u) => sum + u.billed + u.errors, 0);
          const detail = usage.map(formatToolUsage).join(" \u00b7 ");
          return (
            <span className="inline-flex items-center gap-1.5">
              {e.model}
              <span
                className="inline-flex items-center rounded-full border border-[var(--otari-line)] bg-[var(--otari-brand-tint)] px-1.5 py-0.5 text-[11px] font-medium text-[var(--otari-brand-dark)]"
                title={detail}
                aria-label={`Gateway tools: ${detail}`}
              >
                {calls} {calls === 1 ? "tool" : "tools"}
              </span>
            </span>
          );
        },
      },
      {
        id: "routing",
        header: "Routing",
        // The policy the caller named, plus where this row sits in its plan and how
        // that turned out. The Model column keeps meaning the model that actually
        // ran (it is the join key for filters and for spend-by-model), so this is
        // additive: together they answer "what did I ask for, and what served it".
        cell: (e) => <RoutingCell entry={e} outcome={groupOutcomes.get(e.request_group_id ?? "") ?? null} />,
      },
      { id: "api_key", header: "API key", cell: (e) => <span className="text-[var(--otari-muted)]">{apiKeyLabel(e)}</span> },
      { id: "tokens", header: "Tokens", align: "end", cell: (e) => <TokenBar entry={e} /> },
      { id: "cost", header: "Cost", align: "end", cell: (e) => formatUSD(e.cost) },
      {
        id: "latency",
        header: "Total time",
        align: "end",
        // A live row has no total yet, so the column carries the wait so far. Same
        // column deliberately: it is the number an operator is already reading down.
        cell: (e) =>
          e.inFlightStartedAtMs === undefined ? (
            formatLatency(e.latency_ms)
          ) : (
            <InFlightWait startedAtMs={e.inFlightStartedAtMs} />
          ),
      },
      { id: "status", header: "Status", cell: (e) => <StatusPill status={e.status} /> },
    ];
  }, [groupOutcomes]);

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
          {/* Only offered once the window actually contains tool usage, following the
              source select below: a filter whose every option returns nothing is
              noise on the majority of gateways, which run no tools at all. */}
          {toolFilter || contextSummary.data?.by_tool?.length ? (
            <FilterSelect id="filter-tool" label="Tool" value={toolFilter} onChange={(value) => url.patch({ tool: value })}>
              {TOOL_OPTIONS.map((opt) => (
                <option key={opt.value} value={opt.value}>
                  {opt.label}
                </option>
              ))}
            </FilterSelect>
          ) : null}
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
          {/* allowsCustom on all three: the options are the in-window top spenders
              (a breakdown capped at 100), so an entity that exists but ranks below
              that, or has no traffic in the window, is not offered. Enter commits a
              pasted id anyway, the way the Model box already accepts a name the
              suggestions do not cover. */}
          <FilterMultiComboBox
            label="API key"
            values={apiKeyFilters}
            onChange={(values) => url.patch({ api_key_id: values })}
            allowsCustom
            placeholder="All keys"
            options={keyOptions}
          />
          <FilterMultiComboBox
            label="User"
            values={userFilters}
            onChange={(values) => url.patch({ user_id: values })}
            allowsCustom
            placeholder="All users"
            options={userOptionsList}
          />
          <FilterMultiComboBox
            label="Model"
            values={modelFilters}
            onChange={(values) => url.patch({ model: values })}
            allowsCustom
            placeholder="Any model"
            options={modelOptions.map((m) => ({ value: m, label: m }))}
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
        rows={tableRows}
        getRowKey={getActivityRowKey}
        isLoading={usage.isLoading}
        emptyContent={anyFilter ? "No requests match these filters." : "No requests recorded yet."}
        selectionMode={showSelection ? "multiple" : "none"}
        selectedKeys={selection.selectedKeys}
        onSelectionChange={selection.onSelectionChange}
        disabledKeys={disabledKeys}
        // A live row has no detail to open: every field the panel reads (tokens,
        // cost, the pricing breakdown, the attempt plan) is written when the
        // request settles. Clicking it does nothing until it does.
        onRowAction={(key) => {
          if (inFlightIds.has(key)) return;
          setExpandedId((current) => (current === key ? null : key));
        }}
        rowClassName={activityRowClassName}
        detailKey={expandedId}
        renderDetail={renderDetail}
      />

      {/* Only when the endpoint's cap actually bit, which takes more concurrency
          than a live view can usefully list anyway. */}
      {inFlightHidden > 0 ? (
        <p className="text-xs text-[var(--otari-muted)]">
          {inFlightHidden.toLocaleString()} further {inFlightHidden === 1 ? "request is" : "requests are"} in flight
          beyond the {inFlightRows.length.toLocaleString()} listed; the longest-running are shown.
        </p>
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

      <SetPriceDialog
        isOpen={modelPriceKey !== null}
        onOpenChange={(open) => setModelPriceKey(open ? (modelPriceKey ?? "") : null)}
        isPending={setModelPrice.isPending}
        error={setModelPrice.error}
        onSubmit={onSetModelPrice}
        collectModelKey
        initialModelKey={modelPriceKey ?? ""}
        title="Price this model"
        description={() =>
          "Set what this model costs, taken from the request you were looking at. Requests from now on are costed at these rates and counted against budgets; rows already logged keep the cost they were served with."
        }
      />
    </div>
  );
}
