import { useEffect, useState } from "react";

import type { InFlightRequest } from "@/api/types";

// The requests the gateway is serving right now, shown above the activity log.
//
// A separate panel rather than synthetic rows inside the log table, for three
// reasons: an in-flight request has no outcome, cost, or token count, so it would
// fill most of the table's columns with em-dashes; it is not a usage row, so it
// must stay out of the paginator's "N of M", out of the filters, and out of the
// bulk delete / reprice selection; and pinning rows above a paginated list breaks
// the "row N of the current page" reading that the pagination footer promises.
//
// Hidden entirely when nothing is running, so a quiet gateway does not carry a
// permanently empty box.

// How often the elapsed times are re-rendered. The server's `elapsed_ms` is only
// as fresh as the last poll, so ticking locally between polls is what makes a slow
// request read as "still going" rather than as a number that jumps every 2s.
const TICK_MS = 1000;

// Elapsed time, coarser than the log's latency column: this is a wall-clock wait
// an operator is watching, not a settled measurement, so sub-second precision is
// noise. Minutes appear because a stuck local model is the case this exists for.
function formatElapsed(ms: number): string {
  const seconds = Math.floor(ms / 1000);
  if (seconds < 60) return `${seconds}s`;
  const minutes = Math.floor(seconds / 60);
  return `${minutes}m ${String(seconds % 60).padStart(2, "0")}s`;
}

// Endpoints read as their trailing segment ("completions", "messages", "responses"):
// the full path is the same 20 characters on every row and the tail is the part
// that differs.
function endpointLabel(endpoint: string): string {
  const tail = endpoint.split("/").filter(Boolean).pop();
  return tail ?? endpoint;
}

export interface InFlightRequestsProps {
  requests: readonly InFlightRequest[];
  /** True in-flight count, which may exceed `requests.length` (the API caps it). */
  total: number;
  /**
   * When the `requests` snapshot was received (TanStack's `dataUpdatedAt`). Elapsed
   * times are advanced from here rather than from `started_at`, so the display is
   * immune to the browser clock disagreeing with the gateway's.
   */
  updatedAt: number;
}

export function InFlightRequests({ requests, total, updatedAt }: InFlightRequestsProps) {
  // Re-render on a timer so the elapsed column advances between polls. Gated on
  // there being something to tick: the component stays mounted on a quiet gateway
  // (it renders nothing), and an idle second-by-second re-render of the whole
  // Activity page is not free.
  const active = requests.length > 0;
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    if (!active) return;
    const timer = setInterval(() => setNow(Date.now()), TICK_MS);
    return () => clearInterval(timer);
  }, [active]);

  if (!active) {
    return null;
  }

  // Never negative: a re-render can land before `updatedAt` if a poll resolves
  // between the tick and the paint.
  const sincePoll = Math.max(0, now - updatedAt);
  const hidden = total - requests.length;

  return (
    <section
      aria-label="Requests in flight"
      className="rounded-xl border border-[var(--otari-brand-soft)] bg-[var(--otari-surface)]"
    >
      <div className="flex flex-wrap items-center gap-2 border-b border-[var(--otari-line)] bg-[var(--otari-brand-tint)] px-4 py-2">
        {/* The pulse is decorative; the count beside it carries the same meaning in
            text, so nothing is encoded in motion alone. */}
        <span className="h-2 w-2 shrink-0 animate-pulse rounded-full bg-[var(--otari-brand-dark)]" aria-hidden="true" />
        <span className="text-sm font-medium text-[var(--otari-brand-dark)]">
          {total.toLocaleString()} {total === 1 ? "request" : "requests"} in flight
        </span>
        {hidden > 0 ? (
          <span className="text-xs text-[var(--otari-muted)]">showing the {requests.length} longest-running</span>
        ) : null}
      </div>
      <ul className="divide-y divide-[var(--otari-line)]">
        {requests.map((request) => (
          <li key={request.id} className="flex flex-wrap items-baseline gap-x-3 gap-y-1 px-4 py-2 text-sm">
            <span className="font-medium text-[var(--otari-ink)]">{request.model}</span>
            {request.policy_name ? (
              <span className="text-xs text-[var(--otari-muted)]">via {request.policy_name}</span>
            ) : null}
            <span className="text-xs text-[var(--otari-muted)]">{endpointLabel(request.endpoint)}</span>
            {request.user_id ? <span className="text-xs text-[var(--otari-muted)]">{request.user_id}</span> : null}
            <span
              className="ml-auto tabular-nums text-[var(--otari-brand-dark)]"
              title={`Started ${new Date(request.started_at).toLocaleString()}`}
            >
              {formatElapsed(request.elapsed_ms + sincePoll)}
            </span>
          </li>
        ))}
      </ul>
    </section>
  );
}
