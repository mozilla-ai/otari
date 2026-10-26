import { formatLatency, formatNumber } from "@/shared/helpers/format"

const pad = (n: number, width = 2) => String(n).padStart(width, "0")

// `YYYY-MM-DD HH:mm:ss` (`.SSS` in the detail header), in local time: fixed
// width and sortable, so a column of them lines up digit for digit.
export function formatTraceTimestamp(iso: string, withMs = false): string {
  const d = new Date(iso)
  const date = `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}`
  const time = `${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`
  return `${date} ${time}${withMs ? `.${pad(d.getMilliseconds(), 3)}` : ""}`
}

export function formatDuration(ms: number): string {
  if (ms < 60_000) return formatLatency(ms)
  // The unit is picked on the rounded value, or 59m 59.6s would read "60m 00s".
  const seconds = Math.round(ms / 1000)
  if (seconds < 3600) {
    return `${Math.floor(seconds / 60)}m ${pad(seconds % 60)}s`
  }
  const minutes = Math.round(seconds / 60)
  if (minutes < 24 * 60) {
    return `${Math.floor(minutes / 60)}h ${pad(minutes % 60)}m`
  }
  const hours = Math.round(minutes / 60)
  return `${Math.floor(hours / 24)}d ${pad(hours % 24)}h`
}

export function plural(count: number, word: string): string {
  return `${formatNumber(count)} ${word}${count === 1 ? "" : "s"}`
}

// A part's share of a whole, in whole percent. A part too small to round to 1%
// still happened, so it reads "<1%" rather than "0%".
export function formatShare(fraction: number): string {
  const pct = Math.round(fraction * 100)
  return fraction > 0 && pct === 0 ? "<1%" : `${pct}%`
}

// Log View's Start column: `m:ss`, or `m:ss.mmm`, relative to the trace start.
export function formatRelativeTime(ms: number, showMs = false): string {
  const total = Math.floor(ms / 1000)
  const base = `${Math.floor(total / 60)}:${pad(total % 60)}`
  return showMs ? `${base}.${pad(Math.floor(ms % 1000), 3)}` : base
}
