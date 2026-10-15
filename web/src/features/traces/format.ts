import { formatLatency } from "@/shared/helpers/format"

const pad = (n: number, width = 2) => String(n).padStart(width, "0")

// `YYYY-MM-DD HH:mm:ss` (`.SSS` in the detail header), in local time: fixed
// width and sortable, so a column of them lines up digit for digit.
export function formatTraceTimestamp(iso: string, withMs = false): string {
  const d = new Date(iso)
  const date = `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}`
  const time = `${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`
  return `${date} ${time}${withMs ? `.${pad(d.getMilliseconds(), 3)}` : ""}`
}

// `formatLatency` for under a minute; "2m 05s" from 60 s up.
export function formatDuration(ms: number): string {
  if (ms >= 60_000) {
    const total = Math.round(ms / 1000)
    return `${Math.floor(total / 60)}m ${pad(total % 60)}s`
  }
  return formatLatency(ms)
}

// Log View's Start column: `m:ss`, or `m:ss.mmm`, relative to the trace start.
export function formatRelativeTime(ms: number, showMs = false): string {
  const total = Math.floor(ms / 1000)
  const base = `${Math.floor(total / 60)}:${pad(total % 60)}`
  return showMs ? `${base}.${pad(Math.floor(ms % 1000), 3)}` : base
}
