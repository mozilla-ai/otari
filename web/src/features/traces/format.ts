import { formatLatency } from "@/shared/helpers/format"

const pad = (n: number, width = 2) => String(n).padStart(width, "0")

// Langfuse's `YYYY-MM-DD HH:mm:ss` (`.SSS` in the detail header), in local time.
export function formatTraceTimestamp(iso: string, withMs = false): string {
  const d = new Date(iso)
  const date = `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}`
  const time = `${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`
  return `${date} ${time}${withMs ? `.${pad(d.getMilliseconds(), 3)}` : ""}`
}

// `formatLatency` for under a minute; "2m 05s" from 60 s up.
export function formatDuration(ms: number): string | undefined {
  if (ms >= 60_000) {
    const total = Math.round(ms / 1000)
    return `${Math.floor(total / 60)}m ${pad(total % 60)}s`
  }
  return formatLatency(ms)
}
