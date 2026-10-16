import type { KeyboardEvent } from "react"

const KEYS = new Set(["ArrowDown", "ArrowUp", "Home", "End"])

// Up, Down, Home and End move focus between the rows of the nearest
// `[data-trace-rows]` list, which keeps a long session to one tab stop: only
// the open row (or the first) takes Tab, and the arrows reach the rest.
export function moveRowFocus(event: KeyboardEvent<HTMLElement>) {
  if (!KEYS.has(event.key)) return
  const list = event.currentTarget.closest("[data-trace-rows]")
  if (!list) return
  const rows = [...list.querySelectorAll<HTMLElement>("[data-trace-row]")]
  const at = rows.indexOf(event.currentTarget)
  const next =
    event.key === "Home"
      ? 0
      : event.key === "End"
        ? rows.length - 1
        : at + (event.key === "ArrowDown" ? 1 : -1)
  const target = rows[Math.min(rows.length - 1, Math.max(0, next))]
  event.preventDefault()
  target?.focus()
}
