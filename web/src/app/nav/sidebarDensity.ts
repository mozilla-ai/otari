import { useCallback, useSyncExternalStore } from "react"

/**
 * The dense sidebar: the rail at 32px rows, 13px labels and a 240px width,
 * beside the default 44px rows, 14px labels and 264px.
 *
 * A comparison switch. It is a per-browser preference held in `localStorage`,
 * off by default, and the account menu is where it is turned on.
 */
export const DENSE_SIDEBAR_KEY = "otari.dashboard.sidebarDense"

const listeners = new Set<() => void>()
// What a write holds when storage refuses it, so the switch still works for the
// life of the page in a private window.
let held: boolean | null = null

function readStored(): boolean {
  if (held !== null) return held
  try {
    return window.localStorage.getItem(DENSE_SIDEBAR_KEY) === "1"
  } catch {
    return false
  }
}

function subscribe(listener: () => void): () => void {
  listeners.add(listener)
  // Another tab turning it on or off, or clearing storage (a null key). What
  // storage says now is the truth again, so a value held for a refused write
  // gives way to it.
  const onStorage = (event: StorageEvent) => {
    if (event.key !== null && event.key !== DENSE_SIDEBAR_KEY) return
    held = null
    listener()
  }
  window.addEventListener("storage", onStorage)
  return () => {
    listeners.delete(listener)
    window.removeEventListener("storage", onStorage)
  }
}

function write(next: boolean): void {
  try {
    window.localStorage.setItem(DENSE_SIDEBAR_KEY, next ? "1" : "0")
    held = null
  } catch {
    held = next
  }
  for (const listener of listeners) listener()
}

/** Whether the person asked for the dense sidebar, and the way to change it. */
export function useDenseSidebarPreference(): readonly [
  boolean,
  (next: boolean) => void,
] {
  const stored = useSyncExternalStore(subscribe, readStored, () => false)
  const set = useCallback((next: boolean) => write(next), [])
  return [stored, set]
}
