import { useSyncExternalStore } from "react"

// The shell's own boundary (`AppShell`'s MOBILE_QUERY), which is Tailwind's `md`:
// below it the sidebar is already a drawer.
const QUERY = "(max-width: 767px)"

function subscribe(onChange: () => void): () => void {
  if (typeof window.matchMedia !== "function") return () => undefined
  const query = window.matchMedia(QUERY)
  query.addEventListener("change", onChange)
  return () => query.removeEventListener("change", onChange)
}

function getSnapshot(): boolean {
  return (
    typeof window.matchMedia === "function" && window.matchMedia(QUERY).matches
  )
}

/** Whether the viewport is phone-width, for a component that swaps structure rather than style. */
export function useIsMobile(): boolean {
  return useSyncExternalStore(subscribe, getSnapshot, () => false)
}
