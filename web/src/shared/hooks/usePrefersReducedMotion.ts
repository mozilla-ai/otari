import { useEffect, useState } from "react"

const QUERY = "(prefers-reduced-motion: reduce)"

function readPreference(): boolean {
  if (
    typeof window === "undefined" ||
    typeof window.matchMedia !== "function"
  ) {
    return false
  }
  return window.matchMedia(QUERY).matches
}

/**
 * Whether this reader has asked the system for less motion.
 *
 * Almost nothing here needs it: `motion-reduce:` covers a CSS transition or
 * animation without a component knowing anything. This is for the case CSS
 * cannot reach, which today is a canvas that animates itself and has to be told
 * to stop.
 *
 * Subscribed rather than read once, because the preference can change while the
 * page is open (a system setting, or a browser's own toggle), and guarded for
 * the environments without `matchMedia` the way `useTheme` is.
 */
export function usePrefersReducedMotion(): boolean {
  const [prefersReduced, setPrefersReduced] = useState<boolean>(readPreference)

  useEffect(() => {
    if (
      typeof window === "undefined" ||
      typeof window.matchMedia !== "function"
    ) {
      return
    }
    const query = window.matchMedia(QUERY)
    const onChange = (event: MediaQueryListEvent) =>
      setPrefersReduced(event.matches)
    setPrefersReduced(query.matches)
    // Safari below 14 has only the deprecated pair, which the shell's own media
    // query already supports.
    if (query.addEventListener) {
      query.addEventListener("change", onChange)
      return () => query.removeEventListener("change", onChange)
    }
    query.addListener(onChange)
    return () => query.removeListener(onChange)
  }, [])

  return prefersReduced
}
