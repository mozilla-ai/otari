import { useEffect } from "react"

/**
 * Run `onRestore` when the browser serves this page from its back/forward cache.
 *
 * A restored page keeps the state it held when the tab navigated away, so a control left
 * mid-flight comes back mid-flight and stays disabled. A `pageshow` whose `persisted` is
 * true is the only signal that tells a restore apart from an ordinary load.
 */
export function useBackForwardCacheRestore(onRestore: () => void): void {
  useEffect(() => {
    if (typeof window === "undefined") {
      return
    }
    const onPageShow = (event: PageTransitionEvent) => {
      if (event.persisted) {
        onRestore()
      }
    }
    window.addEventListener("pageshow", onPageShow)
    return () => window.removeEventListener("pageshow", onPageShow)
  }, [onRestore])
}
