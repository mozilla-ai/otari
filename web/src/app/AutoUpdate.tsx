import { useRouter } from "@tanstack/react-router"
import { useEffect, useState } from "react"

import { useDashboardBuild } from "@/shared/api/deployment"

// How long after a stale-chunk reload another one is refused, so a chunk that is
// broken in the new build too surfaces as an error instead of a reload loop.
const STALE_CHUNK_RELOAD_GAP_MS = 10_000
const STALE_CHUNK_RELOAD_KEY = "otari:stale-chunk-reload"

// True once the gateway starts serving a different bundle than the one this tab
// loaded. The comparison is against the first build this tab ever saw rather
// than a value baked in at build time: "the app changed while you had it open"
// is the question worth asking, and a tab opened after a deploy is already
// current and must stay quiet.
export function useUpdateAvailable(): boolean {
  const { data } = useDashboardBuild()
  const [loadedBuild, setLoadedBuild] = useState<string | null>(null)

  // Set once, during render, so the first build is never compared against
  // itself; React re-renders with it before committing.
  if (data && loadedBuild === null) {
    setLoadedBuild(data.build)
  }
  return data != null && loadedBuild != null && data.build !== loadedBuild
}

// Picks up a new build without asking: once one is out, the next move to another
// page is a full load of that page rather than a client-side one. Leaving a page
// already discards its state, so nothing an operator is part-way through is lost,
// and a filter or tab change (same path) stays client-side. A plain reload is
// enough because the gateway serves index.html with no-store, and the sign-in is
// an HttpOnly session cookie that survives it.
export function AutoUpdate() {
  const updateAvailable = useUpdateAvailable()
  const router = useRouter()

  useEffect(() => {
    if (!updateAvailable) {
      return
    }
    return router.subscribe("onBeforeNavigate", ({ pathChanged }) => {
      if (!pathChanged) {
        return
      }
      // The history writes the new URL a microtask later; reloading before that
      // would reload the page being left.
      router.history.flush()
      window.location.reload()
    })
  }, [updateAvailable, router])

  return null
}

// A deploy replaces the hashed chunks, so a stale tab that opens a page it has not
// loaded yet asks for a file the gateway no longer has. Vite reports that as
// `vite:preloadError`; reloading fetches the new build, and the page with it.
export function reloadOnStaleChunk(): () => void {
  const onPreloadError = () => {
    try {
      const last = Number(sessionStorage.getItem(STALE_CHUNK_RELOAD_KEY))
      if (Date.now() - last < STALE_CHUNK_RELOAD_GAP_MS) {
        return
      }
      sessionStorage.setItem(STALE_CHUNK_RELOAD_KEY, String(Date.now()))
    } catch {
      // Without storage there is no way to tell a second failure from the first.
      return
    }
    window.location.reload()
  }
  window.addEventListener("vite:preloadError", onPreloadError)
  return () => window.removeEventListener("vite:preloadError", onPreloadError)
}
