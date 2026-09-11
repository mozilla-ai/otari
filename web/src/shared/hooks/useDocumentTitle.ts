import { useEffect } from "react"

export function useDocumentTitle(page?: string) {
  useEffect(() => {
    document.title = page ? `${page} · Otari` : "Otari"
    // An error boundary can unmount the page without mounting another title owner.
    return () => {
      document.title = "Otari"
    }
  }, [page])
}
