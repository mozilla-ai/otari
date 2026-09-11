import { useEffect } from "react"

export function useDocumentTitle(page?: string) {
  useEffect(() => {
    document.title = page ? `${page} · Otari` : "Otari"
    return () => {
      document.title = "Otari"
    }
  }, [page])
}
