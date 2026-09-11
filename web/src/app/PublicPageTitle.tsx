import type { ReactNode } from "react"
import { useDocumentTitle } from "@/shared/hooks/useDocumentTitle"

export function PublicPageTitle({
  page,
  children,
}: {
  page?: string
  children: ReactNode
}) {
  useDocumentTitle(page)
  return children
}
