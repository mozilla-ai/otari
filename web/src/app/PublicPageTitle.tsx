import type { ReactNode } from "react"
import { useDocumentTitle } from "@/shared/hooks/useDocumentTitle"

/** Gives public pages outside the authenticated router a mounted title owner. */
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
