import { createFileRoute } from "@tanstack/react-router"

import { WebSearchPage } from "@/features/webSearch/WebSearchPage"

export const Route = createFileRoute("/admin/web-search")({
  component: WebSearchPage,
})
