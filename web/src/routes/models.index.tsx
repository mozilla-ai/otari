import { createFileRoute } from "@tanstack/react-router"

import { ModelCatalogPage } from "@/features/models/ModelCatalogPage"

export const Route = createFileRoute("/models/")({
  component: ModelCatalogPage,
})
