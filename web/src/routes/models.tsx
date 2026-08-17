import { createFileRoute } from "@tanstack/react-router";

import { ModelsPage } from "@/features/models/ModelsPage";

export const Route = createFileRoute("/models")({
  component: ModelsPage,
});
