import { createFileRoute } from "@tanstack/react-router";

import { ModelsPage } from "@/pages/ModelsPage";

export const Route = createFileRoute("/models")({
  component: ModelsPage,
});
