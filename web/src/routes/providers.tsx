import { createFileRoute } from "@tanstack/react-router";

import { ProvidersPage } from "@/pages/ProvidersPage";

export const Route = createFileRoute("/providers")({
  component: ProvidersPage,
});
