import { createFileRoute } from "@tanstack/react-router";

import { OverviewIndex } from "@/pages/OverviewPage";

export const Route = createFileRoute("/")({
  component: OverviewIndex,
});
