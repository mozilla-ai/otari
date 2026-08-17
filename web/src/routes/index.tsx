import { createFileRoute } from "@tanstack/react-router";

import { OverviewIndex } from "@/features/overview/OverviewPage";

export const Route = createFileRoute("/")({
  component: OverviewIndex,
});
