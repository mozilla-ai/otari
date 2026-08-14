import { createFileRoute } from "@tanstack/react-router";

import { ActivityPage } from "@/pages/ActivityPage";

export const Route = createFileRoute("/activity")({
  component: ActivityPage,
});
