import { createFileRoute } from "@tanstack/react-router";

import { RoutingPage } from "@/pages/RoutingPage";

export const Route = createFileRoute("/routing")({
  component: RoutingPage,
});
