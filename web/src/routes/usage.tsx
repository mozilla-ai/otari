import { createFileRoute } from "@tanstack/react-router";

import { UsagePage } from "@/pages/UsagePage";

export const Route = createFileRoute("/usage")({
  component: UsagePage,
});
