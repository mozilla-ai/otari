import { createFileRoute } from "@tanstack/react-router";

import { ToolsGuardrailsPage } from "@/pages/ToolsGuardrailsPage";

export const Route = createFileRoute("/tools")({
  component: ToolsGuardrailsPage,
});
