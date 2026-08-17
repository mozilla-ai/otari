import { createFileRoute } from "@tanstack/react-router";

import { DocsPage } from "@/features/docs/DocsPage";

export const Route = createFileRoute("/docs")({
  component: DocsPage,
});
