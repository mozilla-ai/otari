import { createFileRoute } from "@tanstack/react-router";

import { KeysPage } from "@/pages/KeysPage";

export const Route = createFileRoute("/keys")({
  component: KeysPage,
});
