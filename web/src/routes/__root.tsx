import { createRootRoute } from "@tanstack/react-router";

import { AppShell } from "@/components/AppShell";
import { validateSearch } from "@/lib/search";

export const Route = createRootRoute({
  component: AppShell,
  validateSearch,
});
