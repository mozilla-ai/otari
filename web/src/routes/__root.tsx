import { createRootRoute } from "@tanstack/react-router";

import { AppShell } from "@/app/AppShell";
import { validateSearch } from "@/shared/lib/search";

export const Route = createRootRoute({
  component: AppShell,
  validateSearch,
});
