import { QueryClientProvider } from "@tanstack/react-query";
import { useState } from "react";
import type { ReactNode } from "react";

import { createQueryClient } from "@/shared/api/queryClient";

/**
 * A fresh query client per render, configured as the app configures it.
 *
 * For a test that mounts a feature's own provider (auth is the one that has one):
 * the app's `Provider` composes this with `AuthProvider`, and a feature may not
 * import the composition root, so a feature test wraps the two itself. Most page
 * tests need neither, because they mock `@/shared/api/hooks` outright.
 */
export function QueryHarness({ children }: { children: ReactNode }) {
  const [client] = useState(createQueryClient);
  return <QueryClientProvider client={client}>{children}</QueryClientProvider>;
}
