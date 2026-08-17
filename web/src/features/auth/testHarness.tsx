import type { ReactNode } from "react";

import { QueryHarness } from "@/shared/test/query";

import { AuthProvider } from "./AuthContext";

/**
 * What `src/app/provider.tsx` mounts, assembled here instead.
 *
 * The auth tests exercise `AuthProvider` against a live query client, which is the
 * pair the app's `Provider` composes. A feature may not import the composition root
 * (see biome.jsonc), and it should not have to: the shell's job is to decide what
 * else goes in the tree, and none of the rest of it is under test here.
 */
export function AuthHarness({ children }: { children: ReactNode }) {
  return (
    <QueryHarness>
      <AuthProvider>{children}</AuthProvider>
    </QueryHarness>
  );
}
