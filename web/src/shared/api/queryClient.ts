import { QueryClient } from "@tanstack/react-query";

import { ApiError } from "./client";

// The app's one query-client configuration. It lives here rather than in
// src/app/provider.tsx because a test that mounts a feature needs the same client
// the app runs, and a feature (or its test) may not import the composition root.
export function createQueryClient(): QueryClient {
  return new QueryClient({
    defaultOptions: {
      queries: {
        refetchOnWindowFocus: false,
        retry: (failureCount, error) => {
          // Never retry auth failures; they won't fix themselves.
          if (error instanceof ApiError && (error.status === 401 || error.status === 403)) {
            return false;
          }
          return failureCount < 2;
        },
      },
    },
  });
}
