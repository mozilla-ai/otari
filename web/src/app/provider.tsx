import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import type { ReactNode } from "react"
import { useState } from "react"
import { AuthProvider } from "@/features/auth/AuthContext"
import { defaultQueryRetry } from "@/shared/api/client"
import { ThemeProvider } from "@/shared/hooks/useTheme"

export function Provider({ children }: { children: ReactNode }) {
  const [queryClient] = useState(
    () =>
      new QueryClient({
        defaultOptions: {
          queries: {
            refetchOnWindowFocus: false,
            retry: defaultQueryRetry,
          },
        },
      }),
  )

  return (
    <QueryClientProvider client={queryClient}>
      {/* Outside the auth gate: the sign-in screen is painted in the operator's
          chosen theme too, and the preference is a browser one rather than
          anything the session owns. */}
      <ThemeProvider>
        <AuthProvider>{children}</AuthProvider>
      </ThemeProvider>
    </QueryClientProvider>
  )
}
