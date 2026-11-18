import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { renderHook, waitFor } from "@testing-library/react"
import type { ReactNode } from "react"
import { afterEach, describe, expect, it, vi } from "vitest"
import { useCanCreateKey } from "@/features/organization/access"
import { API_ROOT } from "@/shared/api/client"

function wrapper({ children }: { children: ReactNode }) {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  })
  return <QueryClientProvider client={client}>{children}</QueryClientProvider>
}

describe("useCanCreateKey", () => {
  afterEach(() => {
    vi.restoreAllMocks()
  })

  it("offers nothing to someone in no organization", async () => {
    // A 404 on the context is the answer, not a wait: the server refuses
    // them a key, so the control is not offered.
    const fetch = vi
      .spyOn(globalThis, "fetch")
      .mockImplementation(async () => new Response("{}", { status: 404 }))
    const { result } = renderHook(() => useCanCreateKey(), { wrapper })
    await waitFor(() =>
      expect(
        fetch.mock.calls.some(([input]) =>
          String(input).startsWith(`${API_ROOT}/organizations/me`),
        ),
      ).toBe(true),
    )
    await waitFor(() => expect(result.current).toBe(false))
  })
})
