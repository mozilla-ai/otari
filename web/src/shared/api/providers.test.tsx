import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { renderHook, waitFor } from "@testing-library/react"
import type { ReactNode } from "react"
import { afterEach, describe, expect, it, vi } from "vitest"

import { useCreateStoredProvider } from "@/shared/api/providers"
import { CATALOG, MODELS, PROVIDERS } from "@/shared/api/queryKeys"

// A provider added on Providers changes what the Models page lists, and the
// grouped catalog is its own query beside the flat model list: leaving it out
// of the invalidation left Models on its cached empty answer until a reload.
describe("useCreateStoredProvider", () => {
  afterEach(() => {
    vi.restoreAllMocks()
  })

  it("invalidates the grouped catalog along with the model list", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(JSON.stringify({ id: "p1", instance: "openai" }), {
        status: 201,
        headers: { "Content-Type": "application/json" },
      }),
    )
    const client = new QueryClient({
      defaultOptions: { queries: { retry: false } },
    })
    const invalidated = vi.spyOn(client, "invalidateQueries")
    const wrapper = ({ children }: { children: ReactNode }) => (
      <QueryClientProvider client={client}>{children}</QueryClientProvider>
    )

    const { result } = renderHook(() => useCreateStoredProvider(), { wrapper })
    result.current.mutate({
      instance: "openai",
      provider_type: "openai",
      api_key: "sk-test",
    } as never)

    await waitFor(() => expect(result.current.isSuccess).toBe(true))
    const keys = invalidated.mock.calls.map(([options]) => options?.queryKey)
    expect(keys).toContainEqual([CATALOG])
    expect(keys).toContainEqual([MODELS])
    expect(keys).toContainEqual([PROVIDERS])
  })
})
