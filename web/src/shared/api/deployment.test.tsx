import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { renderHook, waitFor } from "@testing-library/react"
import type { ReactNode } from "react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import { API_ROOT } from "@/shared/api/client"
import { useDashboardBuild } from "@/shared/api/deployment"

function wrapper({ children }: { children: ReactNode }) {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  })
  return <QueryClientProvider client={client}>{children}</QueryClientProvider>
}

beforeEach(() => {
  vi.clearAllMocks()
})

afterEach(() => {
  vi.restoreAllMocks()
})

describe("useDashboardBuild", () => {
  it("asks the gateway's own root, not the API root", async () => {
    // The regression this exists for, and the reason it spies on `fetch` rather
    // than on a helper: what went wrong was the *choice of helper*. Routing this
    // through `apiFetch` prepends `API_ROOT` to a route the gateway mounts
    // beside the dashboard, so every poll 404s, and the hook swallows a failed
    // poll by design, so nothing reports it. Only the URL that reaches the
    // network says which helper was used.
    const fetchSpy = vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(JSON.stringify({ build: "abc123", version: "1.2.3" }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      }),
    )

    const { result } = renderHook(() => useDashboardBuild(), { wrapper })

    await waitFor(() => {
      expect(result.current.data).toEqual({ build: "abc123", version: "1.2.3" })
    })

    const requested = String(fetchSpy.mock.calls[0]?.[0])
    expect(requested).toBe("/dashboard-build.json")
    expect(requested.startsWith(API_ROOT)).toBe(false)
  })

  it("keeps the tab working when the poll fails", async () => {
    // The other half of why the bug was silent, asserted so it stays deliberate:
    // a failed check is not surfaced, because the tab is fine and the next poll
    // retries. That is right, and it is exactly what hid a permanent 404.
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response("nope", { status: 404 }),
    )

    const { result } = renderHook(() => useDashboardBuild(), { wrapper })

    await waitFor(() => {
      expect(result.current.isError).toBe(true)
    })
    expect(result.current.data).toBeUndefined()
  })
})
