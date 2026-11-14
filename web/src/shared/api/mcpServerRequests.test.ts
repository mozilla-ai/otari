import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { act, renderHook, waitFor } from "@testing-library/react"
import { createElement, type ReactNode } from "react"
import { afterEach, expect, it, vi } from "vitest"

import { useMcpServerRequests } from "@/shared/api/mcpServerRequests"

afterEach(() => {
  vi.restoreAllMocks()
})

it("keeps the last page of requests while the next loads, but never another workspace's", async () => {
  const urls: string[] = []
  const pending: Array<() => void> = []
  vi.spyOn(globalThis, "fetch").mockImplementation(
    (input) =>
      new Promise((resolve) => {
        urls.push(String(input))
        pending.push(() =>
          resolve(Response.json({ data: [], count: 60, server_count: 30 })),
        )
      }),
  )
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  })
  const wrapper = ({ children }: { children: ReactNode }) =>
    createElement(QueryClientProvider, { client }, children)
  const { result, rerender } = renderHook(
    ({ workspaceId, page }: { workspaceId: string; page: number }) =>
      useMcpServerRequests(workspaceId, { page, pageSize: 25 }, true),
    { wrapper, initialProps: { workspaceId: "w-1", page: 0 } },
  )
  await waitFor(() => expect(pending).toHaveLength(1))
  act(() => pending[0]())
  await waitFor(() => expect(result.current.isSuccess).toBe(true))

  rerender({ workspaceId: "w-1", page: 1 })
  expect(result.current.isPlaceholderData).toBe(true)
  await waitFor(() => expect(urls).toHaveLength(2))
  expect(urls[1]).toMatch(
    /\/workspaces\/w-1\/mcp-server-requests\?skip=25&limit=25$/,
  )

  rerender({ workspaceId: "w-2", page: 1 })
  expect(result.current.data).toBeUndefined()
})
