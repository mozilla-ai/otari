import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { act, renderHook, waitFor } from "@testing-library/react"
import { createElement, type ReactNode } from "react"
import { afterEach, expect, it, vi } from "vitest"

import type { McpServerRequest } from "@/client"
import {
  useAcknowledgeMcpServerRequest,
  useMcpServerRequests,
  useMyMcpServerRequests,
} from "@/shared/api/mcpServerRequests"

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

it("puts back only the answer whose dismissal failed, where it was, when the list cannot be read again", async () => {
  const answer = (id: string) => ({ id }) as McpServerRequest
  let reads = 0
  let finishFirst: () => void = () => {}
  vi.spyOn(globalThis, "fetch").mockImplementation((input) => {
    const url = String(input)
    if (url.endsWith("/a/acknowledge")) {
      return new Promise((resolve) => {
        finishFirst = () => resolve(new Response(null, { status: 204 }))
      })
    }
    if (url.endsWith("/b/acknowledge")) {
      return Promise.resolve(Response.json({ detail: "No" }, { status: 500 }))
    }
    reads += 1
    return Promise.resolve(
      reads === 1
        ? Response.json({ data: ["a", "b", "c"].map(answer) })
        : Response.json({ detail: "Down" }, { status: 503 }),
    )
  })
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  })
  const wrapper = ({ children }: { children: ReactNode }) =>
    createElement(QueryClientProvider, { client }, children)
  const { result } = renderHook(
    () => ({
      mine: useMyMcpServerRequests("w-1"),
      acknowledge: useAcknowledgeMcpServerRequest(),
    }),
    { wrapper },
  )
  await waitFor(() => expect(result.current.mine.isSuccess).toBe(true))
  const ids = () => result.current.mine.data?.data.map((entry) => entry.id)

  act(() =>
    result.current.acknowledge.mutate({ workspaceId: "w-1", requestId: "a" }),
  )
  await waitFor(() => expect(ids()).toEqual(["b", "c"]))
  act(() =>
    result.current.acknowledge.mutate({ workspaceId: "w-1", requestId: "b" }),
  )
  // b is refused while a is still on its way: b comes back, a stays gone.
  await waitFor(() => expect(result.current.acknowledge.isError).toBe(true))
  expect(ids()).toEqual(["b", "c"])

  act(() => finishFirst())
  // The read after the last dismissal fails, which leaves the list as it is.
  await waitFor(() => expect(result.current.mine.isError).toBe(true))
  expect(ids()).toEqual(["b", "c"])
})
