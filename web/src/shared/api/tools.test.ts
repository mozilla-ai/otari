import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { act, renderHook, waitFor } from "@testing-library/react"
import { createElement, type ReactNode } from "react"
import { afterEach, describe, expect, it, vi } from "vitest"

import {
  useCheckWorkspaceMcpServer,
  useDisconnectMcp,
  useMcpServerConnections,
  useMyMcpServerTools,
  useRemoveMcpServerConnection,
  useWorkspaceMcpServers,
} from "@/shared/api/tools"

afterEach(() => {
  vi.restoreAllMocks()
})

function setup() {
  const toolReads: string[] = []
  vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
    const url = String(input)
    if (url.endsWith("/my-tools")) toolReads.push(url)
    return Response.json({})
  })
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  })
  const wrapper = ({ children }: { children: ReactNode }) =>
    createElement(QueryClientProvider, { client }, children)
  return { wrapper, toolReads }
}

describe("a member's own tools on a server", () => {
  it("are read again after a check of the server", async () => {
    const { wrapper, toolReads } = setup()
    const { result } = renderHook(
      () => ({
        tools: useMyMcpServerTools("w-1", "s-1", true),
        check: useCheckWorkspaceMcpServer(),
      }),
      { wrapper },
    )
    await waitFor(() => expect(result.current.tools.isSuccess).toBe(true))
    expect(toolReads).toHaveLength(1)

    await act(() =>
      result.current.check.mutateAsync({ workspaceId: "w-1", serverId: "s-1" }),
    )

    await waitFor(() => expect(toolReads).toHaveLength(2))
  })

  it("are read again after the member disconnects", async () => {
    const { wrapper, toolReads } = setup()
    const { result } = renderHook(
      () => ({
        tools: useMyMcpServerTools("w-1", "s-1", true),
        disconnect: useDisconnectMcp(),
      }),
      { wrapper },
    )
    await waitFor(() => expect(result.current.tools.isSuccess).toBe(true))

    await act(() =>
      result.current.disconnect.mutateAsync({
        workspaceId: "w-1",
        serverId: "s-1",
      }),
    )

    await waitFor(() => expect(toolReads).toHaveLength(2))
  })
})

describe("the workspace's server list", () => {
  function listReads() {
    const reads: string[] = []
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
      const url = String(input)
      if ((init?.method ?? "GET") === "DELETE") {
        return new Response(null, { status: 204 })
      }
      if (url.includes("/mcp-servers?")) reads.push(url)
      return Response.json({ data: [], count: 0 })
    })
    const client = new QueryClient({
      defaultOptions: { queries: { retry: false } },
    })
    const wrapper = ({ children }: { children: ReactNode }) =>
      createElement(QueryClientProvider, { client }, children)
    return { wrapper, reads }
  }

  // Its Connections column counts connected users, so either disconnect moves it.
  it("is read again after the caller disconnects", async () => {
    const { wrapper, reads } = listReads()
    const { result } = renderHook(
      () => ({
        servers: useWorkspaceMcpServers("w-1"),
        disconnect: useDisconnectMcp(),
      }),
      { wrapper },
    )
    await waitFor(() => expect(result.current.servers.isSuccess).toBe(true))

    await act(() =>
      result.current.disconnect.mutateAsync({
        workspaceId: "w-1",
        serverId: "s-1",
      }),
    )

    await waitFor(() => expect(reads).toHaveLength(2))
  })

  it("is read again after a manager removes someone's connection", async () => {
    const { wrapper, reads } = listReads()
    const { result } = renderHook(
      () => ({
        servers: useWorkspaceMcpServers("w-1"),
        remove: useRemoveMcpServerConnection(),
      }),
      { wrapper },
    )
    await waitFor(() => expect(result.current.servers.isSuccess).toBe(true))

    await act(() =>
      result.current.remove.mutateAsync({
        workspaceId: "w-1",
        serverId: "s-1",
        userId: "alice",
      }),
    )

    await waitFor(() => expect(reads).toHaveLength(2))
  })
})

describe("a server's connected users", () => {
  it("keep the last page on screen while the next loads, but never another server's", async () => {
    const pending: Array<() => void> = []
    vi.spyOn(globalThis, "fetch").mockImplementation(
      (input) =>
        new Promise((resolve) => {
          const reply = () =>
            resolve(
              Response.json({
                data: [{ user_id: String(input) }],
                count: 60,
              }),
            )
          pending.push(reply)
        }),
    )
    const client = new QueryClient({
      defaultOptions: { queries: { retry: false } },
    })
    const wrapper = ({ children }: { children: ReactNode }) =>
      createElement(QueryClientProvider, { client }, children)
    const { result, rerender } = renderHook(
      ({ serverId, page }: { serverId: string; page: number }) =>
        useMcpServerConnections("w-1", serverId, { page, pageSize: 25 }, true),
      { wrapper, initialProps: { serverId: "s-1", page: 0 } },
    )
    await waitFor(() => expect(pending).toHaveLength(1))
    act(() => pending[0]())
    await waitFor(() => expect(result.current.isSuccess).toBe(true))

    rerender({ serverId: "s-1", page: 1 })
    expect(result.current.data?.data).toHaveLength(1)
    expect(result.current.isPlaceholderData).toBe(true)

    rerender({ serverId: "s-2", page: 1 })
    expect(result.current.data).toBeUndefined()
  })
})
