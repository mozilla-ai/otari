import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { act, renderHook, waitFor } from "@testing-library/react"
import { createElement, type ReactNode } from "react"
import { afterEach, describe, expect, it, vi } from "vitest"

import {
  useCheckWorkspaceMcpServer,
  useDisconnectMcp,
  useMyMcpServerTools,
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
