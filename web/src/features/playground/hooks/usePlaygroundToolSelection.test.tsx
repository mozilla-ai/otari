import { act, renderHook } from "@testing-library/react"
import { describe, expect, it } from "vitest"

import type { PlaygroundTools } from "@/client"

import {
  keepAttachableMcpIds,
  usePlaygroundToolSelection,
} from "./usePlaygroundToolSelection"

function tools(overrides: Partial<PlaygroundTools> = {}): PlaygroundTools {
  return {
    web_search: { configured: true, enabled: true, reason: null },
    web_fetch: { configured: true, enabled: true, reason: null },
    code_execution: { configured: true, enabled: true, reason: null },
    files: { configured: true, enabled: true, reason: null },
    mcp_servers: [
      {
        id: "srv-1",
        name: "Docs",
        purpose_hint: null,
        enabled: true,
        sign_in_required: false,
      },
      {
        id: "srv-2",
        name: "Tickets",
        purpose_hint: null,
        enabled: true,
        sign_in_required: false,
      },
    ],
    ...overrides,
  }
}

describe("keepAttachableMcpIds", () => {
  it("drops an id the workspace no longer offers", () => {
    expect(keepAttachableMcpIds(["a", "b"], new Set(["a"]))).toEqual(["a"])
  })

  it("returns the very same array when nothing changed", () => {
    // Identity, not equality, and the test is written as `toBe` because that is
    // the whole contract: the reconciliation effect calls this on every render,
    // and a copy would commit new state, re-render, and run the effect again.
    // An earlier version of this returned a copy and hung the page.
    const selected = ["a"]
    expect(keepAttachableMcpIds(selected, new Set(["a"]))).toBe(selected)
  })
})

describe("usePlaygroundToolSelection", () => {
  it("attaches nothing by default", () => {
    const { result } = renderHook(() => usePlaygroundToolSelection(tools()))
    expect(result.current.selection).toEqual({
      isWebSearchOn: false,
      isWebFetchOn: false,
      isCodeExecutionOn: false,
      mcpServerIds: [],
    })
  })

  it("keeps web search, code execution and MCP servers attached together", () => {
    const { result } = renderHook(() => usePlaygroundToolSelection(tools()))
    act(() => result.current.toggleWebSearch(true))
    act(() => result.current.toggleCodeExecution(true))
    act(() => result.current.toggleMcpServer("srv-1", true))
    expect(result.current.selection).toEqual({
      isWebSearchOn: true,
      isWebFetchOn: false,
      isCodeExecutionOn: true,
      mcpServerIds: ["srv-1"],
    })

    act(() => result.current.toggleCodeExecution(false))
    act(() => result.current.toggleMcpServer("srv-1", false))
    expect(result.current.selection).toEqual({
      isWebSearchOn: true,
      isWebFetchOn: false,
      isCodeExecutionOn: false,
      mcpServerIds: [],
    })
  })

  it("attaches web fetch without web search, and both together", () => {
    const { result } = renderHook(() => usePlaygroundToolSelection(tools()))
    act(() => result.current.toggleWebFetch(true))
    expect(result.current.selection).toMatchObject({
      isWebSearchOn: false,
      isWebFetchOn: true,
    })

    act(() => result.current.toggleWebSearch(true))
    expect(result.current.selection).toMatchObject({
      isWebSearchOn: true,
      isWebFetchOn: true,
    })

    act(() => result.current.toggleWebFetch(false))
    expect(result.current.selection).toMatchObject({
      isWebSearchOn: true,
      isWebFetchOn: false,
    })
  })

  it("attaches several MCP servers at once", () => {
    const { result } = renderHook(() => usePlaygroundToolSelection(tools()))
    act(() => result.current.toggleMcpServer("srv-1", true))
    act(() => result.current.toggleMcpServer("srv-2", true))
    expect(result.current.selectedMcpIds).toEqual(["srv-1", "srv-2"])
  })

  it("drops a tool the workspace turns off while the page is open", () => {
    // Otherwise the next send fails for a reason nothing on screen explains.
    const { result, rerender } = renderHook(
      ({ available }: { available: PlaygroundTools }) =>
        usePlaygroundToolSelection(available),
      { initialProps: { available: tools() } },
    )
    act(() => result.current.toggleWebSearch(true))
    expect(result.current.isWebSearchOn).toBe(true)

    rerender({
      available: tools({
        web_search: {
          configured: true,
          enabled: false,
          reason: "Turned off for this workspace.",
        },
      }),
    })
    expect(result.current.isWebSearchOn).toBe(false)
  })

  it("drops web fetch when the workspace turns web access off while the page is open", () => {
    const { result, rerender } = renderHook(
      ({ available }: { available: PlaygroundTools }) =>
        usePlaygroundToolSelection(available),
      { initialProps: { available: tools() } },
    )
    act(() => result.current.toggleWebFetch(true))
    expect(result.current.isWebFetchOn).toBe(true)

    rerender({
      available: tools({
        web_fetch: {
          configured: true,
          enabled: false,
          reason: "Turned off for this workspace.",
        },
      }),
    })
    expect(result.current.isWebFetchOn).toBe(false)
  })

  it("declares nothing while the current workspace's tools are unknown", () => {
    // A workspace switch clears the read until the new workspace answers, and a
    // choice made in the old one must not ride a message sent meanwhile.
    const { result, rerender } = renderHook(
      ({ available }: { available: PlaygroundTools | undefined }) =>
        usePlaygroundToolSelection(available),
      { initialProps: { available: tools() as PlaygroundTools | undefined } },
    )
    act(() => result.current.toggleWebSearch(true))
    act(() => result.current.toggleWebFetch(true))
    act(() => result.current.toggleCodeExecution(true))
    act(() => result.current.toggleMcpServer("srv-1", true))
    expect(result.current.selection).toEqual({
      isWebSearchOn: true,
      isWebFetchOn: true,
      isCodeExecutionOn: true,
      mcpServerIds: ["srv-1"],
    })

    rerender({ available: undefined })
    expect(result.current.selection).toEqual({
      isWebSearchOn: false,
      isWebFetchOn: false,
      isCodeExecutionOn: false,
      mcpServerIds: [],
    })
    // The chips read the same state, so they show nothing either.
    expect(result.current.isWebFetchOn).toBe(false)
    expect(result.current.selectedMcpIds).toEqual([])

    // The choice comes back once the workspace's read allows it.
    rerender({ available: tools() })
    expect(result.current.isWebFetchOn).toBe(true)
    expect(result.current.selectedMcpIds).toEqual(["srv-1"])
  })

  it("drops an MCP server that is deleted while the page is open", () => {
    const { result, rerender } = renderHook(
      ({ available }: { available: PlaygroundTools }) =>
        usePlaygroundToolSelection(available),
      { initialProps: { available: tools() } },
    )
    act(() => result.current.toggleMcpServer("srv-2", true))
    expect(result.current.selectedMcpIds).toEqual(["srv-2"])

    rerender({
      available: tools({
        mcp_servers: [
          {
            id: "srv-1",
            name: "Docs",
            purpose_hint: null,
            enabled: true,
            sign_in_required: false,
          },
        ],
      }),
    })
    expect(result.current.selectedMcpIds).toEqual([])
  })

  it("never sends an MCP server the latest read leaves out, even before it reconciles", () => {
    // Send can run on the render a refreshed read arrives in, before the
    // reconciliation effect commits, so every render's selection is checked.
    const sent: string[][] = []
    const { result, rerender } = renderHook(
      ({ available }: { available: PlaygroundTools }) => {
        const state = usePlaygroundToolSelection(available)
        sent.push(state.selection.mcpServerIds)
        return state
      },
      { initialProps: { available: tools() } },
    )
    act(() => result.current.toggleMcpServer("srv-1", true))
    act(() => result.current.toggleMcpServer("srv-2", true))
    expect(result.current.selection.mcpServerIds).toEqual(["srv-1", "srv-2"])
    sent.length = 0

    rerender({
      available: tools({
        mcp_servers: [
          {
            id: "srv-1",
            name: "Docs",
            purpose_hint: null,
            enabled: true,
            sign_in_required: false,
          },
        ],
      }),
    })
    expect(sent.length).toBeGreaterThan(0)
    expect(sent.every((ids) => ids.length === 1 && ids[0] === "srv-1")).toBe(
      true,
    )
  })

  it("drops an MCP server the caller is disconnected from while the page is open", () => {
    const { result, rerender } = renderHook(
      ({ available }: { available: PlaygroundTools }) =>
        usePlaygroundToolSelection(available),
      { initialProps: { available: tools() } },
    )
    act(() => result.current.toggleMcpServer("srv-1", true))
    act(() => result.current.toggleMcpServer("srv-2", true))

    rerender({
      available: tools({
        mcp_servers: [
          {
            id: "srv-1",
            name: "Docs",
            purpose_hint: null,
            enabled: true,
            sign_in_required: true,
          },
          {
            id: "srv-2",
            name: "Tickets",
            purpose_hint: null,
            enabled: true,
            sign_in_required: false,
          },
        ],
      }),
    })
    expect(result.current.selectedMcpIds).toEqual(["srv-2"])
  })

  it("keeps a selection while the availability read has not answered", () => {
    // An undefined read means the tools are unknown, not that nothing is
    // attachable: the choice is not sent meanwhile, and is not dropped either.
    const { result, rerender } = renderHook(
      ({ available }: { available: PlaygroundTools | undefined }) =>
        usePlaygroundToolSelection(available),
      { initialProps: { available: undefined as PlaygroundTools | undefined } },
    )
    act(() => result.current.toggleWebSearch(true))
    expect(result.current.selection.isWebSearchOn).toBe(false)

    rerender({ available: tools() })
    expect(result.current.isWebSearchOn).toBe(true)
  })
})
