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
    code_execution: { configured: true, enabled: true, reason: null },
    mcp_servers: [
      { id: "srv-1", name: "Docs", purpose_hint: null, enabled: true },
      { id: "srv-2", name: "Tickets", purpose_hint: null, enabled: true },
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
      isCodeExecutionOn: false,
      mcpServerIds: [],
    })
  })

  it("clears the others when one is turned on", () => {
    // One request carries one tool mode, so a selection the gateway would
    // refuse is not reachable from the menu.
    const { result } = renderHook(() => usePlaygroundToolSelection(tools()))
    act(() => result.current.toggleWebSearch(true))
    act(() => result.current.toggleMcpServer("srv-1", true))
    expect(result.current.isWebSearchOn).toBe(false)
    expect(result.current.selectedMcpIds).toEqual(["srv-1"])

    act(() => result.current.toggleCodeExecution(true))
    expect(result.current.selectedMcpIds).toEqual([])
    expect(result.current.isCodeExecutionOn).toBe(true)
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
          { id: "srv-1", name: "Docs", purpose_hint: null, enabled: true },
        ],
      }),
    })
    expect(result.current.selectedMcpIds).toEqual([])
  })

  it("keeps a selection while the availability read has not answered", () => {
    // An undefined read is "not known yet", not "nothing is attachable".
    const { result } = renderHook(() => usePlaygroundToolSelection(undefined))
    act(() => result.current.toggleWebSearch(true))
    expect(result.current.isWebSearchOn).toBe(true)
  })
})
