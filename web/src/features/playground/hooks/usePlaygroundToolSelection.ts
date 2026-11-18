import { useCallback, useEffect, useMemo, useState } from "react"

import type { PlaygroundTools } from "@/client"
import type { ToolSelection } from "@/shared/api/playground"

/**
 * Drop any selected MCP id the workspace no longer offers.
 *
 * Returns the same array reference when nothing changed, which is what keeps the
 * reconciliation effect below from setting state on every render: a new array of
 * equal contents is a new reference, and `setState` with one re-renders, which
 * re-runs the effect.
 */
export function keepAttachableMcpIds(
  selected: string[],
  attachable: ReadonlySet<string>,
): string[] {
  const next = selected.filter((id) => attachable.has(id))
  // The *same* array, not a copy of it. A copy is a new reference, so
  // `setState` commits, the component re-renders, the effect runs again, and
  // the page hangs. That is not a hypothetical: it crashed the worker running
  // this hook's own test before this line said `selected`.
  return next.length === selected.length ? selected : next
}

/** The MCP servers a message may attach: on for the workspace, and signed in where the server needs it. */
function attachableMcpIds(tools: PlaygroundTools): Set<string> {
  return new Set(
    tools.mcp_servers
      // One the caller is not signed in to would refuse the message.
      .filter((server) => server.enabled && !server.sign_in_required)
      .map((server) => server.id),
  )
}

export interface PlaygroundToolSelection {
  isWebSearchOn: boolean
  isWebFetchOn: boolean
  isCodeExecutionOn: boolean
  selectedMcpIds: string[]
  toggleWebSearch: (isOn: boolean) => void
  toggleWebFetch: (isOn: boolean) => void
  toggleCodeExecution: (isOn: boolean) => void
  toggleMcpServer: (id: string, isOn: boolean) => void
  selection: ToolSelection
}

/**
 * What is attached to the next message.
 *
 * Any mix is allowed: web search, web fetch, code execution and any number of MCP servers
 * ride one request, and the gateway runs them in a single tool loop.
 *
 * The reconciliation effect is the non-obvious half. A tool can stop being
 * attachable while the page is open (an operator disables it for the workspace,
 * an MCP server is deleted), and a selection that outlives its tool is a request
 * that fails on send for a reason nothing on screen explains. So the selection
 * is narrowed to what the latest availability read allows, every time that read
 * changes.
 */
export function usePlaygroundToolSelection(
  tools: PlaygroundTools | undefined,
): PlaygroundToolSelection {
  const [isWebSearchOn, setWebSearchOn] = useState(false)
  const [isWebFetchOn, setWebFetchOn] = useState(false)
  const [isCodeExecutionOn, setCodeExecutionOn] = useState(false)
  const [selectedMcpIds, setSelectedMcpIds] = useState<string[]>([])

  const toggleWebSearch = useCallback((isOn: boolean) => {
    setWebSearchOn(isOn)
  }, [])

  const toggleWebFetch = useCallback((isOn: boolean) => {
    setWebFetchOn(isOn)
  }, [])

  const toggleCodeExecution = useCallback((isOn: boolean) => {
    setCodeExecutionOn(isOn)
  }, [])

  const toggleMcpServer = useCallback((id: string, isOn: boolean) => {
    setSelectedMcpIds((prev) => {
      if (!isOn) return prev.filter((existing) => existing !== id)
      return prev.includes(id) ? prev : [...prev, id]
    })
  }, [])

  useEffect(() => {
    if (!tools) return
    if (isWebSearchOn && !tools.web_search.enabled) setWebSearchOn(false)
    if (isWebFetchOn && !tools.web_fetch.enabled) setWebFetchOn(false)
    if (isCodeExecutionOn && !tools.code_execution.enabled) {
      setCodeExecutionOn(false)
    }
    const attachable = attachableMcpIds(tools)
    setSelectedMcpIds((prev) => keepAttachableMcpIds(prev, attachable))
  }, [tools, isWebSearchOn, isWebFetchOn, isCodeExecutionOn])

  // What is attached is what the current workspace's read allows: the chips, the
  // menu and the request all show it, so a choice made in another workspace is
  // neither sent nor shown while this one's read loads. The choice itself is
  // kept, and shows again once the read allows it.
  const effectiveWebSearch =
    isWebSearchOn && (tools?.web_search.enabled ?? false)
  const effectiveWebFetch = isWebFetchOn && (tools?.web_fetch.enabled ?? false)
  const effectiveCodeExecution =
    isCodeExecutionOn && (tools?.code_execution.enabled ?? false)
  const effectiveMcpIds = useMemo(
    () =>
      tools
        ? keepAttachableMcpIds(selectedMcpIds, attachableMcpIds(tools))
        : [],
    [tools, selectedMcpIds],
  )
  const selection = useMemo<ToolSelection>(
    () => ({
      isWebSearchOn: effectiveWebSearch,
      isWebFetchOn: effectiveWebFetch,
      isCodeExecutionOn: effectiveCodeExecution,
      mcpServerIds: effectiveMcpIds,
    }),
    [
      effectiveWebSearch,
      effectiveWebFetch,
      effectiveCodeExecution,
      effectiveMcpIds,
    ],
  )

  return {
    isWebSearchOn: effectiveWebSearch,
    isWebFetchOn: effectiveWebFetch,
    isCodeExecutionOn: effectiveCodeExecution,
    selectedMcpIds: effectiveMcpIds,
    toggleWebSearch,
    toggleWebFetch,
    toggleCodeExecution,
    toggleMcpServer,
    selection,
  }
}
