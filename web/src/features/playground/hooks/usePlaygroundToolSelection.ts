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

export interface PlaygroundToolSelection {
  isWebSearchOn: boolean
  isCodeExecutionOn: boolean
  selectedMcpIds: string[]
  toggleWebSearch: (isOn: boolean) => void
  toggleCodeExecution: (isOn: boolean) => void
  toggleMcpServer: (id: string, isOn: boolean) => void
  selection: ToolSelection
}

/**
 * What is attached to the next message.
 *
 * The three are mutually exclusive, which is a fact about the gateway rather
 * than a UI preference: one request carries one tool mode, so enabling any of
 * them clears the others instead of letting somebody build a selection the
 * request would refuse.
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
  const [isCodeExecutionOn, setCodeExecutionOn] = useState(false)
  const [selectedMcpIds, setSelectedMcpIds] = useState<string[]>([])

  const toggleWebSearch = useCallback((isOn: boolean) => {
    setWebSearchOn(isOn)
    if (isOn) {
      setCodeExecutionOn(false)
      setSelectedMcpIds([])
    }
  }, [])

  const toggleCodeExecution = useCallback((isOn: boolean) => {
    setCodeExecutionOn(isOn)
    if (isOn) {
      setWebSearchOn(false)
      setSelectedMcpIds([])
    }
  }, [])

  const toggleMcpServer = useCallback((id: string, isOn: boolean) => {
    if (!isOn) {
      setSelectedMcpIds((prev) => prev.filter((existing) => existing !== id))
      return
    }
    setWebSearchOn(false)
    setCodeExecutionOn(false)
    setSelectedMcpIds((prev) => (prev.includes(id) ? prev : [...prev, id]))
  }, [])

  useEffect(() => {
    if (!tools) return
    if (isWebSearchOn && !tools.web_search.enabled) setWebSearchOn(false)
    if (isCodeExecutionOn && !tools.code_execution.enabled) {
      setCodeExecutionOn(false)
    }
    const attachable = new Set(
      tools.mcp_servers.filter((server) => server.enabled).map((s) => s.id),
    )
    setSelectedMcpIds((prev) => keepAttachableMcpIds(prev, attachable))
  }, [tools, isWebSearchOn, isCodeExecutionOn])

  const selection = useMemo<ToolSelection>(
    () => ({ isWebSearchOn, isCodeExecutionOn, mcpServerIds: selectedMcpIds }),
    [isWebSearchOn, isCodeExecutionOn, selectedMcpIds],
  )

  return {
    isWebSearchOn,
    isCodeExecutionOn,
    selectedMcpIds,
    toggleWebSearch,
    toggleCodeExecution,
    toggleMcpServer,
    selection,
  }
}
