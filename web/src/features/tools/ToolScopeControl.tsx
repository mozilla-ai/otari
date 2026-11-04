import type { ReactNode } from "react"
import {
  AccessListControl,
  type AccessListCopy,
  type AccessOption,
} from "@/features/models/AccessListControl"
import { useDeploymentOperator } from "@/shared/api/organizations"
import { useSearchTools } from "@/shared/api/tools"

export const ALL_SEARCH_TOOLS = "*"

const EVERY_TOOL: AccessOption = {
  id: ALL_SEARCH_TOOLS,
  label: "*  ·  every search and fetch tool",
}

const COPY: Omit<AccessListCopy, "inherit"> = {
  only: "Selected tools",
  block: "No tool",
  blocked: (
    <>
      <strong>every</strong> search and fetch tool on /v1/search and /v1/fetch
      until you change this access.
    </>
  ),
  pickPrompt: "Pick at least one tool below, or choose “No tool”.",
  inputLabel: "Add a tool",
  placeholder: "Search and fetch tools…",
}

// The tool access-list, the same tri-state as model access, granted by name
// apart from it, so no model wildcard reaches a tool. It names search and fetch
// tools alike, builtin_fetch among the fetch ones. Only the deployment operator
// may list the tools, so anyone else is offered every tool or none, and keeps
// the names already on the list.
export function ToolScopeControl({
  initial,
  onChange,
  title = "Tool access",
  description,
  inheritLabel,
}: {
  initial: string[] | undefined
  onChange: (value: string[] | undefined, valid: boolean) => void
  title?: ReactNode
  description: ReactNode
  inheritLabel: string
}) {
  const { isOperator, isSettled } = useDeploymentOperator()
  const canList = isSettled && isOperator
  const searchTools = useSearchTools("search", canList)
  const fetchTools = useSearchTools("fetch", canList)
  const isLoading = searchTools.isLoading || fetchTools.isLoading

  const names = [
    ...new Set(
      [searchTools.data, fetchTools.data].flatMap((listing) =>
        [...(listing?.stored ?? []), ...(listing?.config ?? [])].map(
          (tool) => tool.name,
        ),
      ),
    ),
  ].sort((a, b) => a.localeCompare(b))
  // Nothing to pick while the lists are read, so the dropdown says it is loading
  // rather than offering the star alone as if it were everything.
  const catalog: AccessOption[] = isLoading
    ? []
    : [EVERY_TOOL, ...names.map((name) => ({ id: name, label: name }))]

  const hint =
    isSettled && !isOperator
      ? "Only the deployment operator can list the tools, so you can grant every one or none here."
      : searchTools.isError || fetchTools.isError
        ? "Could not load the tools. Reload to try again."
        : null

  return (
    <AccessListControl
      initial={initial}
      onChange={onChange}
      title={title}
      description={description}
      copy={{ ...COPY, inherit: inheritLabel }}
      catalog={catalog}
      hint={hint}
      loadingMessage={isLoading ? "Loading the tools…" : null}
    />
  )
}
