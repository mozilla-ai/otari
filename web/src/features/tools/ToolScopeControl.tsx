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
  label: "*  ·  every search tool",
}

const COPY: Omit<AccessListCopy, "inherit"> = {
  only: "Selected tools",
  block: "No tool",
  blocked: (
    <>
      <strong>every</strong> search tool on /v1/search until you change this
      access.
    </>
  ),
  pickPrompt: "Pick at least one search tool below, or choose “No tool”.",
  inputLabel: "Add a search tool",
  placeholder: "Search tools…",
}

// The search tool access-list, the same tri-state as model access, granted by
// name apart from it, so no model wildcard reaches a tool. Only the deployment
// operator may list the search tools, so anyone else is offered every tool or
// none, and keeps the names already on the list.
export function ToolScopeControl({
  initial,
  onChange,
  title = "Search tool access",
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
  const tools = useSearchTools("search", isSettled && isOperator)

  const names = [
    ...new Set([
      ...(tools.data?.stored ?? []).map((tool) => tool.name),
      ...(tools.data?.config ?? []).map((tool) => tool.name),
    ]),
  ].sort((a, b) => a.localeCompare(b))
  // Nothing to pick while the list is read, so the dropdown says it is loading
  // rather than offering the star alone as if it were everything.
  const catalog: AccessOption[] = tools.isLoading
    ? []
    : [EVERY_TOOL, ...names.map((name) => ({ id: name, label: name }))]

  const hint =
    isSettled && !isOperator
      ? "Only the deployment operator can list the search tools, so you can grant every one or none here."
      : tools.isError
        ? "Could not load the search tools. Reload to try again."
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
      loadingMessage={tools.isLoading ? "Loading the search tools…" : null}
    />
  )
}
