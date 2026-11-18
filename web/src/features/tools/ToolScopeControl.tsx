import type { ReactNode } from "react"
import {
  AccessListControl,
  type AccessListCopy,
  type AccessOption,
  type AccessPicker,
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
}

type ToolListing = ReturnType<typeof useSearchTools>

// One kind's picker, over that kind's own listing. Nothing to pick while the
// listing is read, so the dropdown says it is loading rather than offering the
// star alone as if it were everything.
function pickerFor(
  kind: "search" | "fetch",
  listing: ToolListing,
): AccessPicker {
  const names = [
    ...new Set(
      [...(listing.data?.stored ?? []), ...(listing.data?.config ?? [])].map(
        (tool) => tool.name,
      ),
    ),
  ].sort((a, b) => a.localeCompare(b))
  const label = kind === "search" ? "Search tools" : "Fetch tools"
  return {
    label,
    catalog: listing.isLoading
      ? []
      : [EVERY_TOOL, ...names.map((name) => ({ id: name, label: name }))],
    inputLabel: `Add a ${kind} tool`,
    placeholder: `${label}…`,
    loadingMessage: listing.isLoading ? `Loading the ${kind} tools…` : "",
  }
}

// The tool access-list, the same tri-state as model access, granted by name
// apart from it, so no model wildcard reaches a tool. allowed_tools holds search
// and fetch tool names in one list, builtin_fetch among the fetch ones; each kind
// is picked from its own listing. Only the deployment operator may list the
// tools, so anyone else is offered every tool or none, and keeps the names
// already on the list.
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
  const pickers = [
    pickerFor("search", searchTools),
    pickerFor("fetch", fetchTools),
  ]

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
      pickers={pickers}
      hint={hint}
    />
  )
}
