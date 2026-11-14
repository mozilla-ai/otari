import { type ReactNode, useState } from "react"

import type { WorkspaceMcpServer } from "@/client"
import { ErrorBanner } from "@/design-system/feedback/ErrorBanner"
import { SearchField } from "@/design-system/forms/SearchField"
import { Chip } from "@/design-system/indicators/Chip"
import {
  SettingsBlock,
  SettingsGroup,
} from "@/design-system/layout/SettingsGroup"
import { useMyMcpConnections, useMyMcpServerTools } from "@/shared/api/tools"
import { formatDateTime } from "@/shared/helpers/format"

/**
 * The tools this member may use on a server, under the access rules that apply
 * to them, with each tool's own one-line description where the server gave
 * one, a Read-only chip on the tools labeled read, and a search over names and
 * descriptions. Says plainly when the list is not known yet, and why.
 */
export function McpMemberTools({
  workspaceId,
  server,
}: {
  workspaceId: string
  server: WorkspaceMcpServer
}) {
  const mine = useMyMcpConnections(workspaceId)
  const isConnected = (mine.data?.data ?? []).some(
    (entry) => entry.server_id === server.id,
  )
  const tools = useMyMcpServerTools(workspaceId, server.id, true)
  const data = tools.data
  const isPersonal = server.credential_mode === "per_user"
  const [query, setQuery] = useState("")
  const needle = query.trim().toLowerCase()
  const shown = (data?.tools ?? []).filter(
    (tool) =>
      needle === "" ||
      tool.name.toLowerCase().includes(needle) ||
      (tool.description ?? "").toLowerCase().includes(needle),
  )

  let body: ReactNode
  if (tools.isPending && !data) {
    body = <p className="text-caption">Reading your tools…</p>
  } else if (!data) {
    body = <ErrorBanner error={tools.error} />
  } else if (data.access === "none") {
    body = (
      <p className="text-body">
        {data.decided_by === "server"
          ? "This server is turned off, so no tool on it can be used."
          : "The access rules here don't let you use any tool on this server. Ask a workspace admin if you need one."}
      </p>
    )
  } else if (!data.known) {
    body = (
      <p className="text-body">
        {isPersonal && !isConnected
          ? "Not known yet. Connect your account above to see the tools you can use."
          : "Not known yet. The list fills in the first time a request uses this server, or when a workspace admin checks it."}
        {data.access === "all"
          ? " Your access rules allow every tool it offers."
          : ""}
      </p>
    )
  } else if (data.tools.length === 0) {
    body = (
      <p className="text-body">
        None of the tools this server offers are allowed for you.
      </p>
    )
  } else {
    body = (
      <div className="flex flex-col gap-2">
        <SearchField
          label="Search your tools"
          placeholder="Search tools"
          value={query}
          onChange={setQuery}
        />
        {shown.length === 0 ? (
          <p className="text-caption">No tool matches "{query.trim()}".</p>
        ) : (
          <ul
            aria-label={`Your tools on ${server.name}`}
            className="flex max-h-96 flex-col overflow-y-auto"
          >
            {shown.map((tool) => (
              <li
                key={tool.name}
                className="flex min-w-0 flex-col gap-0.5 border-b border-border-subtle py-2 last:border-b-0"
              >
                <span className="flex min-w-0 flex-wrap items-center gap-2">
                  <span className="font-mono text-sm break-all">
                    {tool.name}
                  </span>
                  {tool.read_only ? <Chip tone="info">Read-only</Chip> : null}
                </span>
                {tool.description ? (
                  <span className="text-caption break-words">
                    {tool.description}
                  </span>
                ) : null}
              </li>
            ))}
          </ul>
        )}
      </div>
    )
  }

  return (
    <SettingsGroup
      title="Your tools"
      count={data?.known ? data.tools.length : undefined}
      description={
        data?.known && data.checked_at
          ? `What you can use on this server, as of ${formatDateTime(data.checked_at)}.`
          : "What you can use on this server."
      }
      isBounded
    >
      <SettingsBlock>{body}</SettingsBlock>
    </SettingsGroup>
  )
}
