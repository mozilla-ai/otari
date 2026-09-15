import { FiCode, FiGlobe, FiServer } from "react-icons/fi"

import type { PlaygroundTools } from "@/client"
import { DismissChip } from "@/design-system/indicators/DismissChip"

/**
 * What is attached to the next message, each removable.
 *
 * The tools menu is behind a popover, so without this the attachment is
 * invisible the moment the menu closes: somebody who turned on web search three
 * messages ago has no way to see that every reply since has been searching, and
 * no way to stop it without reopening the menu. Renders nothing when nothing is
 * attached.
 */
export function ActiveToolChips({
  isWebSearchOn,
  isCodeExecutionOn,
  selectedMcpIds,
  tools,
  onToggleWebSearch,
  onToggleCodeExecution,
  onToggleMcpServer,
}: {
  isWebSearchOn: boolean
  isCodeExecutionOn: boolean
  selectedMcpIds: readonly string[]
  tools: PlaygroundTools | undefined
  onToggleWebSearch: (isOn: boolean) => void
  onToggleCodeExecution: (isOn: boolean) => void
  onToggleMcpServer: (id: string, isOn: boolean) => void
}) {
  if (!isWebSearchOn && !isCodeExecutionOn && selectedMcpIds.length === 0) {
    return null
  }

  return (
    <div className="flex flex-wrap items-center gap-2">
      {isWebSearchOn ? (
        <AttachedTool
          name="Web search"
          icon={<FiGlobe aria-hidden className="size-4" />}
          onRemove={() => onToggleWebSearch(false)}
        />
      ) : null}
      {isCodeExecutionOn ? (
        <AttachedTool
          name="Code execution"
          icon={<FiCode aria-hidden className="size-4" />}
          onRemove={() => onToggleCodeExecution(false)}
        />
      ) : null}
      {selectedMcpIds.map((id) => (
        <AttachedTool
          key={id}
          name={
            tools?.mcp_servers.find((server) => server.id === id)?.name ??
            "MCP server"
          }
          icon={<FiServer aria-hidden className="size-4" />}
          onRemove={() => onToggleMcpServer(id, false)}
        />
      ))}
    </div>
  )
}

function AttachedTool({
  name,
  icon,
  onRemove,
}: {
  name: string
  icon: React.ReactNode
  onRemove: () => void
}) {
  return (
    <DismissChip
      value={
        <>
          {icon}
          {name}
        </>
      }
      dismissLabel={`Remove ${name}`}
      onDismiss={onRemove}
    />
  )
}
