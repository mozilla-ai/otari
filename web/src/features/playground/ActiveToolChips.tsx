import { FiCode, FiGlobe, FiServer, FiX } from "react-icons/fi"

import type { PlaygroundTools } from "@/client"
import { Chip } from "@/design-system/indicators/Chip"

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
    <div className="flex flex-wrap items-center gap-2 px-1 pt-1">
      {isWebSearchOn ? (
        <AttachedTool
          name="Web search"
          icon={<FiGlobe aria-hidden className="size-3.5" />}
          onRemove={() => onToggleWebSearch(false)}
        />
      ) : null}
      {isCodeExecutionOn ? (
        <AttachedTool
          name="Code execution"
          icon={<FiCode aria-hidden className="size-3.5" />}
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
          icon={<FiServer aria-hidden className="size-3.5" />}
          onRemove={() => onToggleMcpServer(id, false)}
        />
      ))}
    </div>
  )
}

/**
 * One chip with its own remove control.
 *
 * The control is a real button inside the chip rather than the chip being
 * pressable: a whole chip that removes on press has no affordance saying so, and
 * `Chip` is an indicator here as everywhere else in the product.
 */
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
    <Chip tone="accent" className="flex items-center gap-1.5">
      {icon}
      {name}
      <button
        type="button"
        aria-label={`Remove ${name}`}
        onClick={onRemove}
        className="-mr-0.5 flex size-4 items-center justify-center rounded-sm text-primary-subtle-foreground/70 transition-colors hover:text-primary-subtle-foreground"
      >
        <FiX aria-hidden className="size-3" />
      </button>
    </Chip>
  )
}
