import { Link } from "@tanstack/react-router"
import { useState } from "react"
import { FiPlus, FiServer, FiTool } from "react-icons/fi"
import type { PlaygroundTools } from "@/client"
import { IconButton } from "@/design-system/actions/IconButton"
import { Checkbox } from "@/design-system/forms/Checkbox"
import { Divider } from "@/design-system/layout/Divider"
import { Popover } from "@/design-system/overlays/Popover"

/**
 * The composer's "+" menu: which of this workspace's tools to attach.
 *
 * Three states per tool, from the availability read's two fields, and drawing
 * all three is the point. A tool the deployment never configured is not offered
 * at all, with a link to where it would be configured. One the deployment has
 * and this workspace turned off is shown **disabled with its reason**, because
 * somebody looking for it needs to learn where it went rather than conclude the
 * page is broken. Only an attachable tool is a live checkbox, which is what
 * stops a checkbox from looking attachable and then failing at request time
 * (otari-ai#1419).
 *
 * The three are mutually exclusive: one request carries one tool mode, so the
 * others disable while any is in use, and the menu says why rather than leaving
 * a dead checkbox unexplained.
 */
export function ToolsMenu({
  tools,
  isWebSearchOn,
  isCodeExecutionOn,
  selectedMcpIds,
  onToggleWebSearch,
  onToggleCodeExecution,
  onToggleMcpServer,
  isDisabled,
}: {
  tools: PlaygroundTools | undefined
  isWebSearchOn: boolean
  isCodeExecutionOn: boolean
  selectedMcpIds: readonly string[]
  onToggleWebSearch: (isOn: boolean) => void
  onToggleCodeExecution: (isOn: boolean) => void
  onToggleMcpServer: (id: string, isOn: boolean) => void
  isDisabled?: boolean
}) {
  const [isOpen, setIsOpen] = useState(false)

  const webSearch = tools?.web_search
  const codeExecution = tools?.code_execution
  const mcpServers = tools?.mcp_servers ?? []
  // Configured, not enabled: a tool this workspace disabled still has a row,
  // carrying the reason.
  const hasGatewayTools =
    (webSearch?.configured ?? false) || (codeExecution?.configured ?? false)
  const hasAnything = hasGatewayTools || mcpServers.length > 0
  const isMcpInUse = selectedMcpIds.length > 0

  return (
    <Popover
      isOpen={isOpen}
      onOpenChange={setIsOpen}
      placement="top"
      trigger={
        <IconButton
          label="Add tools"
          className="rounded-full"
          isDisabled={isDisabled}
        >
          <FiPlus aria-hidden className="size-4" />
        </IconButton>
      }
    >
      <div className="flex w-72 max-w-[calc(100vw-2rem)] flex-col gap-3 p-3">
        {hasAnything ? (
          <>
            {hasGatewayTools ? (
              <div className="flex flex-col gap-2">
                <span className="flex items-center gap-2 text-overline">
                  <FiTool aria-hidden className="size-3.5" />
                  Tools
                </span>
                {webSearch?.configured ? (
                  <ToolRow
                    name="Web search"
                    isSelected={isWebSearchOn}
                    isDisabled={
                      !webSearch.enabled || isCodeExecutionOn || isMcpInUse
                    }
                    onChange={onToggleWebSearch}
                    note={
                      webSearch.enabled
                        ? undefined
                        : (webSearch.reason ?? undefined)
                    }
                  />
                ) : null}
                {codeExecution?.configured ? (
                  <ToolRow
                    name="Code execution"
                    isSelected={isCodeExecutionOn}
                    isDisabled={
                      !codeExecution.enabled || isWebSearchOn || isMcpInUse
                    }
                    onChange={onToggleCodeExecution}
                    note={
                      codeExecution.enabled
                        ? "Runs in a sandbox."
                        : (codeExecution.reason ?? undefined)
                    }
                  />
                ) : null}
              </div>
            ) : null}

            {mcpServers.length > 0 ? (
              <div className="flex flex-col gap-2">
                {hasGatewayTools ? <Divider /> : null}
                <span className="flex items-center gap-2 text-overline">
                  <FiServer aria-hidden className="size-3.5" />
                  MCP servers
                </span>
                {mcpServers.map((server) => (
                  <ToolRow
                    key={server.id}
                    name={server.name}
                    isSelected={selectedMcpIds.includes(server.id)}
                    isDisabled={
                      !server.enabled || isWebSearchOn || isCodeExecutionOn
                    }
                    onChange={(isOn) => onToggleMcpServer(server.id, isOn)}
                    note={
                      server.enabled
                        ? (server.purpose_hint ?? undefined)
                        : "Turned off for this workspace."
                    }
                  />
                ))}
              </div>
            ) : null}

            {isWebSearchOn || isCodeExecutionOn || isMcpInUse ? (
              <p className="text-caption">
                Web search, code execution, and MCP cannot be combined in one
                message.
              </p>
            ) : null}
          </>
        ) : (
          <div className="flex flex-col gap-2">
            <p className="text-emphasis">No tools configured</p>
            <p className="text-caption">
              Configure web search, code execution, or an MCP server, then
              attach it here.
            </p>
            <div className="flex flex-col gap-1 pt-1">
              <Link
                to="/tools/web-search"
                className="text-caption text-link hover:text-link-hover"
              >
                Configure web search
              </Link>
              <Link
                to="/tools/code-execution"
                className="text-caption text-link hover:text-link-hover"
              >
                Configure code execution
              </Link>
              <Link
                to="/tools/mcp-servers"
                className="text-caption text-link hover:text-link-hover"
              >
                Configure MCP servers
              </Link>
            </div>
          </div>
        )}
      </div>
    </Popover>
  )
}

/** One attachable tool: its checkbox, and the line under it that qualifies it. */
function ToolRow({
  name,
  isSelected,
  isDisabled,
  onChange,
  note,
}: {
  name: string
  isSelected: boolean
  isDisabled: boolean
  onChange: (isOn: boolean) => void
  /** Why it cannot be attached, or what it does. Omitted when neither applies. */
  note?: string
}) {
  return (
    <div className="flex flex-col gap-0.5">
      <Checkbox
        isSelected={isSelected}
        isDisabled={isDisabled}
        onChange={onChange}
      >
        <span className="text-sm">{name}</span>
      </Checkbox>
      {note ? (
        <span className="line-clamp-2 pl-6 text-caption">{note}</span>
      ) : null}
    </div>
  )
}
