import { useState } from "react"

import { Button } from "@/design-system/actions/Button"
import { ErrorBanner } from "@/design-system/feedback/ErrorBanner"
import { ComboBoxField } from "@/design-system/forms/ComboBoxField"
import { Select } from "@/design-system/forms/Select"
import { Dot } from "@/design-system/indicators/Dot"
import { verdictLabel } from "@/features/tools/mcpAccess"
import { useWorkspaceMcpServerAccessCheck } from "@/shared/api/tools"

/**
 * Asks the gateway whether one key may call one tool here, and which rule
 * says so: the same order a request takes, so the answer is the gateway's and
 * not a second reading of the list above.
 */
export function McpAccessTest({
  workspaceId,
  serverId,
  serverEnabled,
  keys,
  tools,
  ruleOwner,
}: {
  workspaceId: string
  serverId: string
  serverEnabled: boolean
  /** The keys to pick from, each with the name the verdict uses. */
  keys: readonly { id: string; label: string }[]
  /** Tool names to suggest; anything typed is asked about as well. */
  tools: readonly string[]
  /** Who a deciding grant is for, by its id. */
  ruleOwner: (grantId: string) => string
}) {
  const [keyId, setKeyId] = useState("")
  const [tool, setTool] = useState("")
  const [query, setQuery] = useState("")
  const [asked, setAsked] = useState({ keyId: "", tool: "" })
  const check = useWorkspaceMcpServerAccessCheck({
    workspaceId,
    serverId,
    apiKeyId: asked.keyId,
    tool: asked.tool,
  })
  const needle = query.trim().toLowerCase()
  const suggestions = tools
    .filter((name) => name.toLowerCase().includes(needle))
    .map((name) => ({ value: name, label: name }))
  const keyLabel = keys.find((key) => key.id === asked.keyId)?.label ?? ""
  const result = check.data

  return (
    <div className="flex flex-col gap-3">
      <div className="grid gap-3 md:grid-cols-[1fr_1fr_auto] md:items-end">
        <Select
          label="Source key"
          value={keyId}
          onChange={setKeyId}
          placeholder="Pick a key"
          options={keys.map((key) => ({ value: key.id, label: key.label }))}
          shouldReserveMessage={false}
        />
        <ComboBoxField
          label="Tool"
          value={tool}
          onChange={setTool}
          onQueryChange={setQuery}
          options={suggestions}
          allowsCustomValue
          placeholder="create_issue"
          isSourceEmpty={tools.length === 0}
          emptyMessage="The server's last check listed no tools. Type a name."
          noMatchesMessage="No listed tool matches. Test it as typed."
          shouldReserveMessage={false}
        />
        <Button
          onPress={() => setAsked({ keyId, tool: tool.trim() })}
          isDisabled={keyId === "" || tool.trim() === ""}
          isPending={check.isFetching}
          className="min-h-11 md:min-h-0"
        >
          Trace
        </Button>
      </div>
      <ErrorBanner error={check.error} />
      {result ? (
        <p
          role="status"
          className="flex flex-wrap items-center gap-2 text-body"
        >
          <Dot className={result.allowed ? "bg-success" : "bg-danger"} />
          <span className="break-words">
            {keyLabel} · <code className="font-mono">{asked.tool}</code> →{" "}
            {verdictLabel(result, {
              ruleOwner: result.grant_id ? ruleOwner(result.grant_id) : "",
              serverEnabled,
            })}
          </span>
        </p>
      ) : null}
    </div>
  )
}
