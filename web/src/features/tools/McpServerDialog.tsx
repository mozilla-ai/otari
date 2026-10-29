import { useState } from "react"
import { FiCheck } from "react-icons/fi"

import { Button } from "@/design-system/actions/Button"
import { ErrorBanner } from "@/design-system/feedback/ErrorBanner"
import { FormDialog } from "@/design-system/feedback/FormDialog"
import { InfoBanner } from "@/design-system/feedback/InfoBanner"
import { Checkbox } from "@/design-system/forms/Checkbox"
import { Field } from "@/design-system/forms/Field"
import { RadioGroup, type RadioOption } from "@/design-system/forms/RadioGroup"
import { SecretField } from "@/design-system/forms/SecretField"
import { useDirtySnapshot } from "@/design-system/forms/useDirtySnapshot"
import { ALL_TOOLS } from "@/features/tools/mcpAccess"
import { parseAllowedTools, urlProblem } from "@/features/tools/mcpServerForm"
import { healthDetail } from "@/features/tools/mcpServerHealth"
import { useTestWorkspaceMcpServerConnection } from "@/shared/api/tools"

// The form that registers a server. Changing one afterwards happens on the
// server's own page (`McpServerDetailPage`), so this is an add and nothing else.

export interface McpServerDraft {
  name: string
  url: string
  purpose_hint: string | null
  allowed_tools: string[] | null
  enabled: boolean
  /** What every key of the workspace may call: `["*"]` or nobody yet, `[]`. */
  everyone_tools: string[]
  /** Absent when the box was left blank. */
  authorization_token: string | undefined
}

type WhoCanUse = "everyone" | "nobody"

const WHO_CAN_USE: RadioOption[] = [
  {
    value: "everyone",
    label: "Allow",
    description:
      "Every key in this workspace may call every tool the server exposes, unless a rule says otherwise.",
  },
  {
    value: "nobody",
    label: "Deny",
    description:
      "No key may call it until you add allow rules for keys or users on the server's page.",
  },
]

// The server's catalog as a check listed it, each name a toggle on the
// allow-list, so choosing what to expose is picking from what is there rather
// than typing names from memory. An allow-listed name the server does not offer
// is called out, because a request would quietly expose nothing for it.
function ToolPicker({
  tools,
  allowed,
  onToggle,
}: {
  tools: string[]
  allowed: string[]
  onToggle: (name: string) => void
}) {
  const missing = allowed.filter((name) => !tools.includes(name))
  return (
    <div className="flex flex-col gap-2">
      <p className="text-sm text-muted">
        {tools.length === 0
          ? "Connected. The server offers no tools."
          : `Connected. ${tools.length} ${tools.length === 1 ? "tool" : "tools"} available; click to allow.`}
      </p>
      {tools.length > 0 ? (
        <ul
          aria-label="Tools the server offers"
          className="flex max-h-40 flex-wrap gap-1 overflow-y-auto"
        >
          {tools.map((name) => {
            const isAllowed = allowed.includes(name)
            // An empty allow-list exposes every tool, so taking the last name
            // off would widen it to all of them. Clearing the field is how an
            // operator asks for that, deliberately.
            const isLast = isAllowed && allowed.length === 1
            return (
              <li key={name}>
                <Button
                  size="sm"
                  aria-pressed={isAllowed}
                  isDisabled={isLast}
                  onPress={() => onToggle(name)}
                  className="gap-1"
                >
                  {isAllowed ? (
                    <FiCheck aria-hidden className="size-3" />
                  ) : null}
                  <span className="font-mono">{name}</span>
                </Button>
              </li>
            )
          })}
        </ul>
      ) : null}
      {missing.length > 0 ? (
        <InfoBanner tone="warning">
          Not offered by the server: {missing.join(", ")}
        </InfoBanner>
      ) : null}
    </div>
  )
}

export interface McpServerDialogProps {
  isOpen: boolean
  onOpenChange: (open: boolean) => void
  /** Where Test connection checks the draft, without storing it. */
  workspaceId: string
  isPending: boolean
  error: unknown
  onSubmit: (draft: McpServerDraft) => void
}

export function McpServerDialog({
  isOpen,
  onOpenChange,
  workspaceId,
  isPending,
  error,
  onSubmit,
}: McpServerDialogProps) {
  // The caller remounts this on each open, so the draft is cleared on the way
  // in and the fields survive the closing animation intact.
  const [name, setName] = useState("")
  const [url, setUrl] = useState("")
  const [token, setToken] = useState("")
  const [hint, setHint] = useState("")
  const [allowedTools, setAllowedTools] = useState("")
  const [enabled, setEnabled] = useState(true)
  const [whoCanUse, setWhoCanUse] = useState<WhoCanUse>("everyone")

  const typedToken = token.trim() !== ""
  const urlReason = urlProblem(url, typedToken)
  const isInvalid =
    name.trim() === "" || url.trim() === "" || urlReason !== undefined
  // One predicate naming every field the operator can change, so "is there
  // anything to lose" cannot drift from what the form actually holds.
  const { isDirty } = useDirtySnapshot({
    name,
    url,
    token,
    hint,
    allowedTools,
    enabled,
    whoCanUse,
  })

  const tokenField = typedToken ? token.trim() : undefined
  // A test answers for the URL and token it was run with, and stops answering
  // once either changes. The allow-list is left out on purpose: the picker below
  // edits it, and the catalog a test listed does not depend on it.
  const connection = JSON.stringify([url.trim(), tokenField ?? null])
  const test = useTestWorkspaceMcpServerConnection()
  const [testedConnection, setTestedConnection] = useState<string>()
  const testIsCurrent = testedConnection === connection && !test.isPending
  const tested = testIsCurrent ? test.data : undefined
  const listed = tested?.status === "healthy" ? tested : undefined
  const allowed = parseAllowedTools(allowedTools) ?? []
  const toggleTool = (name: string) => {
    const next = allowed.includes(name)
      ? allowed.filter((entry) => entry !== name)
      : [...allowed, name]
    if (next.length > 0) setAllowedTools(next.join(", "))
  }
  const runTest = () => {
    setTestedConnection(connection)
    test.mutate({
      workspaceId,
      body: {
        url: url.trim(),
        authorization_token: tokenField,
        allowed_tools: parseAllowedTools(allowedTools),
      },
    })
  }

  const submit = () => {
    if (isInvalid) return
    onSubmit({
      name: name.trim(),
      url: url.trim(),
      purpose_hint: hint.trim() === "" ? null : hint.trim(),
      allowed_tools: parseAllowedTools(allowedTools),
      enabled,
      everyone_tools: whoCanUse === "everyone" ? [ALL_TOOLS] : [],
      authorization_token: tokenField,
    })
  }

  return (
    <FormDialog
      isOpen={isOpen}
      onOpenChange={onOpenChange}
      size="lg"
      title="New MCP server"
      description={
        <>
          Requests use it by id in{" "}
          <code className="font-mono">mcp_server_ids</code>. It must be
          reachable from the gateway.
        </>
      }
      submitLabel="Add MCP server"
      onSubmit={submit}
      isPending={isPending}
      isSubmitDisabled={isInvalid}
      isDirty={isDirty}
      error={error}
      footerStart={
        <Button
          onPress={runTest}
          isPending={test.isPending}
          isDisabled={url.trim() === "" || urlReason !== undefined}
        >
          Test connection
        </Button>
      }
    >
      <Field
        label="Name"
        value={name}
        onChange={setName}
        placeholder="github"
        isRequired
        autoFocus
        description="Unique in this workspace."
      />
      <Field
        label="URL"
        value={url}
        onChange={setUrl}
        placeholder="https://mcp.example.com/github"
        isRequired
        // On the field rather than at the foot of the dialog, which is five
        // controls further down: the refusal is about this input, so it is
        // announced with this input.
        isInvalid={urlReason !== undefined}
        errorMessage={urlReason}
      />

      {/* Masked, like the provider API key it is the sibling of: the gateway
          stores this encrypted and never reads it back, so it must not sit in
          the clear on the one form that collects it, nor be offered to a
          password manager. */}
      <SecretField
        label="Authorization token"
        value={token}
        onChange={setToken}
        placeholder="Optional"
        description="Sent as a bearer token. Encrypted, never shown again."
      />

      <Field
        label="Purpose hint"
        value={hint}
        onChange={setHint}
        placeholder="Use for repository and issue lookups"
        description="Helps the model choose this server's tools."
      />
      <Field
        label="Allowed tools"
        value={allowedTools}
        onChange={setAllowedTools}
        placeholder="All tools"
        description="Comma separated. Blank allows every tool."
      />
      <div role="status" className="flex flex-col gap-2">
        {testIsCurrent ? <ErrorBanner error={test.error} /> : null}
        {tested?.status === "failing" ? (
          <InfoBanner tone="warning">
            The test failed. {healthDetail(tested)}
          </InfoBanner>
        ) : null}
        {listed ? (
          <ToolPicker
            tools={listed.tools ?? []}
            allowed={allowed}
            onToggle={toggleTool}
          />
        ) : null}
      </div>

      <div className="flex flex-col gap-1">
        <Checkbox isSelected={enabled} onChange={setEnabled}>
          Enabled
        </Checkbox>
        <span className="text-caption">Requests skip a disabled server.</span>
      </div>

      <RadioGroup
        label="Default policy"
        value={whoCanUse}
        onChange={(next) => setWhoCanUse(next as WhoCanUse)}
        options={WHO_CAN_USE}
      />
    </FormDialog>
  )
}
