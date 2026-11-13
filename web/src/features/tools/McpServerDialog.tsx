import { useState } from "react"
import type { McpCredentialMode } from "@/client"
import { Button } from "@/design-system/actions/Button"
import { ErrorBanner } from "@/design-system/feedback/ErrorBanner"
import { FormDialog } from "@/design-system/feedback/FormDialog"
import { InfoBanner } from "@/design-system/feedback/InfoBanner"
import { Field } from "@/design-system/forms/Field"
import { RadioGroup, type RadioOption } from "@/design-system/forms/RadioGroup"
import { SecretField } from "@/design-system/forms/SecretField"
import { useDirtySnapshot } from "@/design-system/forms/useDirtySnapshot"
import { ALL_TOOLS, isLiteralToolName } from "@/features/tools/mcpAccess"
import {
  AVAILABILITY_OPTIONS,
  availabilityFields,
  type McpAvailability,
} from "@/features/tools/mcpAvailability"
import {
  parseAllowedTools,
  SIGN_IN_MODE_LABEL,
  SIGN_IN_MODES,
  urlProblem,
} from "@/features/tools/mcpServerForm"
import { healthDetail } from "@/features/tools/mcpServerHealth"
import { ToolChecklist } from "@/features/tools/ToolChecklist"
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
  /** Whose credential a call carries: the workspace's token, or each user's own. */
  credential_mode: McpCredentialMode
  /** Absent to let Otari register itself with the server's authorization server. */
  oauth_client_id: string | undefined
  oauth_client_secret: string | undefined
  /** Added to every model request from the workspace's keys without being named. */
  attach_by_default: boolean
}

type WhoCanUse = "everyone" | "nobody"

const WHO_CAN_USE: RadioOption[] = [
  {
    value: "everyone",
    label: "Allow",
    description:
      "Everyone in the workspace can use it, unless an access rule says otherwise.",
  },
  {
    value: "nobody",
    label: "Deny",
    description:
      "Nobody can use it until you add access rules on the server's page.",
  },
]

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
  const [availability, setAvailability] =
    useState<McpAvailability>("on_request")
  const [whoCanUse, setWhoCanUse] = useState<WhoCanUse>("everyone")
  const [mode, setMode] = useState<McpCredentialMode>("shared")
  const [clientId, setClientId] = useState("")
  const [clientSecret, setClientSecret] = useState("")

  const isPerUser = mode === "per_user"
  const typedToken = !isPerUser && token.trim() !== ""
  // A user's own token travels to a per-user server, so it needs https as one with a token does.
  const urlReason = urlProblem(url, typedToken, isPerUser)
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
    availability,
    whoCanUse,
    mode,
    clientId,
    clientSecret,
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
      ...availabilityFields(availability),
      everyone_tools: whoCanUse === "everyone" ? [ALL_TOOLS] : [],
      authorization_token: tokenField,
      credential_mode: mode,
      oauth_client_id:
        isPerUser && clientId.trim() !== "" ? clientId.trim() : undefined,
      oauth_client_secret:
        isPerUser && clientId.trim() !== "" && clientSecret.trim() !== ""
          ? clientSecret.trim()
          : undefined,
    })
  }

  return (
    <FormDialog
      isOpen={isOpen}
      onOpenChange={onOpenChange}
      size="lg"
      title="New MCP server"
      description="Connect a remote MCP server to this workspace."
      submitLabel="Add MCP server"
      onSubmit={submit}
      isPending={isPending}
      isSubmitDisabled={isInvalid}
      isDirty={isDirty}
      error={error}
      footerStart={
        // A per-user server is checked with a person's own connection, which
        // exists only once the server is stored.
        isPerUser ? null : (
          <Button
            onPress={runTest}
            isPending={test.isPending}
            isDisabled={url.trim() === "" || urlReason !== undefined}
          >
            Test connection
          </Button>
        )
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

      <RadioGroup
        label={SIGN_IN_MODE_LABEL}
        value={mode}
        onChange={(next) => setMode(next as McpCredentialMode)}
        options={SIGN_IN_MODES}
      />
      {isPerUser ? (
        <>
          <InfoBanner>
            After adding it, use Check on its page to list its tools. Checking
            opens a sign-in and connects nobody.
          </InfoBanner>
          <Field
            label="OAuth client ID"
            value={clientId}
            onChange={setClientId}
            placeholder="Optional"
            description="Only for a server that does not let Otari register itself. Blank registers automatically."
          />
          {/* Always on screen beside the ID, so a provider that wants both
              (GitHub) is not met with a field that appears only after typing. */}
          <SecretField
            label="Client secret"
            value={clientSecret}
            onChange={setClientSecret}
            placeholder="Optional"
            isDisabled={clientId.trim() === ""}
            description={
              clientId.trim() === ""
                ? "Goes with a client ID. Some providers, such as GitHub, need both."
                : "Encrypted, never shown again."
            }
          />
        </>
      ) : (
        // Masked, like the provider API key it is the sibling of: the gateway
        // stores this encrypted and never reads it back, so it must not sit in
        // the clear on the one form that collects it, nor be offered to a
        // password manager.
        <SecretField
          label="Authorization token"
          value={token}
          onChange={setToken}
          placeholder="Optional"
          description="Sent as a bearer token. To sign in for everyone instead, leave this blank and set the account up on the server's page after adding."
        />
      )}

      <Field
        label="Purpose hint"
        value={hint}
        onChange={setHint}
        placeholder="Use for repository and issue lookups"
        description="Helps the model choose this server's tools."
      />
      {listed ? null : (
        <Field
          label="Allowed tools"
          value={allowedTools}
          onChange={setAllowedTools}
          placeholder="All tools"
          description="Comma separated. Leave blank for every tool."
        />
      )}
      <div role="status" className="flex flex-col gap-2">
        {testIsCurrent ? <ErrorBanner error={test.error} /> : null}
        {tested?.status === "failing" ? (
          <InfoBanner tone="warning">
            The test failed. {healthDetail(tested)}
          </InfoBanner>
        ) : null}
        {listed ? (
          <div className="flex flex-col gap-2">
            <p className="text-sm text-muted">
              Allowed tools. Pick none to allow every tool the server offers.
            </p>
            <ToolChecklist
              label="Allowed tools"
              tools={(listed.tools ?? []).filter(isLiteralToolName)}
              descriptions={listed.tool_descriptions ?? {}}
              isKnown
              selected={allowed}
              onChange={(next) => setAllowedTools(next.join(", "))}
              allowPatterns={false}
            />
          </div>
        ) : null}
      </div>

      <RadioGroup
        label="Availability"
        value={availability}
        onChange={(next) => setAvailability(next as McpAvailability)}
        options={AVAILABILITY_OPTIONS}
      />

      <RadioGroup
        label="Default policy"
        value={whoCanUse}
        onChange={(next) => setWhoCanUse(next as WhoCanUse)}
        options={WHO_CAN_USE}
      />
    </FormDialog>
  )
}
