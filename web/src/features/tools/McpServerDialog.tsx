import { useState } from "react"
import { FiCheck } from "react-icons/fi"

import type { WorkspaceMcpServer } from "@/client"
import { Button } from "@/design-system/actions/Button"
import { ErrorBanner } from "@/design-system/feedback/ErrorBanner"
import { FormDialog } from "@/design-system/feedback/FormDialog"
import { InfoBanner } from "@/design-system/feedback/InfoBanner"
import { Checkbox } from "@/design-system/forms/Checkbox"
import { Field } from "@/design-system/forms/Field"
import { SecretField } from "@/design-system/forms/SecretField"
import { useDirtySnapshot } from "@/design-system/forms/useDirtySnapshot"
import { healthDetail } from "@/features/tools/mcpServerHealth"
import { useTestWorkspaceMcpServerConnection } from "@/shared/api/tools"

// The form behind both Add and Edit, one component rather than two: the only
// field that behaves differently between them is the bearer token, and keeping
// one form is what stops the two drifting apart.
//
// **The token has three states, not two.** The server never returns it, only
// whether one is stored (`has_token`), so an empty box cannot mean "no token":
// it means "I was shown nothing and typed nothing". The endpoint spells the
// three out (`WorkspaceMcpServerUpdate`) and this dialog maps a form onto them:
// leave the box alone to keep the stored token, type a value to rotate it, or
// tick Remove to clear it. That is why an edit sends every other field back and
// still omits this one.

export interface McpServerDraft {
  name: string
  url: string
  purpose_hint: string | null
  allowed_tools: string[] | null
  enabled: boolean
  /** `undefined` keeps the stored token, `""` clears it, a value rotates it. */
  authorization_token: string | undefined
}

// Comma-separated in the form, a list on the wire, and null rather than `[]`
// for "expose every tool this server offers": the tool loop reads `[]` as a
// deny-all, so a blank field must not send one.
function parseAllowedTools(raw: string): string[] | null {
  const names = raw
    .split(",")
    .map((name) => name.trim())
    .filter((name) => name !== "")
  return names.length > 0 ? names : null
}

// The same two rules the gateway applies when it stores a URL
// (`services/url_safety.validate_mcp_url`): an http(s) endpoint, and https once
// a bearer token rides on it. Checked here as well so the message lands under
// the field rather than arriving as a banner after a round trip. The SSRF half
// of that check is deliberately not mirrored: it resolves the host, which only
// the gateway can do, and it stays the authority either way.
function urlProblem(raw: string, willHaveToken: boolean): string | undefined {
  const trimmed = raw.trim()
  if (trimmed === "") return undefined
  let parsed: URL
  try {
    parsed = new URL(trimmed)
  } catch {
    return "Give the server's full endpoint, such as https://mcp.example.com/github."
  }
  if (parsed.protocol !== "http:" && parsed.protocol !== "https:") {
    return "An MCP endpoint is reached over http or https."
  }
  if (willHaveToken && parsed.protocol !== "https:") {
    return "A server with an authorization token needs an https URL, so the token is not sent in the clear."
  }
  return undefined
}

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

// What a save's connection check found, when it found something. The server is
// stored either way, which is what the first word says.
function CheckNotice({ server }: { server: WorkspaceMcpServer }) {
  const health = server.health
  return (
    <div role="status">
      <InfoBanner tone="warning">
        {health === null
          ? "Saved, but not checked: the gateway was busy. Use Recheck on its row."
          : health.status === "failing"
            ? `Saved, but the connection failed. ${healthDetail(health)}`
            : "Saved. Some allowed tools are not offered; see below."}
      </InfoBanner>
    </div>
  )
}

export interface McpServerDialogProps {
  isOpen: boolean
  onOpenChange: (open: boolean) => void
  /** The server being edited; absent means this is an add. */
  editing?: WorkspaceMcpServer
  /**
   * The server as this dialog last saved it, when that save's connection check
   * needs a look. The dialog stays open on it, so a fix is an edit of what was
   * just stored.
   */
  saved?: WorkspaceMcpServer
  /** Where Test connection checks the draft, without storing it. */
  workspaceId: string
  isPending: boolean
  error: unknown
  onSubmit: (draft: McpServerDraft) => void
}

export function McpServerDialog({
  isOpen,
  onOpenChange,
  editing,
  saved,
  workspaceId,
  isPending,
  error,
  onSubmit,
}: McpServerDialogProps) {
  // Seeded on mount only. The caller remounts this on each open, so there is
  // no effect reseeding it: a draft is cleared on the way in rather than on the
  // way out, and the fields survive the closing animation intact. The token box
  // is always seeded empty, because there is nothing to seed it from: the
  // server never returns one.
  const seed = {
    name: editing?.name ?? "",
    url: editing?.url ?? "",
    hint: editing?.purpose_hint ?? "",
    allowedTools: (editing?.allowed_tools ?? []).join(", "),
    enabled: editing?.enabled ?? true,
  }
  const [name, setName] = useState(seed.name)
  const [url, setUrl] = useState(seed.url)
  const [token, setToken] = useState("")
  const [clearToken, setClearToken] = useState(false)
  const [hint, setHint] = useState(seed.hint)
  const [allowedTools, setAllowedTools] = useState(seed.allowedTools)
  const [enabled, setEnabled] = useState(seed.enabled)

  const typedToken = token.trim() !== ""
  // What the row will hold once this save lands, which is what the https rule
  // is really about: a rename that leaves a stored token in place still has to
  // satisfy it, and a PATCH that only clears the token no longer does.
  const willHaveToken =
    typedToken || (!clearToken && Boolean(editing?.has_token))

  const urlReason = urlProblem(url, willHaveToken)
  const isInvalid =
    name.trim() === "" || url.trim() === "" || urlReason !== undefined
  // One predicate naming every field the operator can change, so "is there
  // anything to lose" cannot drift from what the form actually holds.
  const { isDirty, reset } = useDirtySnapshot({
    name,
    url,
    token,
    clearToken,
    hint,
    allowedTools,
    enabled,
  })
  // A save that kept the dialog open is the new baseline: the token it carried
  // is stored now, so the box empties, and closing asks nothing.
  const [seenSave, setSeenSave] = useState(saved)
  if (saved !== seenSave) {
    setSeenSave(saved)
    setToken("")
    setClearToken(false)
    reset({
      name,
      url,
      token: "",
      clearToken: false,
      hint,
      allowedTools,
      enabled,
    })
  }

  const tokenField = typedToken ? token.trim() : clearToken ? "" : undefined
  // A test answers for the URL and token it was run with, and stops answering
  // once either changes. The allow-list is left out on purpose: the picker below
  // edits it, and the catalog a test listed does not depend on it.
  const connection = JSON.stringify([url.trim(), tokenField ?? null])
  const test = useTestWorkspaceMcpServerConnection()
  const [testedConnection, setTestedConnection] = useState<string>()
  const testIsCurrent = testedConnection === connection && !test.isPending
  const tested = testIsCurrent ? test.data : undefined
  // The stored server's own last check, while the draft still describes the
  // connection that check was made on.
  const stored =
    editing && url.trim() === editing.url && tokenField === undefined
      ? editing.health
      : null
  // Once a test has answered for this connection, it alone decides: a failed
  // test hides the stored catalog rather than showing it beside the failure.
  const latest = testIsCurrent ? tested : stored
  const listed = latest?.status === "healthy" ? latest : undefined
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
        server_id: editing?.id,
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
      authorization_token: typedToken
        ? token.trim()
        : clearToken
          ? ""
          : undefined,
    })
  }

  return (
    <FormDialog
      isOpen={isOpen}
      onOpenChange={onOpenChange}
      size="lg"
      title={editing ? "Edit MCP server" : "New MCP server"}
      description={
        <>
          Requests use it by id in{" "}
          <code className="font-mono">mcp_server_ids</code>. It must be
          reachable from the gateway.
        </>
      }
      submitLabel={editing ? "Save server" : "Add MCP server"}
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
      {saved ? <CheckNotice server={saved} /> : null}
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

      <div className="flex flex-col gap-2">
        {/* Masked, like the provider API key it is the sibling of: the gateway
            stores this encrypted and never reads it back, so it must not sit in
            the clear on the one form that collects it, nor be offered to a
            password manager. */}
        <SecretField
          label="Authorization token"
          value={token}
          // Typing a replacement takes the tick off Remove rather than sitting
          // beside it: the two say opposite things and only one of them can be
          // what the operator meant.
          onChange={(next) => {
            setToken(next)
            if (next.trim() !== "") setClearToken(false)
          }}
          placeholder={
            editing?.has_token
              ? "Leave blank to keep the stored token"
              : "Optional"
          }
          description={
            editing?.has_token
              ? "Stored and never shown. Type a new one to replace it."
              : "Sent as a bearer token. Encrypted, never shown again."
          }
        />
        {editing?.has_token ? (
          <Checkbox isSelected={clearToken} onChange={setClearToken}>
            Remove the stored token
          </Checkbox>
        ) : null}
      </div>

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
    </FormDialog>
  )
}
