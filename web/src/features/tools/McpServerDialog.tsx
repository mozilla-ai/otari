import { useState } from "react"

import type { WorkspaceMcpServer } from "@/client"
import { FormDialog } from "@/design-system/feedback/FormDialog"
import { Checkbox } from "@/design-system/forms/Checkbox"
import { Field } from "@/design-system/forms/Field"
import { SecretField } from "@/design-system/forms/SecretField"
import { useDirtySnapshot } from "@/design-system/forms/useDirtySnapshot"

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
// for "expose every tool this server offers". Both reach that behavior today,
// because `mcp_client` reads a falsy allow-list as no allow-list at all, but
// null is what the column stores for the absent case and what the endpoint's
// own description names, so sending `[]` would only add a second spelling of
// one state.
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

export interface McpServerDialogProps {
  isOpen: boolean
  onOpenChange: (open: boolean) => void
  /** The server being edited; absent means this is an add. */
  editing?: WorkspaceMcpServer
  isPending: boolean
  error: unknown
  onSubmit: (draft: McpServerDraft) => void
}

export function McpServerDialog({
  isOpen,
  onOpenChange,
  editing,
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
  const invalid =
    name.trim() === "" || url.trim() === "" || urlReason !== undefined
  // One predicate naming every field the operator can change, so "is there
  // anything to lose" cannot drift from what the form actually holds.
  const { isDirty } = useDirtySnapshot({
    name,
    url,
    token,
    clearToken,
    hint,
    allowedTools,
    enabled,
  })

  const submit = () => {
    if (invalid) return
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
          An MCP endpoint this workspace's requests can reach by naming its id
          in <code className="font-mono">mcp_server_ids</code>. The gateway
          connects to it while a request runs, so it has to be reachable from
          the gateway rather than from this browser.
        </>
      }
      submitLabel={editing ? "Save server" : "Add MCP server"}
      onSubmit={submit}
      isPending={isPending}
      isSubmitDisabled={invalid}
      isDirty={isDirty}
      error={error}
    >
      <Field
        label="Name"
        value={name}
        onChange={setName}
        placeholder="github"
        isRequired
        autoFocus
        description="Unique within this workspace, and what the model sees this server's tools labeled with."
      />
      <Field
        label="URL"
        value={url}
        onChange={setUrl}
        placeholder="https://mcp.example.com/github"
        isRequired
        description="The streamable HTTP MCP endpoint."
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
              : "Optional bearer token"
          }
          description={
            editing?.has_token
              ? "A token is stored for this server. It is never shown; type a new one only to replace it."
              : "Sent as a bearer token. Stored encrypted and never shown again."
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
        description="Prepended to the system message to help the model choose this server's tools."
      />
      <Field
        label="Allowed tools"
        value={allowedTools}
        onChange={setAllowedTools}
        placeholder="Comma separated, blank for every tool"
        description="Only these tool names are exposed to the model. Blank exposes every tool the server offers."
      />

      <div className="flex flex-col gap-1">
        <Checkbox isSelected={enabled} onChange={setEnabled}>
          Enabled
        </Checkbox>
        <span className="text-caption">
          A disabled server keeps its row and its token. A request that names it
          skips it rather than failing.
        </span>
      </div>
    </FormDialog>
  )
}
