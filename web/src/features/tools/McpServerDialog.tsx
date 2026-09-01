import { AlertDialog, Button } from "@heroui/react"
import { useEffect, useState } from "react"

import type { WorkspaceMcpServer } from "@/client"
import { Field } from "@/shared/components/Field"
import { SecretField } from "@/shared/components/SecretField"
import { Checkbox, ErrorBanner } from "@/shared/components/ui"

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
  const [name, setName] = useState("")
  const [url, setUrl] = useState("")
  const [token, setToken] = useState("")
  const [clearToken, setClearToken] = useState(false)
  const [hint, setHint] = useState("")
  const [allowedTools, setAllowedTools] = useState("")
  const [enabled, setEnabled] = useState(true)

  // The dialog stays mounted across close and reopen, so every field is
  // reseeded each time it opens. The token box is always seeded empty, because
  // there is nothing to seed it from: leaving a previous edit's typed token in
  // it would rotate the wrong server's credential on the next save.
  useEffect(() => {
    if (!isOpen) return
    setName(editing?.name ?? "")
    setUrl(editing?.url ?? "")
    setToken("")
    setClearToken(false)
    setHint(editing?.purpose_hint ?? "")
    setAllowedTools((editing?.allowed_tools ?? []).join(", "))
    setEnabled(editing?.enabled ?? true)
  }, [isOpen, editing])

  const typedToken = token.trim() !== ""
  // What the row will hold once this save lands, which is what the https rule
  // is really about: a rename that leaves a stored token in place still has to
  // satisfy it, and a PATCH that only clears the token no longer does.
  const willHaveToken =
    typedToken || (!clearToken && Boolean(editing?.has_token))

  const urlReason = urlProblem(url, willHaveToken)
  const invalid =
    name.trim() === "" || url.trim() === "" || urlReason !== undefined

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
    <AlertDialog isOpen={isOpen} onOpenChange={onOpenChange}>
      {isOpen ? (
        <AlertDialog.Backdrop
          // Dismissable, unlike HeroUI's default for an alert dialog: this is
          // an ordinary edit form rather than a decision that has to be
          // acknowledged, so clicking away or pressing Escape cancels it.
          isDismissable
          isKeyboardDismissDisabled={false}
        >
          <AlertDialog.Container placement="center" size="lg">
            <AlertDialog.Dialog>
              <AlertDialog.Header>
                <AlertDialog.Heading>
                  {editing ? "Edit MCP server" : "Add MCP server"}
                </AlertDialog.Heading>
              </AlertDialog.Header>
              <AlertDialog.Body className="flex flex-col gap-4">
                <p className="text-sm text-muted">
                  An MCP endpoint this workspace&rsquo;s requests can reach by
                  naming its id in{" "}
                  <code className="font-mono">mcp_server_ids</code>. The gateway
                  connects to it while a request runs, so it has to be reachable
                  from the gateway rather than from this browser.
                </p>
                <ErrorBanner error={error} />

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
                  // On the field rather than at the foot of the dialog, which
                  // is five controls further down: the refusal is about this
                  // input, so it is announced with this input.
                  isInvalid={urlReason !== undefined}
                  errorMessage={urlReason}
                />

                <div className="flex flex-col gap-2">
                  {/* Masked, like the provider API key it is the sibling of:
                      the gateway stores this encrypted and never reads it back,
                      so it must not sit in the clear on the one form that
                      collects it, nor be offered to a password manager. */}
                  <SecretField
                    label="Authorization token"
                    value={token}
                    // Typing a replacement takes the tick off Remove rather
                    // than sitting beside it: the two say opposite things and
                    // only one of them can be what the operator meant.
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
                    A disabled server keeps its row and its token. A request
                    that names it skips it rather than failing.
                  </span>
                </div>
              </AlertDialog.Body>
              <AlertDialog.Footer>
                <Button
                  variant="ghost"
                  isDisabled={isPending}
                  onPress={() => onOpenChange(false)}
                >
                  Cancel
                </Button>
                <Button
                  variant="primary"
                  isDisabled={invalid}
                  isPending={isPending}
                  onPress={submit}
                >
                  {editing ? "Save server" : "Add server"}
                </Button>
              </AlertDialog.Footer>
            </AlertDialog.Dialog>
          </AlertDialog.Container>
        </AlertDialog.Backdrop>
      ) : null}
    </AlertDialog>
  )
}
