import { useState } from "react"

import type {
  UpdateWorkspaceMcpServerRequest,
  WorkspaceMcpServer,
} from "@/client"
import { Button } from "@/design-system/actions/Button"
import { ErrorBanner } from "@/design-system/feedback/ErrorBanner"
import { InfoBanner } from "@/design-system/feedback/InfoBanner"
import { Checkbox } from "@/design-system/forms/Checkbox"
import { Field } from "@/design-system/forms/Field"
import { SecretField } from "@/design-system/forms/SecretField"
import { SettingsGroup } from "@/design-system/layout/SettingsGroup"
import { urlProblem } from "@/features/tools/mcpServerForm"
import {
  healthDetail,
  healthNeedsAttention,
} from "@/features/tools/mcpServerHealth"
import { useUpdateWorkspaceMcpServer } from "@/shared/api/tools"

// A stored server's own fields.
//
// **The token has three states, not two.** The server never returns it, only
// whether one is stored (`has_token`), so an empty box cannot mean "no token":
// it means "I was shown nothing and typed nothing". The endpoint spells the
// three out (`WorkspaceMcpServerUpdate`) and this form maps onto them: leave
// the box alone to keep the stored token, type a value to rotate it, or tick
// Remove to clear it. That is why a save sends every other field and still
// omits this one unless it changed.

/**
 * How otari reaches the server: name, URL, token and purpose hint, as a
 * summary with the form behind Edit. Enabling and deleting live in the page's
 * heading.
 */
export function McpServerConnection({
  workspaceId,
  server,
  canManage,
}: {
  workspaceId: string
  server: WorkspaceMcpServer
  canManage: boolean
}) {
  const update = useUpdateWorkspaceMcpServer()
  const [isEditing, setEditing] = useState(false)
  const [name, setName] = useState(server.name)
  const [url, setUrl] = useState(server.url)
  const [token, setToken] = useState("")
  const [clearToken, setClearToken] = useState(false)
  const [hint, setHint] = useState(server.purpose_hint ?? "")
  // What the last save's connection check found, when it found something.
  const [checked, setChecked] = useState<WorkspaceMcpServer>()
  // The stored row moves under the form too (the page's Disable button, a save
  // here), and the fields follow it rather than reading as an edit.
  const storedFields = JSON.stringify([
    server.name,
    server.url,
    server.purpose_hint,
  ])
  const [seenFields, setSeenFields] = useState(storedFields)
  if (storedFields !== seenFields) {
    setSeenFields(storedFields)
    setName(server.name)
    setUrl(server.url)
    setHint(server.purpose_hint ?? "")
  }

  const typedToken = token.trim() !== ""
  // What the row will hold once this save lands, which is what the https rule
  // is about: a rename that leaves a stored token in place still has to satisfy
  // it, and a save that only clears the token no longer does.
  const willHaveToken = typedToken || (!clearToken && server.has_token)
  const urlReason = urlProblem(url, willHaveToken)
  const tokenField = typedToken ? token.trim() : clearToken ? "" : undefined
  const body: UpdateWorkspaceMcpServerRequest = {
    name: name.trim(),
    url: url.trim(),
    purpose_hint: hint.trim() === "" ? null : hint.trim(),
    // `JSON.stringify` drops an `undefined`, which is what keeps the token.
    authorization_token: tokenField,
  }
  const isChanged =
    body.name !== server.name ||
    body.url !== server.url ||
    body.purpose_hint !== server.purpose_hint ||
    tokenField !== undefined
  const isInvalid =
    name.trim() === "" || url.trim() === "" || urlReason !== undefined
  // The gateway checks the server again when its URL or token changes.
  const rechecks = body.url !== server.url || tokenField !== undefined

  const save = () => {
    if (isInvalid) return
    update.mutate(
      { workspaceId, serverId: server.id, body },
      {
        onSuccess: (saved) => {
          setToken("")
          setClearToken(false)
          setEditing(false)
          setChecked(
            rechecks &&
              (saved.health === null || healthNeedsAttention(saved.health))
              ? saved
              : undefined,
          )
        },
      },
    )
  }

  const cancel = () => {
    setEditing(false)
    setName(server.name)
    setUrl(server.url)
    setHint(server.purpose_hint ?? "")
    setToken("")
    setClearToken(false)
    update.reset()
  }

  const warning = checked ? (
    <div role="status">
      <InfoBanner tone="warning">
        {checked.health === null
          ? "Saved, but not checked: the gateway was busy. Check the connection under Exposed tools."
          : `Saved, but the connection check found a problem. ${healthDetail(checked.health)}`}
      </InfoBanner>
    </div>
  ) : null

  if (!isEditing) {
    return (
      <SettingsGroup
        title="Connection"
        isBounded
        action={
          canManage ? (
            <Button
              onPress={() => setEditing(true)}
              aria-label="Edit connection"
            >
              Edit
            </Button>
          ) : null
        }
      >
        <div className="flex flex-col gap-3">
          {warning}
          <dl className="grid grid-cols-[auto_1fr] gap-x-6 gap-y-2 text-body">
            <dt className="text-caption">URL</dt>
            <dd className="min-w-0 font-mono text-sm break-all">
              {server.url}
            </dd>
            <dt className="text-caption">Token</dt>
            <dd>{server.has_token ? "Stored, never shown" : "None"}</dd>
          </dl>
        </div>
      </SettingsGroup>
    )
  }

  return (
    <div className="flex flex-col gap-8">
      <SettingsGroup title="Connection" isBounded>
        <div className="flex max-w-xl flex-col gap-4">
          {warning}
          <ErrorBanner error={update.error} />
          <Field
            label="Name"
            value={name}
            onChange={setName}
            isRequired
            isDisabled={!canManage}
            description="Unique in this workspace."
          />
          <Field
            label="URL"
            value={url}
            onChange={setUrl}
            isRequired
            isDisabled={!canManage}
            isInvalid={urlReason !== undefined}
            errorMessage={urlReason}
          />
          {canManage ? (
            <div className="flex flex-col gap-2">
              {/* Masked: the gateway stores this encrypted and never reads it
                  back, so it must not sit in the clear here either. */}
              <SecretField
                label="Authorization token"
                value={token}
                // Typing a replacement takes the tick off Remove: the two say
                // opposite things and only one can be what was meant.
                onChange={(next) => {
                  setToken(next)
                  if (next.trim() !== "") setClearToken(false)
                }}
                placeholder={
                  server.has_token
                    ? "Leave blank to keep the stored token"
                    : "Optional"
                }
                description={
                  server.has_token
                    ? "Stored and never shown. Type a new one to replace it."
                    : "Sent as a bearer token. Encrypted, never shown again."
                }
              />
              {server.has_token ? (
                <Checkbox isSelected={clearToken} onChange={setClearToken}>
                  Remove the stored token
                </Checkbox>
              ) : null}
            </div>
          ) : (
            <p className="text-body">
              Token: {server.has_token ? "stored" : "none"}
            </p>
          )}
          <Field
            label="Purpose hint"
            value={hint}
            onChange={setHint}
            isDisabled={!canManage}
            placeholder="Use for repository and issue lookups"
            description="Helps the model choose this server's tools."
          />
          <div className="flex flex-wrap gap-2">
            <Button
              variant="primary"
              onPress={save}
              isPending={update.isPending}
              isDisabled={!isChanged || isInvalid}
            >
              Save server
            </Button>
            <Button onPress={cancel}>Cancel</Button>
          </div>
        </div>
      </SettingsGroup>
    </div>
  )
}
