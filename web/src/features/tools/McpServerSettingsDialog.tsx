import { type ReactNode, useEffect, useId, useRef, useState } from "react"

import type {
  McpCredentialMode,
  UpdateWorkspaceMcpServerRequest,
  WorkspaceMcpServer,
} from "@/client"
import { Button } from "@/design-system/actions/Button"
import { FormDialog } from "@/design-system/feedback/FormDialog"
import { InfoBanner } from "@/design-system/feedback/InfoBanner"
import { Checkbox } from "@/design-system/forms/Checkbox"
import { Field } from "@/design-system/forms/Field"
import { RadioGroup } from "@/design-system/forms/RadioGroup"
import { SecretField } from "@/design-system/forms/SecretField"
import {
  SIGN_IN_MODE_LABEL,
  SIGN_IN_MODES,
  urlProblem,
} from "@/features/tools/mcpServerForm"
import { healthNeedsAttention } from "@/features/tools/mcpServerHealth"
import { useUpdateWorkspaceMcpServer } from "@/shared/api/tools"

// A stored server's own fields, in the one place they are changed.
//
// **The token has three states, not two.** The server never returns it, only
// whether one is stored (`has_token`), so an empty box cannot mean "no token":
// it means "I was shown nothing and typed nothing". The endpoint spells the
// three out (`WorkspaceMcpServerUpdate`) and this form maps onto them: leave
// the box alone to keep the stored token, type a value to rotate it, or tick
// Remove to clear it. That is why a save sends every other field and still
// omits this one unless it changed. The OAuth client's ID and secret follow
// the same three states.

/** The field a press elsewhere on the page opens Settings on. */
export type McpSettingsFocus = "none" | "mode" | "token" | "client_id"

/** A URL's scheme and host, which an OAuth client is bound to; the whole text when it does not parse. */
function getOrigin(url: string): string {
  try {
    const parsed = new URL(url.trim())
    return `${parsed.protocol}//${parsed.host}`
  } catch {
    return url.trim()
  }
}

/** One titled group of the form, ruled off from the one above it. */
function Group({
  title,
  isFirst = false,
  children,
}: {
  title: string
  isFirst?: boolean
  children: ReactNode
}) {
  const headingId = useId()
  return (
    <section
      aria-labelledby={headingId}
      className={`flex flex-col gap-4 ${isFirst ? "" : "border-t border-border pt-4"}`}
    >
      <h3 id={headingId} className="text-title">
        {title}
      </h3>
      {children}
    </section>
  )
}

/**
 * Everything an admin sets on a stored server, behind the heading's Settings
 * button: its name, URL and purpose hint, how people sign in (one shared
 * account or each member's own, with the shared token), the OAuth client a
 * sign-in goes through, and deleting it. The caller remounts this on each
 * open, so a draft never outlives the opening it was typed in.
 */
export function McpServerSettingsDialog({
  isOpen,
  onOpenChange,
  workspaceId,
  server,
  focus = "none",
  onSaved,
  onDelete,
}: {
  isOpen: boolean
  onOpenChange: (isOpen: boolean) => void
  workspaceId: string
  server: WorkspaceMcpServer
  /** The field to land on, for a press that was about one. */
  focus?: McpSettingsFocus
  /**
   * Called once a save lands, with the saved row when its connection check
   * found something the page should say, and nothing otherwise.
   */
  onSaved: (unchecked: WorkspaceMcpServer | undefined) => void
  /** Opens the page's own delete confirmation. */
  onDelete: () => void
}) {
  const update = useUpdateWorkspaceMcpServer()
  const [name, setName] = useState(server.name)
  const [url, setUrl] = useState(server.url)
  const [token, setToken] = useState("")
  const [clearToken, setClearToken] = useState(false)
  const [hint, setHint] = useState(server.purpose_hint ?? "")
  const [mode, setMode] = useState<McpCredentialMode>(server.credential_mode)
  const [clientId, setClientId] = useState(server.oauth_client_id ?? "")
  const [clientSecret, setClientSecret] = useState("")
  const [clearClientSecret, setClearClientSecret] = useState(false)

  const frames = {
    mode: useRef<HTMLDivElement>(null),
    token: useRef<HTMLDivElement>(null),
    client_id: useRef<HTMLDivElement>(null),
  }
  const focusFrame = focus === "none" ? undefined : frames[focus]
  useEffect(() => {
    if (!isOpen || !focusFrame) return
    const land = () => {
      const frame = focusFrame.current
      const field =
        frame?.querySelector<HTMLInputElement>("input:checked") ??
        frame?.querySelector<HTMLInputElement>("input")
      if (!field || document.activeElement === field) return
      // Optional call: jsdom has no scrolling.
      field.scrollIntoView?.({ block: "center" })
      field.focus({ preventScroll: true })
    }
    land()
    // Again once the frame has settled, in case it took focus for itself.
    const settled = requestAnimationFrame(land)
    return () => cancelAnimationFrame(settled)
  }, [isOpen, focusFrame])

  const isPerUser = mode === "per_user"
  const wasPerUser = server.credential_mode === "per_user"
  const typedToken = !isPerUser && token.trim() !== ""
  // What the row will hold once this save lands, which is what the https rule
  // is about: a rename that leaves a stored token in place still has to satisfy
  // it, and a save that only clears the token no longer does.
  const willHaveToken =
    isPerUser || typedToken || (!clearToken && server.has_token)
  const urlReason = urlProblem(url, willHaveToken, isPerUser)
  const tokenField = isPerUser
    ? undefined
    : typedToken
      ? token.trim()
      : clearToken
        ? ""
        : undefined
  // Untouched keeps the client, a value sets it, and an emptied box goes back
  // to the one Otari registers itself.
  const clientIdField =
    clientId.trim() !== (server.oauth_client_id ?? "")
      ? clientId.trim()
      : undefined
  // Clearing the ID clears the secret with it, on the gateway's side, so
  // nothing is sent for it then.
  const keepsClient = clientId.trim() !== "" && clientIdField === undefined
  const clientSecretField =
    clientId.trim() === ""
      ? undefined
      : clientSecret.trim() !== ""
        ? clientSecret.trim()
        : clearClientSecret && keepsClient && server.has_oauth_client_secret
          ? ""
          : undefined
  const body: UpdateWorkspaceMcpServerRequest = {
    name: name.trim(),
    url: url.trim(),
    purpose_hint: hint.trim() === "" ? null : hint.trim(),
    // `JSON.stringify` drops an `undefined`, which is what keeps the token.
    authorization_token: tokenField,
    credential_mode: mode === server.credential_mode ? undefined : mode,
    oauth_client_id: clientIdField,
    oauth_client_secret: clientSecretField,
  }
  const isChanged =
    body.name !== server.name ||
    body.url !== server.url ||
    body.purpose_hint !== server.purpose_hint ||
    tokenField !== undefined ||
    body.credential_mode !== undefined ||
    clientIdField !== undefined ||
    clientSecretField !== undefined
  // The tokens people hold were issued for the server and the client as they were.
  const signInsMove =
    isPerUser !== wasPerUser ||
    body.url !== server.url ||
    clientIdField !== undefined
  const willEndConnections = wasPerUser && signInsMove
  // A server sends one credential, so a token typed here takes the shared account's place.
  const willEndSharedAccount =
    Boolean(server.shared_account) && (signInsMove || typedToken)
  // A client and its secret were issued for the old host, so the gateway drops them on a move.
  const willDropClient =
    server.oauth_client_id !== null &&
    clientIdField === undefined &&
    getOrigin(body.url ?? server.url) !== getOrigin(server.url)
  const isInvalid =
    name.trim() === "" || url.trim() === "" || urlReason !== undefined
  // The gateway checks the server again when its URL or token changes.
  const rechecks =
    body.url !== server.url ||
    tokenField !== undefined ||
    body.credential_mode !== undefined

  const save = () => {
    if (isInvalid || !isChanged) return
    update.mutate(
      { workspaceId, serverId: server.id, body },
      {
        onSuccess: (saved) =>
          // A per-user server with no health was not checked because nobody
          // asking has connected yet, and a server waiting for its account
          // says so at the top of the page.
          onSaved(
            rechecks &&
              saved.credential_mode === "shared" &&
              !saved.needs_setup &&
              (saved.health === null || healthNeedsAttention(saved.health))
              ? saved
              : undefined,
          ),
      },
    )
  }

  return (
    <FormDialog
      isOpen={isOpen}
      onOpenChange={onOpenChange}
      size="lg"
      title={`Settings for ${server.name}`}
      submitLabel="Save"
      onSubmit={save}
      isPending={update.isPending}
      isSubmitDisabled={!isChanged || isInvalid}
      isDirty={isChanged}
      error={update.error}
    >
      <Group title="Server" isFirst>
        <Field
          label="Name"
          value={name}
          onChange={setName}
          isRequired
          autoFocus={focus === "none"}
          description="Unique in this workspace."
        />
        <Field
          label="URL"
          value={url}
          onChange={setUrl}
          isRequired
          isInvalid={urlReason !== undefined}
          errorMessage={urlReason}
        />
        {willDropClient ? (
          <InfoBanner tone="warning">
            This URL is on another host, so saving clears the OAuth client.
            Enter the new host's client, or leave it blank.
          </InfoBanner>
        ) : null}
        <Field
          label="Purpose hint"
          value={hint}
          onChange={setHint}
          placeholder="Use for repository and issue lookups"
          description="Helps the model choose this server's tools."
        />
      </Group>

      <Group title="Credentials">
        <div ref={frames.mode}>
          <RadioGroup
            label={SIGN_IN_MODE_LABEL}
            value={mode}
            onChange={(next) => setMode(next as McpCredentialMode)}
            options={SIGN_IN_MODES}
            description="Switching ends existing connections. Members connect again, or the shared account is set up again."
          />
        </div>
        {willEndConnections ? (
          <InfoBanner tone="warning">
            Saving disconnects everyone who connected their account. They will
            need to connect again.
          </InfoBanner>
        ) : null}
        {willEndSharedAccount ? (
          <InfoBanner tone="warning">
            Saving signs the shared account out.
            {isPerUser ? "" : " Set an account up again afterwards."}
          </InfoBanner>
        ) : null}
        {!isPerUser && wasPerUser ? (
          <InfoBanner>
            After saving, set up the shared account on the server's page: sign
            in for everyone, or paste a token here.
          </InfoBanner>
        ) : null}
        {isPerUser ? null : (
          <div ref={frames.token} className="flex flex-col gap-2">
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
                  : server.shared_account
                    ? "Replaces the account an admin signed in to. Encrypted, never shown again."
                    : "Sent as a bearer token. Encrypted, never shown again."
              }
            />
            {server.has_token ? (
              // And ticking Remove clears what was typed, for the same reason.
              <Checkbox
                isSelected={clearToken}
                onChange={(remove) => {
                  setClearToken(remove)
                  if (remove) setToken("")
                }}
              >
                Remove the stored token
              </Checkbox>
            ) : null}
          </div>
        )}
      </Group>

      <Group title="OAuth client">
        <div ref={frames.client_id}>
          <Field
            label="OAuth client ID"
            value={clientId}
            onChange={setClientId}
            placeholder="Registered by Otari"
            description={
              isPerUser
                ? "Only for a server that does not let Otari register itself. Blank registers automatically."
                : "Only for signing in for everyone on a server that does not let Otari register itself."
            }
          />
        </div>
        {/* Always on screen beside the ID, so a provider that wants both
            (GitHub) is not met with a field that appears only after typing. */}
        <div className="flex flex-col gap-2">
          <SecretField
            label="Client secret"
            value={clientSecret}
            // Typing a replacement takes the tick off Remove, as with the token.
            onChange={(next) => {
              setClientSecret(next)
              if (next.trim() !== "") setClearClientSecret(false)
            }}
            isDisabled={clientId.trim() === ""}
            placeholder={
              server.has_oauth_client_secret && keepsClient
                ? "Leave blank to keep the stored secret"
                : "Optional"
            }
            description={
              clientId.trim() === ""
                ? "Goes with a client ID. Some providers, such as GitHub, need both."
                : server.has_oauth_client_secret && keepsClient
                  ? "Stored and never shown. Type a new one to replace it."
                  : "Used when exchanging and renewing tokens. Encrypted, never shown again."
            }
          />
          {server.has_oauth_client_secret && keepsClient ? (
            <Checkbox
              isSelected={clearClientSecret}
              onChange={(remove) => {
                setClearClientSecret(remove)
                if (remove) setClientSecret("")
              }}
            >
              Remove the stored secret
            </Checkbox>
          ) : null}
        </div>
      </Group>

      <Group title="Delete server">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <p className="min-w-0 max-w-prose text-sm text-muted">
            Removes the server for everyone in the workspace and ends every
            connection to it.
          </p>
          <Button
            variant="danger"
            onPress={onDelete}
            isDisabled={update.isPending}
            aria-label={`Delete ${server.name}`}
            className="min-h-11 shrink-0 md:min-h-0"
          >
            Delete server
          </Button>
        </div>
      </Group>
    </FormDialog>
  )
}
