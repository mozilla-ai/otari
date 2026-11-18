import { useState } from "react"

import type { WorkspaceMcpServer } from "@/client"
import { Button } from "@/design-system/actions/Button"
import { ConfirmDialog } from "@/design-system/feedback/ConfirmDialog"
import { Dot } from "@/design-system/indicators/Dot"
import {
  SettingsBlock,
  SettingsGroup,
} from "@/design-system/layout/SettingsGroup"
import { McpNotice } from "@/features/tools/McpNotice"
import type { McpSettingsFocus } from "@/features/tools/McpServerSettingsDialog"
import { asksForOAuthClient } from "@/features/tools/mcpErrors"
import { useMcpSignIn } from "@/features/tools/useMcpSignIn"
import { useEndMcpSharedAccount } from "@/shared/api/tools"
import { formatDate } from "@/shared/helpers/format"

/** What a Shared server's one credential is, for the list and the server's page. */
export type SharedAccountState =
  | "signed_in"
  | "token"
  | "not_set_up"
  | "open"
  | "unchecked"

/**
 * Which credential a Shared server sends: an account an admin signed in to, a
 * pasted token, none though the server asked for one, none and the last check
 * got an answer without one, or none and no check has said either way.
 */
export function sharedAccountState(
  server: Pick<
    WorkspaceMcpServer,
    "shared_account" | "has_token" | "needs_setup" | "health"
  >,
): SharedAccountState {
  if (server.shared_account) return "signed_in"
  if (server.has_token) return "token"
  if (server.needs_setup) return "not_set_up"
  return server.health?.status === "healthy" ? "open" : "unchecked"
}

const MEANING =
  "Shared: everyone in this workspace uses one account on this server."

/**
 * A Shared server's account, at the top of its page: what everyone's calls
 * carry, and for a manager the two ways to set it up. Signing in for everyone
 * opens the server's own authorization page in a popup (`useMcpSignIn`,
 * "shared"); a token is pasted in the server's Settings, which `onOpenSettings`
 * opens on the field a press is about.
 */
export function McpSharedAccount({
  workspaceId,
  server,
  canManage,
  onOpenSettings,
  onLeave,
  openWindow,
}: {
  workspaceId: string
  server: WorkspaceMcpServer
  canManage: boolean
  /** Opens Settings, where a token is pasted, the mode changed and an OAuth client entered. */
  onOpenSettings: (focus: McpSettingsFocus) => void
  /** Sends the whole page to the authorization page; a test observes it here. */
  onLeave?: (url: string) => void
  /** Opens the popup; a test stands in for the browser's window here. */
  openWindow?: () => Window | undefined
}) {
  const end = useEndMcpSharedAccount()
  const [isEnding, setEnding] = useState(false)
  const signIn = useMcpSignIn({
    workspaceId,
    returnTo: `/tools/mcp-servers/${server.id}`,
    onLeave,
    openWindow,
  })
  const state = sharedAccountState(server)
  const account = server.shared_account
  const isStarting = signIn.running?.purpose === "shared"
  const start = () => signIn.start(server.id, "shared")

  if (!canManage) {
    return (
      <SettingsGroup title="Shared account" isBounded>
        <SettingsBlock>
          <p className="text-body">
            {MEANING}{" "}
            {state === "not_set_up"
              ? "An admin has not set up an account yet, so requests skip this server until one is."
              : state === "open"
                ? "The server answered without an account, so there is nothing for you to connect."
                : state === "unchecked"
                  ? "No account is set up, so there is nothing for you to connect."
                  : "An admin set it up, so there is nothing for you to connect."}
          </p>
        </SettingsBlock>
      </SettingsGroup>
    )
  }

  const signInButton = (
    <Button
      variant={state === "not_set_up" ? "primary" : "ghost"}
      onPress={start}
      isPending={isStarting}
      className="min-h-11 md:min-h-0"
    >
      Sign in for everyone
    </Button>
  )

  return (
    <SettingsGroup
      title="Shared account"
      description={
        <>
          {MEANING} To let each member use their own,{" "}
          <button
            type="button"
            onClick={() => onOpenSettings("mode")}
            className="text-link underline hover:text-link-hover"
          >
            change the sign-in mode
          </button>
          .
        </>
      }
      isBounded
    >
      <SettingsBlock>
        {signIn.error ? (
          <McpNotice
            message={signIn.error}
            action={
              asksForOAuthClient(signIn.error) ? (
                <Button size="sm" onPress={() => onOpenSettings("client_id")}>
                  Open Settings
                </Button>
              ) : undefined
            }
            onDismiss={signIn.clearError}
          />
        ) : null}
        <div className="flex flex-wrap items-center justify-between gap-3">
          <span className="flex min-w-0 items-center gap-2 text-body">
            <Dot
              className={
                state === "not_set_up"
                  ? "bg-warning"
                  : state === "open" || state === "unchecked"
                    ? "bg-text-subtle"
                    : "bg-success"
              }
            />
            <span className="break-words">
              {account
                ? account.signed_in_by
                  ? `Signed in by ${account.signed_in_by} on ${formatDate(account.signed_in_at)}`
                  : `Signed in on ${formatDate(account.signed_in_at)}`
                : state === "token"
                  ? "Uses a stored token"
                  : state === "not_set_up"
                    ? "Not set up yet"
                    : "No account set up"}
            </span>
          </span>
          <div className="flex flex-wrap gap-2">
            {account ? (
              <>
                <Button
                  onPress={start}
                  isPending={isStarting}
                  className="min-h-11 md:min-h-0"
                >
                  Reconnect
                </Button>
                <Button
                  onPress={() => {
                    end.reset()
                    setEnding(true)
                  }}
                  className="min-h-11 md:min-h-0"
                >
                  Disconnect
                </Button>
              </>
            ) : (
              <>
                {signInButton}
                <Button
                  onPress={() => onOpenSettings("token")}
                  className="min-h-11 md:min-h-0"
                >
                  {state === "token" ? "Change token" : "Paste a token"}
                </Button>
              </>
            )}
          </div>
        </div>
        {isStarting ? (
          <p className="text-caption">
            Finish signing in in the window that opened.
          </p>
        ) : null}
        <p className="text-caption">
          {state === "not_set_up"
            ? "This server asks for an account, so requests skip it until one is set up. Sign in for everyone, or paste a token. "
            : state === "open"
              ? "The server answered without an account. Set one up if its tools need it. "
              : state === "unchecked"
                ? "Requests reach the server without an account. Set one up if its tools need it. "
                : state === "token"
                  ? "Signing in for everyone replaces the token. "
                  : ""}
          {state === "token"
            ? null
            : "Everyone in the workspace acts as the account you sign in with, so use a service account rather than your own."}
        </p>
        {account && !account.renewable ? (
          <p className="text-caption">
            This sign-in cannot renew itself. Reconnect when it expires.
          </p>
        ) : null}
      </SettingsBlock>

      {isEnding ? (
        <ConfirmDialog
          isOpen
          onOpenChange={setEnding}
          heading="Disconnect the shared account"
          body={`Everyone's keys and the Playground stop reaching ${server.name} until an account is set up again.`}
          confirmLabel="Disconnect"
          isPending={end.isPending}
          error={end.error}
          onConfirm={() =>
            end.mutate(
              { workspaceId, serverId: server.id },
              { onSuccess: () => setEnding(false) },
            )
          }
        />
      ) : null}
    </SettingsGroup>
  )
}
