import type { WorkspaceMcpServer } from "@/client"
import { Button } from "@/design-system/actions/Button"
import {
  SettingsBlock,
  SettingsGroup,
} from "@/design-system/layout/SettingsGroup"
import { CREDENTIAL_MODE_COPY } from "@/features/tools/mcpServerForm"

/**
 * How Otari reaches the server, to read: its URL, how people sign in, the
 * credential a Shared server sends and the OAuth client a sign-in goes
 * through. All of it is changed in Settings, which `onOpenSettings` opens.
 */
export function McpServerConnection({
  server,
  onOpenSettings,
}: {
  server: WorkspaceMcpServer
  onOpenSettings: () => void
}) {
  const isPersonal = server.credential_mode === "per_user"
  const credential = server.shared_account
    ? "An account an admin signed in to"
    : server.has_token
      ? "Token, stored and never shown"
      : "None"
  return (
    <SettingsGroup
      title="Connection"
      description={
        <>
          How Otari reaches this server. Change it in{" "}
          {/* Named apart from the heading's Settings button, which it repeats. */}
          <button
            type="button"
            aria-label="Open Settings"
            onClick={onOpenSettings}
            className="text-link underline hover:text-link-hover"
          >
            Settings
          </button>
          .
        </>
      }
      action={
        <Button onPress={onOpenSettings} aria-label="Edit connection">
          Edit
        </Button>
      }
      isBounded
    >
      <SettingsBlock>
        <dl className="grid grid-cols-[auto_1fr] gap-x-6 gap-y-2 text-body">
          <dt className="text-caption">URL</dt>
          <dd className="min-w-0 font-mono text-sm break-all">{server.url}</dd>
          <dt className="text-caption">Sign-in</dt>
          <dd>{CREDENTIAL_MODE_COPY[server.credential_mode].longLabel}</dd>
          {isPersonal ? null : (
            <>
              <dt className="text-caption">Credential</dt>
              <dd>{credential}</dd>
            </>
          )}
          {isPersonal || server.shared_account || server.oauth_client_id ? (
            <>
              <dt className="text-caption">OAuth client</dt>
              <dd className="min-w-0 break-all">
                {server.oauth_client_id
                  ? "Your client ID"
                  : "Registered by Otari"}
                {server.oauth_client_id ? (
                  <span className="text-muted">
                    {` ${server.oauth_client_id}`}
                    {server.has_oauth_client_secret
                      ? ", with a stored secret"
                      : ", no secret"}
                  </span>
                ) : null}
              </dd>
            </>
          ) : null}
        </dl>
      </SettingsBlock>
    </SettingsGroup>
  )
}
