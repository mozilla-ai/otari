import type { WorkspaceMcpServer } from "@/client"
import { mcpErrorMessage } from "@/features/tools/mcpErrors"
import { useMcpSignIn } from "@/features/tools/useMcpSignIn"
import { useCheckWorkspaceMcpServer } from "@/shared/api/tools"

/** Said beside a Personal server's Check, so pressing it opens no surprise. */
export const PERSONAL_CHECK_HELP =
  "Checking a Personal server opens a sign-in. Use any account: its tools are listed once and nothing is connected."

/**
 * The one Check action for a stored server, wherever it is offered. A Shared
 * server is checked with its stored token. A Personal server holds none, so its
 * check opens a sign-in made for the check alone (`useMcpSignIn`, "check"):
 * the tools are listed once with whatever account the manager picks, the
 * result is stored, and nothing is connected. Nobody's own connection is used.
 */
export function useMcpServerCheck({
  workspaceId,
  returnTo,
  onLeave,
  openWindow,
}: {
  workspaceId: string
  returnTo: string
  onLeave?: (url: string) => void
  openWindow?: () => Window | undefined
}) {
  const check = useCheckWorkspaceMcpServer()
  const signIn = useMcpSignIn({ workspaceId, returnTo, onLeave, openWindow })

  const run = (server: Pick<WorkspaceMcpServer, "id" | "credential_mode">) => {
    if (server.credential_mode === "per_user") {
      check.reset()
      signIn.start(server.id, "check")
      return
    }
    check.mutate({ workspaceId, serverId: server.id })
  }

  const isRunning = (serverId: string) =>
    (check.isPending && check.variables?.serverId === serverId) ||
    (signIn.running?.purpose === "check" &&
      signIn.running.serverId === serverId)

  const error = check.error
    ? mcpErrorMessage(check.error, "The check could not run. Try again.")
    : signIn.error

  const dismissError = () => {
    check.reset()
    signIn.clearError()
  }

  return { run, isRunning, error, dismissError }
}
