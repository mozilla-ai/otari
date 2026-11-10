import { Link, useNavigate } from "@tanstack/react-router"
import { useEffect, useRef } from "react"

import { ErrorBanner } from "@/design-system/feedback/ErrorBanner"
import { InfoBanner } from "@/design-system/feedback/InfoBanner"
import { PageLoading } from "@/design-system/feedback/PageLoading"
import { useCompleteMcpConnection } from "@/shared/api/tools"

export interface McpConnectionCallbackSearch {
  state?: string
  code?: string
  error?: string
  /** A server a sign-in started outside the dashboard has already connected. */
  connected?: string
  return_to?: string
}

// What the gateway's callback names when it lands here without a code to spend.
const REFUSALS: Record<string, string> = {
  access_denied: "The sign-in was declined, so nothing was connected.",
  unknown_state:
    "This sign-in is unknown, already finished, or took longer than ten minutes.",
}

/** A dashboard path, and nothing that could leave the dashboard. */
function insideDashboard(path: string | null | undefined): string | undefined {
  return path?.startsWith("/") && !path.startsWith("//") ? path : undefined
}

/**
 * Where an MCP server's authorization page sends a person back to.
 *
 * The gateway's own callback cannot finish a sign-in a dashboard session
 * started, because the session cookie does not travel on a redirect from
 * another site. It passes the code and the state on to this page, which posts
 * them from inside the session, once: a code is single-use, so a failure
 * offers the way back to start again rather than a retry.
 */
export function McpConnectionCallbackPage({
  search,
}: {
  search: McpConnectionCallbackSearch
}) {
  const navigate = useNavigate()
  const complete = useCompleteMcpConnection()
  const { mutate } = complete
  const sent = useRef(false)
  const { state, code, error, connected } = search
  const returnTo = insideDashboard(search.return_to)

  useEffect(() => {
    if (connected) {
      // Replaced, here and below: Back must not return to a spent code.
      void navigate({
        to: returnTo ?? `/tools/mcp-servers/${connected}`,
        replace: true,
      })
      return
    }
    if (error || !state || !code || sent.current) return
    sent.current = true
    mutate(
      { state, code },
      {
        onSuccess: (done) =>
          void navigate({
            to:
              insideDashboard(done.return_to) ??
              `/tools/mcp-servers/${done.server_id}`,
            replace: true,
          }),
      },
    )
  }, [connected, error, state, code, returnTo, mutate, navigate])

  const refusal = error
    ? (REFUSALS[error] ??
      "The sign-in did not finish, so nothing was connected.")
    : !connected && (!state || !code)
      ? "This page finishes a sign-in to an MCP server, and it was opened without one."
      : undefined

  if (refusal === undefined && !complete.error) {
    return <PageLoading label="Connecting your account…" />
  }
  return (
    <div className="flex max-w-xl flex-col gap-4">
      <h1 className="text-heading">Connect your account</h1>
      {refusal ? <InfoBanner tone="warning">{refusal}</InfoBanner> : null}
      <ErrorBanner error={complete.error} />
      <Link to="/tools/mcp-servers" className="text-link hover:text-link-hover">
        ← MCP servers
      </Link>
    </div>
  )
}
