import { Link, useNavigate } from "@tanstack/react-router"
import { useEffect, useRef, useState } from "react"

import { InfoBanner } from "@/design-system/feedback/InfoBanner"
import { PageLoading } from "@/design-system/feedback/PageLoading"
import { PageIntro } from "@/design-system/layout/PageIntro"
import {
  forgetSignInOrigin,
  pickDashboardPath,
  readSignInOrigin,
  rememberMcpConnectionNotice,
} from "@/features/tools/mcpConnectionNotice"
import { mcpErrorMessage } from "@/features/tools/mcpErrors"
import {
  forgetSignInWindow,
  isSignInWindow,
  type McpSignInResult,
  publishSignInResult,
} from "@/features/tools/mcpSignInWindow"
import { CANCELLED_NOTICE } from "@/features/tools/useMcpSignIn"
import { useCompleteMcpConnection } from "@/shared/api/tools"

export interface McpConnectionCallbackSearch {
  state?: string
  code?: string
  /** The authorization server's own name (RFC 9207), checked against the one the sign-in went to. */
  iss?: string
  error?: string
  /** A server a sign-in started outside the dashboard has already connected. */
  connected?: string
  return_to?: string
}

const LIST = "/tools/mcp-servers"

// What the gateway's callback names when it lands here without a code to spend.
// A Map: the code comes from the URL, and an object would answer `constructor` with a function.
const REFUSALS = new Map<string, string>([
  [
    "unknown_state",
    "This sign-in is unknown, already finished, or took longer than ten minutes.",
  ],
])

/**
 * Where an MCP server's authorization page sends a person back to.
 *
 * The gateway's own callback cannot finish a sign-in a dashboard session
 * started, because the session cookie does not travel on a redirect from
 * another site. It passes the code and the state on to this page, which posts
 * them from inside the session, once: a code is single-use, so a failure
 * offers the way back to start again rather than a retry.
 *
 * A finished or cancelled sign-in does not stop here: it goes back to the page
 * it started from (the list when none is known). A cancelled one is said
 * there; a finished one is not, since that page shows the result. Only a
 * failure that needs explaining keeps this page.
 *
 * A sign-in a dashboard page ran in a popup (`mcpSignInWindow.ts`) is finished
 * the same way, by the same session-bound POST, and then reported to that page
 * and the popup closed, instead of navigating anywhere.
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
  // Read once: React may run an initializer twice, and the effects below
  // forget it from storage.
  const [origin] = useState(readSignInOrigin)
  const { state, code, iss, error, connected } = search
  const returnTo = pickDashboardPath(search.return_to)
  const isCancelled = error === "access_denied"
  // Read once, for the same reason as the origin: the effect forgets it.
  const [inPopup] = useState(() => isSignInWindow(state))
  const [reported, setReported] = useState(false)

  useEffect(() => {
    if (!inPopup || state === undefined) return
    const report = (result: McpSignInResult) => {
      forgetSignInWindow(state)
      publishSignInResult(result)
      setReported(true)
      window.close()
    }
    if (error) {
      report(
        isCancelled
          ? { state, outcome: "cancelled" }
          : {
              state,
              outcome: "failed",
              message:
                REFUSALS.get(error) ??
                "The sign-in did not finish, so nothing was connected.",
            },
      )
      return
    }
    if (!code || sent.current) return
    sent.current = true
    mutate(
      { state, code, iss },
      {
        onSuccess: (done) =>
          report({
            state,
            outcome: done.checked ? "checked" : "connected",
            serverId: done.server_id,
            serverName: done.server_name,
          }),
        onError: (failure) =>
          report({
            state,
            outcome: "failed",
            message: mcpErrorMessage(
              failure,
              "The sign-in did not finish, so nothing was connected.",
            ),
          }),
      },
    )
  }, [inPopup, state, code, iss, error, isCancelled, mutate])

  useEffect(() => {
    if (inPopup) return
    if (connected) {
      forgetSignInOrigin()
      // Replaced, here and below: Back must not return to a spent code.
      void navigate({
        to: returnTo ?? origin ?? `${LIST}/${connected}`,
        replace: true,
      })
      return
    }
    if (isCancelled) {
      forgetSignInOrigin()
      rememberMcpConnectionNotice({ tone: "info", message: CANCELLED_NOTICE })
      void navigate({ to: origin ?? LIST, replace: true })
      return
    }
    if (error) {
      forgetSignInOrigin()
      return
    }
    if (!state || !code || sent.current) return
    sent.current = true
    mutate(
      { state, code, iss },
      {
        onSuccess: (done) => {
          forgetSignInOrigin()
          void navigate({
            to:
              pickDashboardPath(done.return_to) ??
              origin ??
              `${LIST}/${done.server_id}`,
            replace: true,
          })
        },
      },
    )
  }, [
    inPopup,
    connected,
    isCancelled,
    error,
    state,
    code,
    iss,
    returnTo,
    origin,
    mutate,
    navigate,
  ])

  const refusal =
    error && !isCancelled
      ? (REFUSALS.get(error) ??
        "The sign-in did not finish, so nothing was connected.")
      : !connected && !isCancelled && (!state || !code)
        ? "This page finishes a sign-in to an MCP server, and it was opened without one."
        : undefined

  if (inPopup) {
    // Seen only when the browser keeps the window open after `window.close()`.
    return (
      <PageIntro title={reported ? "You can close this window" : "Signing in…"}>
        {reported
          ? "The page you started from has been updated."
          : "Finishing the sign-in…"}
      </PageIntro>
    )
  }
  if (refusal === undefined && !complete.error) {
    return <PageLoading label="Connecting your account…" />
  }
  return (
    <>
      <PageIntro title="Connect your account">
        <Link to={LIST} className="text-link hover:text-link-hover">
          ← MCP servers
        </Link>
      </PageIntro>
      {refusal ? <InfoBanner tone="warning">{refusal}</InfoBanner> : null}
      {complete.error ? (
        <InfoBanner tone="warning">
          {mcpErrorMessage(
            complete.error,
            "The sign-in did not finish, so nothing was connected.",
          )}
        </InfoBanner>
      ) : null}
    </>
  )
}
