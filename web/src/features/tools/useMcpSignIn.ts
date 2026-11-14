import { useEffect, useRef, useState } from "react"
import {
  type McpConnectionNotice,
  rememberMcpConnectionNotice,
  rememberSignInOrigin,
} from "@/features/tools/mcpConnectionNotice"
import { mcpErrorMessage } from "@/features/tools/mcpErrors"
import {
  type McpSignInResult,
  markSignInWindow,
  openSignInWindow,
  sendSignInWindow,
  stateOf,
  subscribeToSignInResult,
} from "@/features/tools/mcpSignInWindow"
import {
  useRefreshAfterMcpSignIn,
  useStartMcpCheckSignIn,
  useStartMcpConnection,
  useStartMcpSharedSignIn,
} from "@/shared/api/tools"

/** What a person cancelling at the provider is told, back on the page they left. */
export const CANCELLED_NOTICE = "Sign-in canceled. Nothing was connected."

/**
 * Why a sign-in is being made: to connect the person's own account, only to
 * check a Personal server (nothing is connected, and the person's own
 * connection, if any, is left alone), or to set up a Shared server's one
 * account for everyone.
 */
export type McpSignInPurpose = "connect" | "check" | "shared"

function leaveFor(url: string) {
  window.location.assign(url)
}

/**
 * The notice a finished sign-in leaves on the page: one for a sign-in that was
 * canceled or failed, none for one that worked, since the page shows that.
 */
export function noticeFor(
  result: McpSignInResult,
): McpConnectionNotice | undefined {
  switch (result.outcome) {
    case "connected":
    case "checked":
      return undefined
    case "cancelled":
      return { tone: "info", message: CANCELLED_NOTICE }
    case "failed":
      return { tone: "warning", message: result.message }
  }
}

/**
 * Starts an MCP sign-in in a popup and updates this page in place when it
 * finishes. Where the browser blocks the popup, the whole page goes to the
 * provider instead and the callback page brings it back, as before.
 */
export function useMcpSignIn({
  workspaceId,
  returnTo,
  onLeave = leaveFor,
  openWindow = openSignInWindow,
}: {
  workspaceId: string
  /** The dashboard path to come back to after a full-page sign-in. */
  returnTo: string
  /** Sends the whole page to the provider; a test observes it here. */
  onLeave?: (url: string) => void
  /** Opens the popup; a test stands in for the browser's window here. */
  openWindow?: () => Window | undefined
}) {
  const connect = useStartMcpConnection()
  const check = useStartMcpCheckSignIn()
  const shared = useStartMcpSharedSignIn()
  const refresh = useRefreshAfterMcpSignIn()
  // The server whose sign-in is starting or running in a popup, and why.
  const [running, setRunning] = useState<{
    serverId: string
    purpose: McpSignInPurpose
  }>()
  const [error, setError] = useState("")
  const popup = useRef<Window | undefined>(undefined)
  // The popup while it is still blank, waiting on the start to answer. Leaving
  // the page drops that answer, so the blank popup is closed with it.
  const blank = useRef<Window | undefined>(undefined)
  const unsubscribe = useRef<() => void>(() => {})

  useEffect(
    () => () => {
      unsubscribe.current()
      blank.current?.close()
    },
    [],
  )
  // Closing the popup without finishing reports nothing, so coming back to
  // this page while it is closed stops the wait. The subscription stays: a
  // provider that cut the popup off from this page makes it read as closed
  // while it is still open, and its result is still welcome when it arrives.
  useEffect(() => {
    const onFocus = () => {
      if (popup.current?.closed) setRunning(undefined)
    }
    window.addEventListener("focus", onFocus)
    return () => window.removeEventListener("focus", onFocus)
  }, [])

  const start = (serverId: string, purpose: McpSignInPurpose) => {
    setError("")
    unsubscribe.current()
    // Opened now, inside the press, which is what the browser allows.
    const opened = openWindow()
    popup.current = opened
    blank.current = opened
    setRunning({ serverId, purpose })
    const mutation =
      purpose === "check" ? check : purpose === "shared" ? shared : connect
    mutation.mutate(
      { workspaceId, serverId, returnTo },
      {
        onSuccess: (started) => {
          blank.current = undefined
          const url = started.authorization_url
          if (opened === undefined) {
            rememberSignInOrigin(returnTo)
            onLeave(url)
            return
          }
          // Closed by the person while the start was on its way: they backed out.
          if (opened.closed) {
            setRunning(undefined)
            return
          }
          const state = stateOf(url)
          markSignInWindow(state)
          unsubscribe.current = subscribeToSignInResult(state, (result) => {
            setRunning(undefined)
            refresh(workspaceId, serverId)
            const notice = noticeFor(result)
            if (notice) rememberMcpConnectionNotice(notice)
          })
          sendSignInWindow(opened, url)
        },
        onError: (failure) => {
          blank.current = undefined
          opened?.close()
          setRunning(undefined)
          setError(
            mcpErrorMessage(
              failure,
              purpose === "check"
                ? "The check could not start. Try again."
                : "The sign-in could not start. Try again.",
            ),
          )
        },
      },
    )
  }

  return {
    start,
    /** The server whose sign-in is starting or waiting, and why; undefined when none. */
    running,
    /** One plain line when a sign-in could not start; `""` otherwise. */
    error,
    /** Takes the line down, for a dismiss control. */
    clearError: () => setError(""),
  }
}
