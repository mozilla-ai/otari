import { useEffect, useRef, useState } from "react"

import type { McpServerRequest, WorkspaceMcpServer } from "@/client"
import { Button } from "@/design-system/actions/Button"
import { ErrorBanner } from "@/design-system/feedback/ErrorBanner"
import { McpNotice, type McpNoticeTone } from "@/features/tools/McpNotice"
import {
  useAcknowledgeMcpServerRequest,
  useMyMcpServerRequests,
} from "@/shared/api/mcpServerRequests"

/** How many answers show before the rest fold behind "Show all". */
const VISIBLE = 3

interface Answer {
  request: McpServerRequest
  tone: McpNoticeTone
  message: string
  /** The Personal server the member still has to connect their account to. */
  connect?: WorkspaceMcpServer
}

/**
 * What a member is told about one answered request, or nothing when there is
 * nothing true to say yet: the server it produced is not one they can see, or
 * is switched off.
 */
export function answerFor(
  request: McpServerRequest,
  servers: readonly WorkspaceMcpServer[],
  connected: ReadonlySet<string>,
): Answer | undefined {
  if (request.status === "declined") {
    const reason = request.decision_reason?.trim()
    return {
      request,
      tone: "info",
      message: `Your request for ${request.title} was declined.${reason ? ` Reason: ${reason}` : ""}`,
    }
  }
  if (request.status !== "approved") return undefined
  const server = servers.find((entry) => entry.id === request.server_id)
  if (!server?.enabled) return undefined
  if (server.credential_mode === "per_user") {
    // A connected account needs no telling: `McpRequestAnswers` clears it.
    if (connected.has(server.id)) return undefined
    return {
      request,
      tone: "success",
      message: `${server.name} was added. Connect your account to use it.`,
      connect: server,
    }
  }
  return {
    request,
    tone: "success",
    message: server.needs_setup
      ? `${server.name} was added. An admin still has to set up its account.`
      : `${server.name} was added. It is ready to use.`,
  }
}

/**
 * The answers to a member's requests they have not dismissed yet, one quiet
 * line each above their servers: a server that was added, with Connect where
 * it needs their own account, or a request that was declined, with the admin's
 * reason. Dismissing one is stored by the gateway, so it stays gone on every
 * device, and connecting the account dismisses it too. Renders nothing when
 * there is nothing to say.
 */
export function McpRequestAnswers({
  workspaceId,
  servers,
  connected,
  onConnect,
  connectingId,
}: {
  workspaceId: string
  /** The workspace's servers as the member sees them. */
  servers: readonly WorkspaceMcpServer[]
  /** The Personal servers the member has connected; undefined until that is read. */
  connected: ReadonlySet<string> | undefined
  /** Starts the member's own sign-in to a Personal server. */
  onConnect: (server: WorkspaceMcpServer) => void
  /** The server whose sign-in is starting or waiting; `""` when none. */
  connectingId: string
}) {
  const mine = useMyMcpServerRequests(workspaceId)
  const acknowledge = useAcknowledgeMcpServerRequest()
  const [showsAll, setShowsAll] = useState(false)
  const requests = mine.data?.data
  // A request whose Personal server the member has connected is answered in
  // full, however they connected: here, on the server's card, or elsewhere.
  const cleared = useRef(new Set<string>())
  const { mutate } = acknowledge
  useEffect(() => {
    if (!requests || !connected) return
    for (const request of requests) {
      const server = servers.find((entry) => entry.id === request.server_id)
      if (
        request.status === "approved" &&
        server?.credential_mode === "per_user" &&
        connected.has(server.id) &&
        !cleared.current.has(request.id)
      ) {
        cleared.current.add(request.id)
        mutate({ workspaceId, requestId: request.id })
      }
    }
  }, [requests, servers, connected, workspaceId, mutate])

  if (!requests || !connected) return null
  const answers = requests.flatMap(
    (request) => answerFor(request, servers, connected) ?? [],
  )
  if (answers.length === 0) return <ErrorBanner error={acknowledge.error} />
  const shown = showsAll ? answers : answers.slice(0, VISIBLE)

  return (
    <section aria-label="Answers to your requests" className="flex flex-col">
      <ErrorBanner error={acknowledge.error} />
      {shown.map(({ request, tone, message, connect }, index) => (
        // One rule between two notices, not two.
        <div key={request.id} className={index > 0 ? "-mt-px" : undefined}>
          <McpNotice
            tone={tone}
            message={message}
            action={
              connect ? (
                <Button
                  size="sm"
                  variant="primary"
                  onPress={() => onConnect(connect)}
                  isPending={connectingId === connect.id}
                  aria-label={`Connect ${connect.name}`}
                  className="min-h-11 md:min-h-0"
                >
                  Connect
                </Button>
              ) : undefined
            }
            onDismiss={() =>
              acknowledge.mutate({ workspaceId, requestId: request.id })
            }
          />
        </div>
      ))}
      {answers.length > shown.length ? (
        <div className="pt-1">
          <Button size="sm" onPress={() => setShowsAll(true)}>
            Show all ({answers.length})
          </Button>
        </div>
      ) : null}
    </section>
  )
}
