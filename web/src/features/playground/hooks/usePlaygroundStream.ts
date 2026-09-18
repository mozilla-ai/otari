import type { Dispatch, SetStateAction } from "react"
import { useCallback, useEffect, useRef } from "react"

import { errorMessage } from "@/design-system/feedback/errorMessage"
import {
  type StreamUsage,
  streamPlaygroundChat,
  type ToolSelection,
} from "@/shared/api/playground"

import { computeTokensPerSecond } from "../helpers/playgroundCost"
import {
  appendErrorTurn,
  appendStreamDelta,
  patchLastAssistantTurn,
  wireMessages,
} from "../helpers/playgroundTurns"
import type {
  ChatTurn,
  PanelState,
  TurnUsage,
} from "../helpers/playgroundTypes"

type SetPanel = Dispatch<SetStateAction<PanelState>>

export interface PlaygroundStream {
  /** Stream a fresh reply onto `turns`, which must end in a question. */
  streamReply: (
    setPanel: SetPanel,
    model: string,
    turns: ChatTurn[],
  ) => Promise<void>
  /** Abort every reply in flight. Safe to call when none is. */
  stop: () => void
}

/**
 * The Playground's streaming engine: one function, used by send and regenerate.
 *
 * It owns the three things a streamed reply needs and a plain mutation cannot
 * give: the fold of each fragment into the right panel, the clock (total
 * latency and time to first token, which only the client can measure), and the
 * abort handle behind the Stop control.
 *
 * **The cost comes from the gateway's own usage chunk, not from a second call.**
 * The hosted original priced a reply by POSTing the token counts back to a
 * `/chat/completions/cost` endpoint, which is gone with the rest of that
 * backend (otari-ai#1920). It is not missed: this gateway settles the request
 * itself and the figures the stats line shows are the ones it metered, so
 * asking a second time could only disagree with the row that was actually
 * billed.
 *
 * **One controller per call, held in a set, and not one shared controller.**
 * Comparing sends the same question to both panels at once, so a single
 * controller would have B's stream abort A's the moment it started: the second
 * column would answer and the first would stop mid-sentence. The set is what
 * Stop aborts, all of it, and what unmount aborts, because leaving a provider
 * call streaming into a setter whose component is gone is a warning at best and
 * a leaked connection at worst.
 *
 * Nothing here cancels a previous request to make room for a new one, and it
 * does not need to: the composer's send is disabled while a reply is in flight
 * and regenerate declines on a streaming panel, so the only way to end one
 * early is Stop.
 */
export function usePlaygroundStream(params: {
  workspaceId: string
  tools: ToolSelection
}): PlaygroundStream {
  const { workspaceId, tools } = params
  const inFlightRef = useRef(new Set<AbortController>())

  useEffect(() => {
    const inFlight = inFlightRef.current
    return () => {
      inFlight.forEach((controller) => {
        controller.abort()
      })
      inFlight.clear()
    }
  }, [])

  const stop = useCallback(() => {
    inFlightRef.current.forEach((controller) => {
      controller.abort()
    })
    inFlightRef.current.clear()
  }, [])

  const streamReply = useCallback(
    async (setPanel: SetPanel, model: string, turns: ChatTurn[]) => {
      const controller = new AbortController()
      inFlightRef.current.add(controller)

      setPanel((prev) => ({
        ...prev,
        turns,
        isAwaitingFirstToken: true,
        isStreaming: true,
      }))

      const startedAt = performance.now()
      let firstTokenAt: number | undefined

      const recordUsage = (usage: StreamUsage) => {
        const totalMs = performance.now() - startedAt
        const ttftMs =
          firstTokenAt !== undefined ? firstTokenAt - startedAt : undefined
        // Throughput over the decode window only: the time before the first
        // token is prompt processing and queueing, and folding it in reports a
        // rate the model never ran at.
        const generationMs = ttftMs !== undefined ? totalMs - ttftMs : totalMs
        const cachedTokens = usage.prompt_tokens_details?.cached_tokens ?? 0
        const turnUsage: TurnUsage = {
          promptTokens: usage.prompt_tokens,
          completionTokens: usage.completion_tokens,
          cachedTokens,
          costUsd: undefined,
          totalMs,
          ttftMs,
          tokensPerSecond: computeTokensPerSecond(
            usage.completion_tokens,
            generationMs,
          ),
        }
        setPanel((prev) => ({
          ...prev,
          turns: patchLastAssistantTurn(prev.turns, (last) => ({
            ...last,
            usage: turnUsage,
          })),
        }))
      }

      try {
        await streamPlaygroundChat({
          workspaceId,
          model,
          messages: wireMessages(turns),
          tools,
          signal: controller.signal,
          onDelta: (delta) => {
            firstTokenAt ??= performance.now()
            setPanel((prev) => ({
              ...prev,
              turns: appendStreamDelta(prev.turns, delta),
              isAwaitingFirstToken: false,
            }))
          },
          onUsage: recordUsage,
        })
        setPanel((prev) => ({
          ...prev,
          isAwaitingFirstToken: false,
          isStreaming: false,
        }))
      } catch (error) {
        // An abort is somebody pressing Stop, so the partial reply stays as it
        // is and no failure is reported: they already know why it stopped.
        const wasStopped =
          error instanceof DOMException && error.name === "AbortError"
        setPanel((prev) => ({
          ...prev,
          turns: wasStopped
            ? prev.turns
            : appendErrorTurn(prev.turns, errorMessage(error)),
          isAwaitingFirstToken: false,
          isStreaming: false,
        }))
      } finally {
        inFlightRef.current.delete(controller)
      }
    },
    [workspaceId, tools],
  )

  return { streamReply, stop }
}
