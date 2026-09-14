// The Playground's server state: the caller's retention consent, their saved
// transcripts and rated comparisons, their pinned models, and what the tools
// menu may offer. Plus the one call that is not TanStack Query's to own, the
// streaming completion, because its value arrives in pieces rather than at
// once.
//
// Every read is scoped to a workspace by query parameter and to the caller by
// their session; the gateway derives the owner and refuses a workspace that is
// not theirs, so a `workspaceId` here narrows a scope and never widens one.
//
// `undefined` is "no workspace selected yet", which every read here declines to
// run on rather than asking about a workspace nobody named.

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import type {
  PlaygroundComparisons,
  PlaygroundConsent,
  PlaygroundConsentUpdate,
  PlaygroundConversation,
  PlaygroundConversations,
  PlaygroundFavoriteModels,
  PlaygroundMessages,
  PlaygroundTools,
  SavePlaygroundComparisonRequest,
  SavePlaygroundConversationRequest,
} from "@/client"
import { apiFetch, apiStream } from "@/shared/api/client"
import { PLAYGROUND } from "@/shared/api/queryKeys"

const ROOT = "/playground"

/** The query string that scopes a read to one workspace. */
function scope(workspaceId: string): string {
  return `?workspace_id=${encodeURIComponent(workspaceId)}`
}

// ---------------------------------------------------------------------------
// Content-retention consent
//
// Per person rather than per workspace, so its key carries no workspace: the
// disclosure is about what this deployment stores about the caller.
// ---------------------------------------------------------------------------

export function usePlaygroundConsent() {
  return useQuery({
    queryKey: [PLAYGROUND, "consent"],
    queryFn: () => apiFetch<PlaygroundConsent>(`${ROOT}/consent`),
    staleTime: 60_000,
  })
}

export function useUpdatePlaygroundConsent() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (body: PlaygroundConsentUpdate) =>
      apiFetch<PlaygroundConsent>(`${ROOT}/consent`, {
        method: "PUT",
        body: JSON.stringify(body),
      }),
    // The response is the stored record, so seeding it beats invalidating: the
    // grant happens inside a confirm dialog whose next action depends on it,
    // and a refetch would race the save that follows.
    onSuccess: (data) => {
      queryClient.setQueryData([PLAYGROUND, "consent"], data)
    },
  })
}

// ---------------------------------------------------------------------------
// What the tools menu may offer
// ---------------------------------------------------------------------------

export function usePlaygroundTools(workspaceId: string | undefined) {
  return useQuery({
    queryKey: [PLAYGROUND, workspaceId, "tools"],
    queryFn: () =>
      apiFetch<PlaygroundTools>(`${ROOT}/tools${scope(workspaceId as string)}`),
    enabled: workspaceId !== undefined,
    staleTime: 60_000,
  })
}

// ---------------------------------------------------------------------------
// Saved transcripts
// ---------------------------------------------------------------------------

export function usePlaygroundConversations(workspaceId: string | undefined) {
  return useQuery({
    queryKey: [PLAYGROUND, workspaceId, "conversations"],
    queryFn: () =>
      apiFetch<PlaygroundConversations>(
        `${ROOT}/conversations${scope(workspaceId as string)}`,
      ),
    enabled: workspaceId !== undefined,
    staleTime: 30_000,
  })
}

export function useSavePlaygroundConversation(workspaceId: string | undefined) {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (body: SavePlaygroundConversationRequest) =>
      apiFetch<PlaygroundConversation>(`${ROOT}/conversations`, {
        method: "POST",
        body: JSON.stringify(body),
      }),
    // Invalidated rather than appended: the gateway prunes the oldest past its
    // cap on the same write, so the list the server now holds is not the one a
    // client-side append would produce.
    onSuccess: () => {
      void queryClient.invalidateQueries({
        queryKey: [PLAYGROUND, workspaceId, "conversations"],
      })
    },
  })
}

/**
 * One saved transcript's turns, fetched on demand rather than with the list.
 *
 * Not a `useQuery`: this is read when somebody presses a history row, and a
 * query would either need a selected-id state whose only consumer is this
 * fetch, or one query per row mounted. Loading a transcript is an action.
 */
export function fetchPlaygroundConversation(
  conversationId: string,
): Promise<PlaygroundMessages> {
  return apiFetch<PlaygroundMessages>(
    `${ROOT}/conversations/${encodeURIComponent(conversationId)}/messages`,
  )
}

export function useDeletePlaygroundConversation(
  workspaceId: string | undefined,
) {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (conversationId: string) =>
      apiFetch<void>(
        `${ROOT}/conversations/${encodeURIComponent(conversationId)}`,
        { method: "DELETE" },
      ),
    onSuccess: () => {
      void queryClient.invalidateQueries({
        queryKey: [PLAYGROUND, workspaceId, "conversations"],
      })
    },
  })
}

// ---------------------------------------------------------------------------
// Rated comparisons
// ---------------------------------------------------------------------------

export function usePlaygroundComparisons(workspaceId: string | undefined) {
  return useQuery({
    queryKey: [PLAYGROUND, workspaceId, "comparisons"],
    queryFn: () =>
      apiFetch<PlaygroundComparisons>(
        `${ROOT}/comparisons${scope(workspaceId as string)}`,
      ),
    enabled: workspaceId !== undefined,
    staleTime: 30_000,
  })
}

export function useSavePlaygroundComparison(workspaceId: string | undefined) {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (body: SavePlaygroundComparisonRequest) =>
      apiFetch<unknown>(`${ROOT}/comparisons`, {
        method: "POST",
        body: JSON.stringify(body),
      }),
    onSuccess: () => {
      void queryClient.invalidateQueries({
        queryKey: [PLAYGROUND, workspaceId, "comparisons"],
      })
    },
  })
}

export function useDeletePlaygroundComparison(workspaceId: string | undefined) {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (comparisonId: string) =>
      apiFetch<void>(
        `${ROOT}/comparisons/${encodeURIComponent(comparisonId)}`,
        { method: "DELETE" },
      ),
    onSuccess: () => {
      void queryClient.invalidateQueries({
        queryKey: [PLAYGROUND, workspaceId, "comparisons"],
      })
    },
  })
}

// ---------------------------------------------------------------------------
// Pinned models
// ---------------------------------------------------------------------------

export function usePlaygroundFavoriteModels(workspaceId: string | undefined) {
  return useQuery({
    queryKey: [PLAYGROUND, workspaceId, "favorite-models"],
    queryFn: () =>
      apiFetch<PlaygroundFavoriteModels>(
        `${ROOT}/favorite-models${scope(workspaceId as string)}`,
      ),
    enabled: workspaceId !== undefined,
    staleTime: 60_000,
  })
}

/**
 * Replace the pin list, optimistically.
 *
 * The star has to react to the press: a pin is a one-click preference and a
 * round trip of latency on it reads as a broken control. The previous list is
 * kept so a refused write rolls the star back rather than leaving the UI
 * claiming something the server did not store.
 */
export function useReplacePlaygroundFavoriteModels(
  workspaceId: string | undefined,
) {
  const queryClient = useQueryClient()
  const queryKey = [PLAYGROUND, workspaceId, "favorite-models"]
  return useMutation({
    mutationFn: (modelKeys: string[]) =>
      apiFetch<PlaygroundFavoriteModels>(
        `${ROOT}/favorite-models${scope(workspaceId as string)}`,
        { method: "PUT", body: JSON.stringify({ model_keys: modelKeys }) },
      ),
    onMutate: async (modelKeys) => {
      await queryClient.cancelQueries({ queryKey })
      const previous =
        queryClient.getQueryData<PlaygroundFavoriteModels>(queryKey)
      queryClient.setQueryData<PlaygroundFavoriteModels>(queryKey, {
        model_keys: modelKeys,
      })
      return { previous }
    },
    onError: (_error, _keys, context) => {
      if (context?.previous !== undefined) {
        queryClient.setQueryData(queryKey, context.previous)
      }
    },
    onSuccess: (data) => {
      queryClient.setQueryData(queryKey, data)
    },
  })
}

// ---------------------------------------------------------------------------
// The streaming completion
// ---------------------------------------------------------------------------

/** One streamed fragment: visible answer text, reasoning text, or both. */
export interface StreamDelta {
  content?: string
  reasoning?: string
}

/**
 * The token counts the gateway's final usage chunk carries.
 *
 * The one shape here that carries `null`, because it is the wire's and not
 * ours: OpenAI's streaming format spells an absent cache breakdown that way,
 * and a provider that reports no caching sends `prompt_tokens_details: null`.
 * Converted at the boundary (`?? 0`) so nothing above this module sees it.
 */
export interface StreamUsage {
  prompt_tokens: number
  completion_tokens: number
  total_tokens: number
  /** Cached input tokens, billed at the cache-read rate. A subset of the prompt. */
  prompt_tokens_details?: { cached_tokens?: number | null } | null
}

/** What the gateway can attach to one Playground message. */
export interface ToolSelection {
  isWebSearchOn: boolean
  isCodeExecutionOn: boolean
  mcpServerIds: string[]
}

interface StreamChunk {
  error?: string
  // Some models stream chain-of-thought in a field of its own rather than in
  // inline `<think>` tags; both are surfaced.
  choices?: { delta?: StreamDelta }[]
  // Present on the final chunk only, because the gateway is asked for it.
  usage?: StreamUsage | null
}

// The canonical tool types this gateway runs itself, which a request declares in
// `tools[]` the same way an SDK caller would (`GET /api/v1/tools` publishes
// them). Spelled here rather than read from that endpoint because they are the
// contract: the Playground attaches exactly these two, and the availability read
// is what decides whether it may.
const WEB_SEARCH_TOOL = "otari_web_search"
const CODE_EXECUTION_TOOL = "otari_code_execution"

export interface StreamChatParams {
  workspaceId: string
  /** The `instance:model` selector, as the catalog publishes it. */
  model: string
  messages: { role: string; content: string }[]
  tools: ToolSelection
  onDelta: (delta: StreamDelta) => void
  onUsage?: (usage: StreamUsage) => void
  /** Aborts the request; the Stop control and unmounting both use it. */
  signal?: AbortSignal
}

/**
 * Stream one completion, calling `onDelta` with each fragment as it arrives.
 *
 * The gateway answers in the OpenAI streaming format: `data:`-prefixed JSON
 * frames separated by a blank line, ending in `data: [DONE]`, with the final
 * frame carrying usage. A failure that happens after the headers are sent
 * arrives as a frame carrying `error`, because the status is already 200 by
 * then; one that happens before is an `ApiError` from `apiStream`.
 *
 * Written as a plain async function rather than a mutation because the value it
 * produces is not its return: the reply is delivered through `onDelta`, turn by
 * turn, and the caller folds each piece into the panel it belongs to.
 */
export async function streamPlaygroundChat({
  workspaceId,
  model,
  messages,
  tools,
  onDelta,
  onUsage,
  signal,
}: StreamChatParams): Promise<void> {
  const declaredTools = [
    ...(tools.isWebSearchOn ? [{ type: WEB_SEARCH_TOOL }] : []),
    ...(tools.isCodeExecutionOn ? [{ type: CODE_EXECUTION_TOOL }] : []),
  ]
  const body: Record<string, unknown> = {
    model,
    messages,
    stream: true,
  }
  if (declaredTools.length > 0) body.tools = declaredTools
  if (tools.mcpServerIds.length > 0) body.mcp_server_ids = tools.mcpServerIds

  const response = await apiStream(
    `${ROOT}/chat/completions${scope(workspaceId)}`,
    { method: "POST", body: JSON.stringify(body), signal },
  )

  const reader = (response.body as ReadableStream<Uint8Array>).getReader()
  const decoder = new TextDecoder()
  let buffer = ""
  try {
    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      buffer += decoder.decode(value, { stream: true })
      // Events are separated by a blank line; the trailing partial stays in the
      // buffer until the rest of it arrives.
      const events = buffer.split("\n\n")
      buffer = events.pop() ?? ""
      for (const event of events) {
        const line = event.split("\n").find((l) => l.startsWith("data:"))
        // A comment line (an SSE keep-alive) carries nothing to render.
        if (!line) continue
        const payload = line.slice("data:".length).trim()
        if (payload === "[DONE]") return
        let chunk: StreamChunk
        try {
          chunk = JSON.parse(payload) as StreamChunk
        } catch {
          continue
        }
        if (chunk.error) throw new Error(chunk.error)
        if (chunk.usage) onUsage?.(chunk.usage)
        const delta = chunk.choices?.[0]?.delta
        if (delta?.content || delta?.reasoning) {
          onDelta({ content: delta.content, reasoning: delta.reasoning })
        }
      }
    }
  } finally {
    // Releasing the lock lets the body be cancelled by the abort that may have
    // ended this loop; without it the connection is held until GC.
    reader.releaseLock()
  }
}
