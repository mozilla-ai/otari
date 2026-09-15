// The shapes the Playground's own state is written in, which are not the wire's.
//
// A panel is what one model is saying in one column, and a turn is what it said
// plus what it cost. Neither is a generated type: the wire carries a completion
// and a usage chunk, and this is the conversation assembled out of them, which
// is the page's own model of what is on screen.

/** Tokens, cost and timing for one finished assistant turn. */
export interface TurnUsage {
  promptTokens: number
  completionTokens: number
  /** Cached input tokens, a subset of `promptTokens`, billed at the cache rate. */
  cachedTokens: number
  /** Undefined when the deployment prices nothing for this model. */
  costUsd: number | undefined
  totalMs: number
  /** Time to the first visible token. Undefined if nothing streamed. */
  ttftMs: number | undefined
  tokensPerSecond: number | undefined
}

export interface ChatTurn {
  role: "user" | "assistant"
  content: string
  /** Chain-of-thought, from a model that streams it in a field of its own. */
  reasoning?: string
  /** Present once an assistant turn has finished and reported its usage. */
  usage?: TurnUsage
  /** Set instead of `usage` when the turn ended in a refusal or a fault. */
  errorMessage?: string
}

export interface PanelState {
  /** The `instance:model` selector, or "" before one is chosen. */
  model: string
  turns: ChatTurn[]
  /** True before the first token, which is what the pending indicator reads. */
  isAwaitingFirstToken: boolean
  /** True for the whole request, including while tokens are still arriving. */
  isStreaming: boolean
}

export const EMPTY_PANEL: PanelState = {
  model: "",
  turns: [],
  isAwaitingFirstToken: false,
  isStreaming: false,
}
