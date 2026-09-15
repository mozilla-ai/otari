// Turning what is on screen into what gets stored.
//
// Both builders here drop something on purpose, and that is the reason they are
// functions rather than inline object literals at the call site. A transcript is
// saved without its usage figures, because those describe the request that ran
// rather than the conversation, and a resumed transcript that showed yesterday's
// latency as if it were this session's would be lying. And a comparison is built
// from the *latest exchange only*, not the whole transcript, because a rating is
// a judgment about one question.

import type {
  SavePlaygroundComparisonRequest,
  SavePlaygroundConversationRequest,
} from "@/client"

import type { ChatTurn } from "./playgroundTypes"

/** The longest title kept before it is truncated for the history list. */
const MAX_TITLE_LENGTH = 60

/**
 * A title derived from the first question, truncated so the list stays scannable.
 *
 * Derived rather than asked for: the save is one press, and a dialog demanding a
 * name for something somebody wants to keep is the reason they stop keeping
 * things. "Untitled" only for a transcript with no question in it, which nothing
 * on the page can currently produce.
 */
export function buildConversationTitle(turns: ChatTurn[]): string {
  const firstQuestion =
    turns.find((turn) => turn.role === "user")?.content ?? "Untitled"
  return firstQuestion.length > MAX_TITLE_LENGTH
    ? `${firstQuestion.slice(0, MAX_TITLE_LENGTH)}…`
    : firstQuestion
}

/** The request body that stores one panel's transcript. */
export function buildConversationRequest(params: {
  workspaceId: string
  model: string
  turns: ChatTurn[]
}): SavePlaygroundConversationRequest {
  return {
    workspace_id: params.workspaceId,
    model: params.model,
    title: buildConversationTitle(params.turns),
    // A turn with no reasoning omits the field rather than sending an explicit
    // null: the schema defaults it, and this way the request carries only what
    // the conversation actually had.
    messages: params.turns.map((turn) => ({
      role: turn.role,
      content: turn.content,
      ...(turn.reasoning === undefined ? {} : { reasoning: turn.reasoning }),
    })),
  }
}

export interface RatedExchange {
  question: string
  answerA: string
  answerB: string
}

/**
 * The latest question and both answers to it, or undefined when there is
 * nothing rateable yet.
 *
 * Found from panel A's last question rather than from either panel's end,
 * because the two panels can be a turn out of step while one is still
 * streaming: the question is asked once and sent to both, so A's copy of it is
 * the anchor and each panel's first assistant turn after that index is its
 * answer. Undefined when either panel has not answered, which is what keeps the
 * rating bar from recording half a comparison.
 */
export function findRatedExchange(
  turnsA: ChatTurn[],
  turnsB: ChatTurn[],
): RatedExchange | undefined {
  const questionIndex = turnsA.map((turn) => turn.role).lastIndexOf("user")
  if (questionIndex === -1) return undefined
  const question = turnsA[questionIndex]?.content
  if (question === undefined) return undefined

  const answerA = turnsA
    .slice(questionIndex + 1)
    .find((turn) => turn.role === "assistant")?.content
  const answerB = turnsB
    .slice(questionIndex + 1)
    .find((turn) => turn.role === "assistant")?.content
  if (!answerA || !answerB) return undefined

  return { question, answerA, answerB }
}

/** The request body that records one rated exchange. */
export function buildComparisonRequest(params: {
  workspaceId: string
  modelA: string
  modelB: string
  exchange: RatedExchange
  preference: SavePlaygroundComparisonRequest["preference"]
}): SavePlaygroundComparisonRequest {
  return {
    workspace_id: params.workspaceId,
    user_question: params.exchange.question,
    model_a: params.modelA,
    model_b: params.modelB,
    model_a_answer: params.exchange.answerA,
    model_b_answer: params.exchange.answerB,
    preference: params.preference,
  }
}

/**
 * Toggle one model's pin, returning the next list.
 *
 * A newly pinned model leads, so the Favorites group is ordered most-recent
 * first: somebody pinning a model is about to use it, and putting it at the
 * bottom of a list of fifty defeats the pin.
 */
export function togglePinnedModel(
  pinned: readonly string[],
  modelKey: string,
): string[] {
  if (pinned.includes(modelKey)) {
    return pinned.filter((key) => key !== modelKey)
  }
  return [modelKey, ...pinned]
}
