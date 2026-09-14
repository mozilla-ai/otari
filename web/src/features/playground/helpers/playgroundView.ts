// Which screen the Playground is on, derived rather than tracked.
//
// Extracted from the page component so the branching that decides what renders
// is unit-testable without a React and TanStack Query render. Four questions,
// each of which was a boolean somewhere in the old page and each of which can be
// answered from the state that already exists: adding a flag for any of them is
// how a page comes to show a welcome screen over a conversation.

import type { PanelState } from "./playgroundTypes"

/**
 * Which gate stands between the caller and a usable Playground.
 *
 * `"ready"` means this deployment has at least one model the caller could
 * route to, which is the only precondition: the credential is their session,
 * and the workspace is the one the sidebar has selected.
 */
export type PlaygroundGate =
  | "loading"
  | "catalogError"
  | "noWorkspace"
  | "noModels"
  | "ready"

/**
 * Resolve the gate. Order matters and is not alphabetical.
 *
 * A still-loading read wins over everything, because every answer below it
 * would otherwise be reported from an empty list. Then a genuine fetch failure
 * is separated from an empty catalog: the first is not actionable by configuring
 * a provider and saying so would send somebody to the wrong page. A caller with
 * no workspace comes before "no models" because the catalog read is scoped to a
 * workspace, so with none the model count is meaningless rather than zero.
 */
export function derivePlaygroundGate(params: {
  isLoading: boolean
  isCatalogError: boolean
  hasWorkspace: boolean
  modelCount: number
}): PlaygroundGate {
  if (params.isLoading) return "loading"
  if (params.isCatalogError) return "catalogError"
  if (!params.hasWorkspace) return "noWorkspace"
  if (params.modelCount === 0) return "noModels"
  return "ready"
}

/** True while either panel is preparing or streaming a reply. */
export function isAnyPanelBusy(
  panelA: PanelState,
  panelB: PanelState,
): boolean {
  return (
    panelA.isAwaitingFirstToken ||
    panelB.isAwaitingFirstToken ||
    panelA.isStreaming ||
    panelB.isStreaming
  )
}

/**
 * Whether the rating bar is eligible: comparing, nothing streaming, and both
 * panels have answered the latest question.
 *
 * The last condition is what a `hasRated` flag cannot express. A rating is about
 * one exchange, so the bar has to reappear for the next one and disappear the
 * moment a new question is asked, which is exactly "the last turn in each panel
 * is an assistant's".
 */
export function haveBothPanelsAnswered(
  isComparing: boolean,
  panelA: PanelState,
  panelB: PanelState,
): boolean {
  return (
    isComparing &&
    !panelA.isStreaming &&
    !panelB.isStreaming &&
    panelA.turns.length > 0 &&
    panelB.turns.length > 0 &&
    panelA.turns[panelA.turns.length - 1]?.role === "assistant" &&
    panelB.turns[panelB.turns.length - 1]?.role === "assistant"
  )
}

/**
 * Whether to show the greeting with the composer centred under it.
 *
 * Single view only, and only before the first question: comparing starts with
 * two empty columns, which already says what the screen is for, and a greeting
 * over them would push both below the fold.
 */
export function shouldShowWelcome(
  isComparing: boolean,
  panelA: PanelState,
): boolean {
  return (
    !isComparing && panelA.turns.length === 0 && !panelA.isAwaitingFirstToken
  )
}
