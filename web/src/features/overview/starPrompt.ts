import type { UsageSummary } from "@/client"

/**
 * When the Overview may invite an owner or admin to star Otari on GitHub, and
 * how a browser remembers that it no longer should.
 *
 * Pure apart from storage, so the rule is tested without a page around it.
 */

/** Successful requests the gateway must have served before it asks. */
export const STAR_PROMPT_MIN_REQUESTS = 25

/**
 * Distinct days those requests must span.
 *
 * The count alone cannot tell a gateway somebody kept from one they tried: a
 * coding agent makes dozens of calls for a single prompt, so 25 can land in the
 * first afternoon. Coming back on a second day is the signal the count is a
 * proxy for, and it is what keeps the invitation away from a first test.
 */
export const STAR_PROMPT_MIN_DAYS = 2

/** How far back the count looks, in local calendar days including today. */
export const STAR_PROMPT_WINDOW_DAYS = 30

/**
 * The browser's record that the invitation was answered, either way.
 *
 * Per browser rather than per account: the gateway keeps no per-person
 * preference store, and one for a single line of copy would be a migration and
 * an endpoint for nothing else. A new browser may therefore ask once more, still
 * behind the same usage bar.
 */
export const STAR_PROMPT_STORE_KEY = "otari.dashboard.starPrompt"

export type StarPromptAnswer = "starred" | "dismissed"

/**
 * Whether a window of usage, read as successes only and bucketed by day, has
 * earned the invitation.
 *
 * Successes only because a failed request is not use anybody would thank the
 * project for, and the request filter rather than a subtraction because the
 * gateway already excludes a routing policy's absorbed attempts from the count.
 */
export function hasEarnedStarPrompt(
  successes: UsageSummary | undefined,
): boolean {
  if (!successes) {
    return false
  }
  const activeDays = successes.series.filter(
    (point) => point.requests > 0,
  ).length
  return (
    successes.totals.request_count >= STAR_PROMPT_MIN_REQUESTS &&
    activeDays >= STAR_PROMPT_MIN_DAYS
  )
}

/**
 * Whether this browser may still be asked.
 *
 * No when it has answered, and also no when storage is unavailable: a blocked
 * or full store cannot remember an answer, and an invitation that cannot be
 * put away would come back on every visit, which is the one thing it must
 * never do.
 */
export function mayOfferStarPrompt(): boolean {
  try {
    return window.localStorage.getItem(STAR_PROMPT_STORE_KEY) === null
  } catch {
    return false
  }
}

/** Record the answer, so this browser is never asked again. */
export function retireStarPrompt(answer: StarPromptAnswer): void {
  try {
    window.localStorage.setItem(STAR_PROMPT_STORE_KEY, answer)
  } catch {
    // Retired for this page view regardless; a store that refuses the write
    // also refuses the read `mayOfferStarPrompt` makes, so it is not asked on
    // the next one either.
  }
}
