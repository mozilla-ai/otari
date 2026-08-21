import type { ActivationErrorCategory } from "@/client"

/**
 * What the setup guide says about a failed first request, per category.
 *
 * The gateway sends a category and never the provider's error text, so these
 * sentences are the whole of what an operator reads: one cause, and the one page
 * in this dashboard that fixes it. Deliberately calmer and shorter than the
 * message the API would return for the same failure, because this is a first-run
 * surface and the reader has not learned the product's vocabulary yet.
 *
 * Kept in step with `activation_error_category` in
 * `src/gateway/services/tenancy/workspace_activation_service.py`, which is where
 * the categories are assigned.
 */
export interface SetupFailure {
  /** One sentence, safe to show. Never an upstream payload. */
  cause: string
  /** Where to go next, when a page in this dashboard is the answer. */
  hint?: { label: string; to: "/models" | "/budgets" | "/activity" }
}

const FAILURES: Record<ActivationErrorCategory, SetupFailure> = {
  invalid_request: {
    cause: "The request was malformed or named something the gateway refused.",
    hint: { label: "Check the model catalog", to: "/models" },
  },
  configuration: {
    cause: "The gateway could not price the model the request named.",
    // 402 is the missing-pricing rejection, and pricing is edited beside the
    // catalog on the Models page.
    hint: { label: "Open models and pricing", to: "/models" },
  },
  policy: {
    cause:
      "A budget, a model allow-list, or a rate limit rejected the request.",
    hint: { label: "Open budgets", to: "/budgets" },
  },
  upstream: {
    // No hint on purpose: nothing in this dashboard fixes a provider outage, and
    // the card already offers a re-check.
    cause: "The model provider returned an error.",
  },
  timeout: {
    cause: "The model provider did not respond in time.",
  },
  internal: {
    cause: "The gateway could not complete the request.",
    hint: { label: "Open the activity log", to: "/activity" },
  },
}

/** Used when the category is missing or one this dashboard does not know yet. */
export const UNKNOWN_FAILURE: SetupFailure = {
  cause: "The request did not go through.",
}

export function setupFailureCopy(
  category: ActivationErrorCategory | null | undefined,
): SetupFailure {
  return (category ? FAILURES[category] : undefined) ?? UNKNOWN_FAILURE
}
