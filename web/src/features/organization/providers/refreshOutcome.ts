import type { OrgProviderModelsRefresh } from "@/client"

/** Which button was pressed, since the two report different kinds of nothing. */
export type RefreshKind = "models" | "pricing"

const plural = (count: number, one: string, many: string) =>
  `${count} ${count === 1 ? one : many}`

/**
 * One sentence for what a refresh did.
 *
 * Worth saying either way: silence after a press reads as a button that did
 * nothing, and "nothing changed" is the common and correct answer for both of
 * these. A refusal comes back in the body rather than thrown (an unreachable
 * provider is a fact about the provider), so it is worded here too.
 *
 * The two kinds differ only in what they say when nothing moved, because the
 * questions differ: Refresh models asked the provider and Refresh pricing asked
 * the community dataset, and an operator pressing one wants to know which of
 * those had no news.
 */
export function refreshOutcome(
  result: OrgProviderModelsRefresh,
  kind: RefreshKind,
): string {
  if (result.error) return result.error
  if (result.discovery_unsupported) {
    return "This provider does not publish a model list. Add the models you want by name."
  }
  const parts = [
    result.added.length > 0
      ? `Offered ${plural(result.added.length, "new model", "new models")}.`
      : undefined,
    result.repriced.length > 0
      ? `Moved ${plural(result.repriced.length, "default rate", "default rates")} to today's.`
      : undefined,
  ].filter((part) => part !== undefined)
  if (parts.length > 0) return parts.join(" ")
  return kind === "models"
    ? "The provider lists nothing new."
    : "Every default rate is already current."
}
