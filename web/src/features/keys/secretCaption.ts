import type { CreateKeyResponse } from "@/client"
import { accessLabel } from "@/features/models/ModelScopeControl"
import { formatDate } from "@/shared/helpers/format"

export const isVirtualUser = (userId: string | null): boolean =>
  (userId ?? "").startsWith("apikey-")

/**
 * The line under the key, naming what was just made.
 *
 * Three facts, each dropped when it is not there rather than printed empty. The
 * owner is a name or nothing: `user_id` is an identifier, and an identifier in a
 * sentence is noise to the person who just chose the owner from a list. There is
 * deliberately no spend figure, though the frame it sits in is where one would
 * look for it: a key has no budget of its own, only its owner's, which is what
 * the paragraph above the table already says and links to.
 */
export function secretCaption(
  result: CreateKeyResponse,
  memberLabels: ReadonlyMap<string, string>,
): string {
  const owner =
    result.user_id && !isVirtualUser(result.user_id)
      ? memberLabels.get(result.user_id)
      : undefined
  return [
    owner ? `Owner ${owner}` : null,
    accessLabel(result.allowed_models).text,
    result.expires_at
      ? `expires ${formatDate(result.expires_at)}`
      : "never expires",
  ]
    .filter((part) => part !== null)
    .join(" · ")
}
