import type { ApiKey } from "@/client"
import {
  type MemberAttention,
  memberAttention,
  type Owner,
  ownReadState,
} from "@/features/home/memberHomeModel"
import { toMinute } from "@/features/spend/period"
import { readState } from "@/features/spend/useSpend"
import { useRefusalsSince, useUsageSummary } from "@/shared/api/usage"

const DAY_MS = 86_400_000

/**
 * Reads what a member's band needs, over the caller's own requests in the
 * team for the last 24 hours whatever window the page shows, and hands it to
 * `memberAttention`. Usage is read on the organization surface, which serves
 * a member their own workspaces.
 */
export function useMemberAttention({
  now,
  owner,
  workspaceId,
  keys,
}: {
  now: number
  owner: Owner
  workspaceId: string
  keys: { data: readonly ApiKey[] | undefined; isError: boolean }
}): MemberAttention {
  const { userId } = owner
  const since = new Date(toMinute(now) - DAY_MS).toISOString()
  const day = useUsageSummary(
    {
      workspace_id: workspaceId,
      start_date: since,
      user_id: userId ? [userId] : undefined,
    },
    "hour",
    ["status_code"],
    userId !== undefined,
    "organization",
  )
  // The summary filters take no status code, so the refusals are their own read.
  const refused = useRefusalsSince(
    since,
    workspaceId,
    userId !== undefined,
    userId,
  )
  return memberAttention({
    now,
    userId,
    states: {
      refusals: ownReadState(owner, refused),
      failures: ownReadState(owner, day),
      keys: readState(keys),
    },
    refused: refused.data?.totals.request_count ?? 0,
    day: {
      requests: day.data?.totals.request_count ?? 0,
      errors: (day.data?.errors_by_status_code ?? []).map((row) => ({
        errorClass: row.error_class,
        requests: row.requests,
      })),
    },
    keys: keys.data ?? [],
  })
}
