/**
 * What a member's or viewer's Home says, as pure functions over what the page
 * read: the limit figure and the Needs attention band. Every input is a read
 * the server gives anyone in the organization; nothing here needs an owner's.
 */

import type {
  ApiKey,
  OrganizationMember,
  WorkspaceBudgetDefault,
} from "@/client"
import {
  type AttentionInput,
  type AttentionState,
  countedFailures,
  readsOf,
} from "@/features/home/attention"
import { keyState } from "@/features/keys/keyList"
import { perWords } from "@/features/spend/limits"
import { formatNumber, formatPct } from "@/shared/helpers/format"

const DAY_MS = 86_400_000
/** How far ahead a key's expiry is worth saying, and how long after. */
const EXPIRY_DAYS = 7

/**
 * The caller's limit in a team, from what a member may read: their own row on
 * the roster carries the amount of the ceiling on them there, and the team's
 * per-person default carries its period. A roster ceiling drawn from that
 * default (the same budget) is the default, so it reads as everyone's.
 *
 * Neither is a limit set on them across the organization ("Set spend limit…"
 * on a person), which only owners and admins may read. So where neither caps
 * them the limit is unseen, never none: one may still be set.
 */
export type MemberLimit =
  | { kind: "own"; amount: number }
  | {
      kind: "default"
      amount: number
      per: string
      /** Where its current period began, or "" where that is not known. */
      periodStart: string
    }
  | { kind: "unseen" }

type Default = Pick<
  WorkspaceBudgetDefault,
  | "budget_id"
  | "max_budget"
  | "provider_key_id"
  | "reset_alignment"
  | "budget_duration_sec"
>

/**
 * Whether any of the caller's keys can make a request now: not disabled and
 * not past its expiry, as the API keys page counts it. A member whose only key
 * has expired is where a first run starts again.
 */
export function hasActiveKey(
  keys: readonly Pick<ApiKey, "is_active" | "expires_at">[],
  now: number,
): boolean {
  return keys.some((key) => keyState(key as ApiKey, now) === "active")
}

/**
 * What the figure on a member's Home is. A ceiling set on the person in this
 * workspace is theirs; anything else is the workspace's amount for each
 * person, which is not a claim about the limit that binds them.
 */
export function memberLimitLabel(
  limit: MemberLimit | undefined,
  workspace: string,
): string {
  return limit?.kind === "own"
    ? `Your limit in ${workspace}`
    : limit?.kind === "unseen"
      ? "Your limit"
      : "Limit per person"
}

export const LOWER_LIMIT_NOTE = "A limit set on you can be lower."

export const UNSEEN_LIMIT_NOTE =
  "Only owners and admins can see a limit set on you."

/**
 * A refusal over spend, among a member's refused requests (403 is also a
 * blocked caller and a model outside a key's list). The server names it in the
 * refusal's message and in no field of its own.
 */
export function isSpendRefusal(message: string | null | undefined): boolean {
  // Matches the gateway's "Member has $X of budget remaining, less than this request's estimated cost".
  return /budget/i.test(message ?? "")
}

/**
 * A refusal over a limit on the member themselves. The gateway names that
 * subject "Member", and a workspace's, the organization's or a key's limit by
 * its own name.
 */
export function isMemberLimitRefusal(
  message: string | null | undefined,
): boolean {
  return /^Member has\b/.test(message ?? "")
}

/**
 * The two lines under an unseen limit, once the member's refusals over
 * `words` (the window) are read; undefined until then, so they do not change
 * under the reader. A refusal over a limit on them says one is set, and then
 * who can see it is the line to give; otherwise the workspace's answer for
 * everyone is. Each note fits the two lines the cell gives it at 1280px.
 */
export function unseenLimitLines({
  limit,
  refusals,
  words,
  workspace,
}: {
  limit: MemberLimit | undefined
  /** The messages of the member's refused requests; undefined until read. */
  refusals: readonly (string | null | undefined)[] | undefined
  words: string
  workspace: string
}): { subline: string; note: string } | undefined {
  if (limit?.kind !== "unseen" || refusals === undefined) return undefined
  return refusals.some(isMemberLimitRefusal)
    ? {
        subline: "Only owners and admins see it",
        note: `A limit set on you refused requests ${words}.`,
      }
    : { subline: `None for everyone in ${workspace}`, note: UNSEEN_LIMIT_NOTE }
}

/**
 * Said under the workspace's per-person figure when it cannot be what refused
 * the member: they were refused over spend in its current period, with less
 * spent in that period than the figure. `spent` and `spendRefusals` are read
 * from `since`, and say nothing unless that is where the period began.
 */
export function lowerLimitNote({
  limit,
  since,
  spent,
  spendRefusals,
}: {
  limit: MemberLimit | undefined
  since: string
  spent: number
  spendRefusals: number
}): string | undefined {
  if (limit?.kind !== "default" || !limit.periodStart) return undefined
  if (Date.parse(since) !== Date.parse(limit.periodStart)) return undefined
  return spendRefusals > 0 && spent < limit.amount
    ? LOWER_LIMIT_NOTE
    : undefined
}

/**
 * Where a default's current period began: the UTC day, ISO week (from Monday)
 * or month the gateway counts it over. A period that rolls from a request
 * starts where the server alone knows, so it has "".
 */
function periodStartOf(alignment: string | null, now: number): string {
  const at = new Date(now)
  const year = at.getUTCFullYear()
  const month = at.getUTCMonth()
  const day = at.getUTCDate()
  const start =
    alignment === "calendar_day"
      ? Date.UTC(year, month, day)
      : alignment === "calendar_week"
        ? Date.UTC(year, month, day - ((at.getUTCDay() + 6) % 7))
        : alignment === "calendar_month"
          ? Date.UTC(year, month, 1)
          : undefined
  return start === undefined ? "" : new Date(start).toISOString()
}

export function memberLimit({
  member,
  workspaceId,
  defaults,
  now,
}: {
  member: Pick<OrganizationMember, "workspaces"> | undefined
  workspaceId: string
  defaults: readonly Default[]
  now: number
}): MemberLimit {
  // A default on a provider key caps one provider, not the person.
  const teamDefault = defaults.find((one) => one.provider_key_id === null)
  const place = (member?.workspaces ?? []).find(
    (one) => one.workspace_id === workspaceId,
  )
  const ceiling = place?.ceiling
  const fromDefault = (amount: number, policy: Default): MemberLimit => ({
    kind: "default",
    amount,
    per: perWords(policy),
    periodStart: periodStartOf(policy.reset_alignment, now),
  })
  // $0 is a limit: nothing may be spent. Only a null amount caps nothing.
  const own = ceiling?.max_budget
  if (ceiling && own != null) {
    return teamDefault && teamDefault.budget_id === ceiling.budget_id
      ? fromDefault(own, teamDefault)
      : { kind: "own", amount: own }
  }
  const everyone = teamDefault?.max_budget
  if (teamDefault && everyone != null) return fromDefault(everyone, teamDefault)
  return { kind: "unseen" }
}

/** Who the caller bills as, as `useCurrentOwner` resolves it. */
export interface Owner {
  userId: string | undefined
  isPending: boolean
  isFailed: boolean
}

/**
 * A read of the caller's own requests. It cannot be asked until their billing
 * id is known, so it fails with the roster; a caller with none has no
 * requests filed under them, which is an answer rather than a wait.
 */
export function ownReadState(
  owner: Owner,
  query: { data: unknown; isError: boolean },
): AttentionState {
  if (owner.isFailed) return "failed"
  if (owner.isPending) return "loading"
  if (!owner.userId) return "ready"
  return query.data !== undefined
    ? "ready"
    : query.isError
      ? "failed"
      : "loading"
}

/** The caller's row on the roster, by either id their spend is filed under. */
export function rosterRowOf<
  T extends Pick<OrganizationMember, "user_id" | "attribution_user_id">,
>(members: readonly T[], userId: string | undefined): T | undefined {
  if (!userId) return undefined
  return members.find(
    (one) => one.user_id === userId || one.attribution_user_id === userId,
  )
}

interface MemberAttentionItem {
  id: string
  isUrgent: boolean
  text: string
  action: {
    label: string
    to: "/requests" | "/keys"
    search?: Record<string, string>
  }
}

export type MemberAttentionKey = Pick<
  ApiKey,
  | "id"
  | "key_name"
  | "key_suffix"
  | "expires_at"
  | "is_active"
  | "disabled_reason"
>

export interface MemberAttentionInput {
  now: number
  /** The caller, for the Requests filter an item opens. */
  userId: string | undefined
  states: {
    refusals: AttentionState
    failures: AttentionState
    keys: AttentionState
  }
  /** The caller's requests the gateway refused with a 403 in the last 24 hours. */
  refused: number
  /** The caller's requests in the last 24 hours, by error class. */
  day: AttentionInput["day"]
  /** The caller's own keys in the team. */
  keys: readonly MemberAttentionKey[]
}

export interface MemberAttention {
  isLoading: boolean
  items: MemberAttentionItem[]
  /** One line per source that could not be read. */
  notChecked: string[]
  /** What to say when nothing qualifies and every source answered. */
  quiet: string | undefined
}

const SOURCE_NAMES = {
  refusals: "Refusals",
  failures: "Failures",
  keys: "Your keys",
} as const

const DATE = new Intl.DateTimeFormat("en-US", {
  month: "short",
  day: "numeric",
})

const keyName = (key: MemberAttentionKey) =>
  key.key_name || `ending ${key.key_suffix ?? ""}`.trim()

/** Whole local days from `now` to `at`: 0 today, 1 tomorrow. */
function calendarDaysUntil(at: number, now: number): number {
  const day = (time: number) => {
    const date = new Date(time)
    return new Date(
      date.getFullYear(),
      date.getMonth(),
      date.getDate(),
    ).getTime()
  }
  return Math.round((day(at) - day(now)) / DAY_MS)
}

function keyItems(
  keys: readonly MemberAttentionKey[],
  now: number,
): MemberAttentionItem[] {
  const items: MemberAttentionItem[] = []
  for (const key of keys) {
    const name = keyName(key)
    if (key.disabled_reason) {
      items.push({
        id: `key-off:${key.id}`,
        isUrgent: true,
        text:
          key.disabled_reason === "revoked"
            ? `Your key ${name} was revoked.`
            : `Your key ${name} was disabled by an operator.`,
        action: {
          label: "Open in API keys",
          to: "/keys",
          search: { state: "disabled" },
        },
      })
      continue
    }
    if (!key.expires_at) continue
    const at = Date.parse(key.expires_at)
    if (at <= now) {
      if (now - at > EXPIRY_DAYS * DAY_MS) continue
      items.push({
        id: `key-expired:${key.id}`,
        isUrgent: true,
        text: `Your key ${name} expired on ${DATE.format(at)}.`,
        action: {
          label: "Open in API keys",
          to: "/keys",
          search: { state: "expired" },
        },
      })
      continue
    }
    if (!key.is_active || at - now > EXPIRY_DAYS * DAY_MS) continue
    const days = calendarDaysUntil(at, now)
    items.push({
      id: `key-expiring:${key.id}`,
      isUrgent: false,
      text: `Your key ${name} expires ${
        days <= 0 ? "today" : days === 1 ? "tomorrow" : `in ${days} days`
      }.`,
      action: { label: "Open in API keys", to: "/keys" },
    })
  }
  return items
}

/**
 * The band on a member's Home: their requests refused or failing in the last
 * 24 hours, and their own keys about to stop. Each item carries one action.
 * Nothing about limits: how much of one is left is not readable to a member.
 */
export function memberAttention(input: MemberAttentionInput): MemberAttention {
  const { states, now } = input
  const { isLoading, notChecked } = readsOf(states, SOURCE_NAMES)
  const mine: Record<string, string> = input.userId
    ? { user_id: input.userId }
    : {}
  const items: MemberAttentionItem[] = []

  if (states.refusals === "ready" && input.refused > 0) {
    items.push({
      id: "refused",
      isUrgent: true,
      text: `${formatNumber(input.refused)} of your requests ${input.refused === 1 ? "was" : "were"} refused in the last 24 hours.`,
      action: {
        label: "See the refusals",
        to: "/requests",
        search: { status_code: "403", range: "24h", ...mine },
      },
    })
  }
  const failures =
    states.failures === "ready" ? countedFailures(input.day) : undefined
  if (failures) {
    items.push({
      id: "failures",
      isUrgent: failures.isUrgent,
      text: `${formatPct(failures.rate)} of your requests failed in the last 24 hours.`,
      action: {
        label: "See the failures",
        to: "/requests",
        search: { status: "error", range: "24h", ...mine },
      },
    })
  }
  if (states.keys === "ready") items.push(...keyItems(input.keys, now))

  // Most urgent first; the order above holds within each.
  items.sort((a, b) => Number(b.isUrgent) - Number(a.isUrgent))
  return {
    isLoading,
    items,
    notChecked,
    quiet:
      !isLoading && items.length === 0 && notChecked.length === 0
        ? "Nothing needs you right now. None of your requests was refused in the last 24 hours, and none of your keys expires this week."
        : undefined,
  }
}
