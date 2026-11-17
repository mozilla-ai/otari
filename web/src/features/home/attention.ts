/**
 * What needs an owner or admin on Home, as one pure function over what the
 * page has read. The hook that reads it lives in `useTeamAttention.ts`.
 *
 * A message earns its place only if all three hold:
 *
 * 1. it means someone's requests are refused or failing now, or will be before
 *    the next reset;
 * 2. an owner or admin can fix it;
 * 3. it carries one action: a dialog on Home, or a link to the exact place.
 *
 * | Type          | Fires when                                                | Severity        |
 * | ------------- | --------------------------------------------------------- | --------------- |
 * | 1 Account     | the edition that holds the organization's money says so   | its own         |
 * |               | (`overlayHomeAccount.tsx`; nothing in this build)         |                 |
 * | 2 Reached     | a limit's spend is at its maximum, period not ended       | urgent          |
 * | 3 Refused     | the gateway sent 403 in the last 24 hours, to someone     | urgent          |
 * |               | with no row of type 2                                     |                 |
 * | 4 Failures    | 5 or more server-side errors in 24 hours at 2% or more    | warning, urgent |
 * |               | of requests; urgent at 10% with 20 or more requests       | at 10% and 20   |
 * | 5 Pace       | organization or workspace limit, period of 7 days or more, | warning         |
 * |               | 10% gone, projected past the maximum, under 80% used      |                 |
 * | 6 Near limit  | 80% or more of a limit used, under 100%, period not ended | warning         |
 *
 * Failures count the error classes a person at the keyboard did not cause:
 * missing pricing (402), rate limits (429), provider errors (any 5xx) and the
 * unknown (no status code). The caller's own 4xx are not counted.
 *
 * A limit whose period ended counts as nothing spent: the gateway resets a
 * ceiling on the next request that reaches it, so until then it still carries
 * last period's figure.
 */

import type { ReactNode } from "react"
import type { OrganizationSpendCeiling } from "@/client"
import type { HomeAccount } from "@/features/home/overlayHomeAccount"
import {
  type LimitTarget,
  perWords,
  usedShare,
  wholePct,
} from "@/features/spend/limits"
import { paceOf } from "@/features/spend/pace"
import type { ReadState } from "@/features/spend/useSpend"
import { formatNumber, formatPct, formatUsd } from "@/shared/helpers/format"

const DAY_MS = 86_400_000
/** A limit this far along is worth a look before it refuses someone. */
const NEAR_LIMIT = 0.8
/** Near-limit rows shown before the rest fold. */
const NEAR_ROWS_SHOWN = 3

const MIN_FAILURES = 5
const MIN_FAILURE_RATE = 0.02
const URGENT_FAILURE_RATE = 0.1
const URGENT_FAILURE_REQUESTS = 20

/** The error classes a failure count includes; see the table above. */
const COUNTED_ERROR_CLASSES: readonly string[] = [
  "pricing",
  "rate_limit",
  "provider_error",
  "unknown",
]

export type AttentionSource = "limits" | "refusals" | "failures" | "account"
export type AttentionState = ReadState

/** Whether a source is still being read, and a sentence for each that failed. */
export function readsOf<Source extends string>(
  states: Record<Source, AttentionState>,
  names: Record<Source, string>,
): { isLoading: boolean; notChecked: string[] } {
  const sources = Object.keys(states) as Source[]
  return {
    isLoading: sources.some((source) => states[source] === "loading"),
    notChecked: sources
      .filter((source) => states[source] === "failed")
      .map(
        (source) =>
          `${names[source]} could not be checked, so this list may be incomplete.`,
      ),
  }
}

const SOURCE_NAMES: Record<Exclude<AttentionSource, "account">, string> = {
  limits: "Limits",
  refusals: "Refusals",
  failures: "Failures",
}

interface AttentionAction {
  label: string
  /** A limit to change in place. */
  limit?: LimitTarget
  to?: "/rules" | "/requests"
  search?: Record<string, string>
  /** A control an edition renders itself (`overlayHomeAccount.tsx`). */
  node?: ReactNode
}

export interface AttentionItem {
  id: string
  isUrgent: boolean
  text: string
  action: AttentionAction
}

export interface AttentionInput {
  now: number
  states: Record<AttentionSource, AttentionState>
  ceilings: readonly OrganizationSpendCeiling[]
  /** Who a ceiling is on, by scope id, as the sentence names them. */
  subjects: ReadonlyMap<string, string>
  workspaceNames: ReadonlyMap<string, string>
  /** What a ceiling's dialog edits, by ceiling id; none where Home has no row. */
  targets: ReadonlyMap<string, LimitTarget>
  /** The user ids behind a person-level scope id (member, workspace member). */
  peopleOf: ReadonlyMap<string, readonly string[]>
  /** What the edition holding the organization's money adds, if any. */
  account: Pick<HomeAccount, "sourceName" | "attention" | "quietClause">
  day: {
    requests: number
    errors: readonly { errorClass: string; requests: number }[]
  }
  /** The 403s of the last 24 hours, by user; a null id is nobody in particular. */
  refusals: readonly { userId: string | null; name: string; requests: number }[]
}

export interface AttentionResult {
  /** Loading until every source has answered, or failed. */
  isLoading: boolean
  items: AttentionItem[]
  /** Near-limit rows past the third, shown on "Show all". */
  folded: AttentionItem[]
  foldText: string
  /** One line per source that could not be read. */
  notChecked: string[]
  /** What to say when nothing qualifies and every source answered. */
  quiet: { lead: string; detail: string } | undefined
}

const SCOPE_RANK: Record<string, number> = {
  organization: 0,
  workspace: 1,
  org_member: 2,
  workspace_member: 2,
  api_token: 3,
}

interface Ranked {
  item: AttentionItem
  /** The row's type in the table above. */
  type: number
  scopeRank: number
  used: number
}

const plural = (n: number, one: string, many: string) =>
  `${formatNumber(n)} ${n === 1 ? one : many}`

function daysUntil(end: string, now: number): number {
  return Math.ceil((Date.parse(end) - now) / DAY_MS)
}

function resetWords(ceiling: OrganizationSpendCeiling, now: number): string {
  if (!ceiling.period_end) return ""
  const days = daysUntil(ceiling.period_end, now)
  if (days <= 0) return ""
  return days === 1 ? " Resets tomorrow." : ` Resets in ${days} days.`
}

const DAY_LABEL = new Intl.DateTimeFormat("en-US", {
  month: "short",
  day: "numeric",
})

/**
 * Who a limit is on when the page has no name for it: a key the caller's list
 * does not carry, someone off the roster, a workspace it has not read.
 */
const UNNAMED: Record<string, string> = {
  organization: "The organization",
  workspace: "A workspace",
  api_token: "An API key",
}

function limitRows(input: AttentionInput): Ranked[] {
  const { now } = input
  const rows: Ranked[] = []
  for (const ceiling of input.ceilings) {
    const max = ceiling.max_budget
    // $0 is a limit, and one that is always reached.
    if (max == null || ceiling.provider_key_id !== null) continue
    const end = ceiling.period_end ? Date.parse(ceiling.period_end) : undefined
    if (end !== undefined && end < now) continue
    const used = usedShare({ spent: ceiling.current_spend, max })
    if (used <= 0) continue
    const scopeRank = SCOPE_RANK[ceiling.scope_type] ?? 4
    const who =
      input.subjects.get(ceiling.scope_id) ??
      UNNAMED[ceiling.scope_type] ??
      "Someone"
    const limit = `${formatUsd(max)} ${perWords(ceiling)}`.trim()
    const target = input.targets.get(ceiling.id)
    const dialog = (label: string): AttentionAction =>
      target ? { label, limit: target } : { label: "Open Rules", to: "/rules" }

    if (used >= 1) {
      const days = ceiling.period_end ? daysUntil(ceiling.period_end, now) : 0
      const until =
        !ceiling.period_end || days <= 0
          ? "the limit is raised"
          : days === 1
            ? "it resets tomorrow"
            : `it resets in ${days} days`
      rows.push({
        type: 2,
        scopeRank,
        used,
        item: {
          id: ceiling.id,
          isUrgent: true,
          text: `${who} is at the limit of ${limit}. Requests are refused until ${until}.`,
          action: dialog("Raise the limit"),
        },
      })
      continue
    }
    if (used >= NEAR_LIMIT) {
      rows.push({
        type: 6,
        scopeRank,
        used,
        item: {
          id: ceiling.id,
          isUrgent: false,
          text: `${who} has used ${wholePct(used)} of ${limit}.${resetWords(ceiling, now)}`,
          action: dialog("Raise the limit"),
        },
      })
      continue
    }
    if (
      ceiling.scope_type !== "organization" &&
      ceiling.scope_type !== "workspace"
    )
      continue
    const pace = paceOf({
      spent: ceiling.current_spend,
      max,
      periodStart: ceiling.period_start,
      periodEnd: ceiling.period_end,
      now,
    })
    const runsOut = pace?.runsOutAt
    if (pace === undefined || runsOut === undefined) continue
    const before = Math.max(1, Math.round((pace.end - runsOut) / DAY_MS))
    const date = DAY_LABEL.format(runsOut)
    const isOrganization = ceiling.scope_type === "organization"
    const name = input.workspaceNames.get(ceiling.scope_id)
    if (!isOrganization && !name) continue
    rows.push({
      type: 5,
      scopeRank,
      used,
      item: {
        id: ceiling.id,
        isUrgent: false,
        text: isOrganization
          ? `At this pace the team budget of ${limit} runs out around ${date}, ${plural(before, "day", "days")} before it resets.`
          : `At this pace the ${name} workspace’s limit of ${limit} runs out around ${date}, ${plural(before, "day", "days")} before it resets.`,
        action: dialog(isOrganization ? "Change the limit" : "Raise the limit"),
      },
    })
  }
  return rows
}

function accountRow(input: AttentionInput): Ranked | undefined {
  const row = input.account.attention
  if (!row) return undefined
  return {
    type: 1,
    scopeRank: 0,
    used: row.isUrgent ? 1 : 0,
    item: {
      id: row.id,
      isUrgent: row.isUrgent,
      text: row.text,
      action: { label: "", node: row.action },
    },
  }
}

function refusedRow(
  input: AttentionInput,
  reached: ReadonlySet<string>,
): Ranked | undefined {
  const rows = input.refusals.filter(
    (row) => row.requests > 0 && !(row.userId && reached.has(row.userId)),
  )
  const total = rows.reduce((sum, row) => sum + row.requests, 0)
  if (total === 0) return undefined
  const people = rows.filter((row) => row.userId)
  const isOnePerson = people.length === 1 && people.length === rows.length
  const text = isOnePerson
    ? total === 1
      ? `The gateway refused a request from ${people[0].name} in the last 24 hours.`
      : `The gateway refused ${formatNumber(total)} requests from ${people[0].name} in the last 24 hours.`
    : people.length > 1
      ? `The gateway refused ${formatNumber(total)} requests from ${people.length} people in the last 24 hours.`
      : total === 1
        ? "The gateway refused a request in the last 24 hours."
        : `The gateway refused ${formatNumber(total)} requests in the last 24 hours.`
  const userId = isOnePerson ? people[0].userId : undefined
  return {
    type: 3,
    scopeRank: 0,
    used: 0,
    item: {
      id: "refused",
      isUrgent: true,
      text,
      action: {
        label: "See the refusals",
        to: "/requests",
        search: {
          status_code: "403",
          range: "24h",
          ...(userId ? { user_id: userId } : {}),
        },
      },
    },
  }
}

/**
 * The last 24 hours' failures, when they are past the thresholds in the table
 * above; undefined below them. Also a member's Home, over their own requests.
 */
export function countedFailures(
  day: AttentionInput["day"],
): { failed: number; rate: number; isUrgent: boolean } | undefined {
  const { requests, errors } = day
  const failed = errors
    .filter((row) => COUNTED_ERROR_CLASSES.includes(row.errorClass))
    .reduce((sum, row) => sum + row.requests, 0)
  if (requests <= 0 || failed < MIN_FAILURES) return undefined
  const rate = failed / requests
  if (rate < MIN_FAILURE_RATE) return undefined
  return {
    failed,
    rate,
    isUrgent:
      rate >= URGENT_FAILURE_RATE && requests >= URGENT_FAILURE_REQUESTS,
  }
}

function failuresRow(input: AttentionInput): Ranked | undefined {
  const { requests } = input.day
  const counted = countedFailures(input.day)
  if (!counted) return undefined
  const { failed, rate } = counted
  return {
    type: 4,
    scopeRank: 0,
    used: rate,
    item: {
      id: "failures",
      isUrgent: counted.isUrgent,
      text: `${formatNumber(failed)} of ${formatNumber(requests)} requests failed in the last 24 hours (${formatPct(rate)}).`,
      action: {
        label: "See the failures",
        to: "/requests",
        search: { status: "error", range: "24h" },
      },
    },
  }
}

function quietLine(input: AttentionInput): { lead: string; detail: string } {
  const clauses = [
    `No limit is past ${wholePct(NEAR_LIMIT)}`,
    "nothing was refused in the last 24 hours",
  ]
  if (input.account.quietClause) clauses.push(input.account.quietClause)
  const last = clauses.pop()
  const detail =
    clauses.length === 1
      ? `${clauses[0]} and ${last}.`
      : `${clauses.join(", ")}, and ${last}.`
  return { lead: "Nothing needs you right now.", detail }
}

/** The ordered list Home's attention band shows, and what to say when empty. */
export function attention(input: AttentionInput): AttentionResult {
  const { states } = input
  const { isLoading, notChecked } = readsOf(states, {
    ...SOURCE_NAMES,
    account: input.account.sourceName,
  })

  const limits = states.limits === "ready" ? limitRows(input) : []
  const reached = new Set<string>()
  for (const row of limits) {
    if (row.type !== 2) continue
    const ceiling = input.ceilings.find((one) => one.id === row.item.id)
    for (const id of input.peopleOf.get(ceiling?.scope_id ?? "") ?? []) {
      reached.add(id)
    }
  }
  const ranked: Ranked[] = [
    ...(states.account === "ready" ? [accountRow(input)] : []),
    ...limits,
    ...(states.refusals === "ready" ? [refusedRow(input, reached)] : []),
    ...(states.failures === "ready" ? [failuresRow(input)] : []),
  ].filter((row): row is Ranked => row !== undefined)

  ranked.sort(
    (a, b) =>
      Number(b.item.isUrgent) - Number(a.item.isUrgent) ||
      a.type - b.type ||
      a.scopeRank - b.scopeRank ||
      b.used - a.used,
  )

  const near = ranked.filter((row) => row.type === 6)
  const folded = near.slice(NEAR_ROWS_SHOWN).map((row) => row.item)
  const hidden = new Set(folded)
  const items = ranked.map((row) => row.item).filter((one) => !hidden.has(one))
  return {
    isLoading,
    items,
    folded,
    foldText: `${plural(folded.length, "more limit is", "more limits are")} past ${wholePct(NEAR_LIMIT)}.`,
    notChecked,
    quiet:
      !isLoading && ranked.length === 0 && notChecked.length === 0
        ? quietLine(input)
        : undefined,
  }
}
