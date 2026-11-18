import { Link } from "@tanstack/react-router"
import { useState } from "react"
import type { OrganizationSpendCeiling } from "@/client"
import { Button } from "@/design-system/actions/Button"
import { InfoBanner } from "@/design-system/feedback/InfoBanner"
import { SpendMeter } from "@/design-system/metrics/SpendMeter"
import { runsOutAt } from "@/features/people/forecast"
import { type PersonRow, rosterLimits } from "@/features/people/people"
import { ruleIdForCeiling } from "@/features/rules/rules"
import { limitDialogCopy } from "@/features/spend/limitCopy"
import {
  type LimitTarget,
  limitLine,
  limitsOnPerson,
} from "@/features/spend/limits"
import { SetLimitDialog } from "@/features/spend/SetLimitDialog"
import { Band } from "@/shared/components/Band"
import { formatUsd } from "@/shared/helpers/format"

const DAY = new Intl.DateTimeFormat("en-US", {
  weekday: "long",
  month: "short",
  day: "numeric",
})

/**
 * The spend rule that applies to one person: the limit closest to its cap of
 * every one that reaches them, how much of it is gone, and when it runs out at
 * the pace of its period so far. Owners and admins set or change their own
 * limit here. Anyone else reads what the roster gives them: the amount on the
 * person in each workspace, and nothing of its period or what is left.
 */
export function PersonBudget({
  person,
  ceilings,
  isReadFailed = false,
}: {
  person: PersonRow
  /** Every ceiling, for a caller who may read them; undefined otherwise. */
  ceilings: readonly OrganizationSpendCeiling[] | undefined
  /**
   * A read behind the page failed, so the limit shown may not be the one the
   * person has, and setting one could add a second.
   */
  isReadFailed?: boolean
}) {
  const [now] = useState(() => Date.now())
  const [editing, setEditing] = useState<LimitTarget>()
  const first = person.name.split(" ")[0]
  const memberId = person.member.organization_member_id

  if (ceilings === undefined) {
    // What the roster gives anyone: the amount on them in each workspace.
    const amounts = rosterLimits(person.member)
    return (
      <Band
        title="Spend limit"
        description="Each workspace’s limit for each person."
      >
        {amounts.length > 0 ? (
          <ul className="flex max-w-md flex-col border-t border-border">
            {amounts.map((one) => (
              <li
                key={one.workspaceId}
                className="flex min-h-11 items-center justify-between gap-4 border-b border-border-subtle py-2"
              >
                <span className="text-body">{one.team}</span>
                <span className="text-mono-caption text-foreground">
                  {formatUsd(one.amount)}
                </span>
              </li>
            ))}
          </ul>
        ) : (
          <p className="text-body text-muted">
            No workspace sets a limit for each person.
          </p>
        )}
        <InfoBanner>
          Only owners and admins can see each limit’s name, how often it resets
          and how much is used.
        </InfoBanner>
      </Band>
    )
  }

  const limits = limitsOnPerson(ceilings, person.member)
  const binding = limits.binding
  const target: LimitTarget | undefined = memberId
    ? {
        scope: "org_member",
        scopeId: memberId,
        name: person.name,
        own: limits.own,
      }
    : undefined
  const outAt = binding
    ? runsOutAt({
        spent: binding.spent,
        max: binding.max,
        periodStart: binding.ceiling.period_start,
        periodEnd: binding.ceiling.period_end,
        now,
      })
    : undefined

  return (
    <Band
      title="Spend limit"
      description="The spend limit that applies to this person."
      action={
        target && !person.isInvited && !isReadFailed ? (
          <Button onPress={() => setEditing(target)}>
            {limits.own ? "Change spend limit" : "Set spend limit"}
          </Button>
        ) : undefined
      }
    >
      {binding ? (
        <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
          <div className="flex min-w-0 flex-col gap-1">
            <Link
              to="/rules/$ruleId"
              params={{ ruleId: ruleIdForCeiling(binding.ceiling) }}
              className="w-fit text-emphasis underline underline-offset-2 max-sm:inline-flex max-sm:min-h-11 max-sm:items-center"
            >
              {binding.ceiling.name ?? "Spend limit"}
              {binding.origin ? `, ${binding.origin}` : ""}
            </Link>
            <span className="text-caption">
              {limits.count > 1
                ? `The tightest of ${limits.count} limits that reach ${first}. `
                : ""}
              {binding.max <= 0
                ? "The limit is $0: nothing may be spent, so every request is refused."
                : outAt === undefined
                  ? "On pace to stay under the limit this period."
                  : outAt <= now
                    ? "The limit is used up; requests are refused until it resets."
                    : `At this period’s pace the limit runs out ${DAY.format(outAt)}.`}
            </span>
          </div>
          <div className="flex shrink-0 flex-col gap-1.5 md:w-72">
            <SpendMeter
              spent={binding.spent}
              allocated={binding.max}
              ariaLabel={`${person.name}’s spend against the limit`}
              className="w-full"
            />
            <span className="text-caption tabular-nums">
              {limitLine(binding)}
            </span>
          </div>
        </div>
      ) : isReadFailed ? null : (
        <p className="text-body text-muted">
          No spend limit applies to {first}.
        </p>
      )}
      {editing ? (
        <SetLimitDialog
          key={editing.scopeId}
          target={editing}
          {...limitDialogCopy(editing)}
          onClose={() => setEditing(undefined)}
        />
      ) : null}
    </Band>
  )
}
