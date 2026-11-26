import { Link } from "@tanstack/react-router"
import { useState } from "react"
import { Button } from "@/design-system/actions/Button"
import { EmptyState } from "@/design-system/feedback/EmptyState"
import { ErrorBanner } from "@/design-system/feedback/ErrorBanner"
import { PageLoading } from "@/design-system/feedback/PageLoading"
import { Chip } from "@/design-system/indicators/Chip"
import { PageIntro } from "@/design-system/layout/PageIntro"
import { KpiCell } from "@/design-system/metrics/KpiCell"
import { KpiStrip } from "@/design-system/metrics/KpiStrip"
import { MemberAvatar } from "@/features/account/avatar/MemberAvatar"
import { membershipLabel } from "@/features/organization/roles"
import { ChangeRoleDialog } from "@/features/people/ChangeRoleDialog"
import { PersonBudget } from "@/features/people/PersonBudget"
import { PersonModels } from "@/features/people/PersonModels"
import { PersonSessions } from "@/features/people/PersonSessions"
import { activeKeysOf } from "@/features/people/people"
import { personActions } from "@/features/people/personActions"
import { RemovePersonDialog } from "@/features/people/RemovePersonDialog"
import { useMonth, usePeople } from "@/features/people/usePeople"
import { useKeys } from "@/shared/api/apiKeys"
import {
  useOrganizationContext,
  useOrganizationMembers,
} from "@/shared/api/organizations"
import { BackLink } from "@/shared/components/BackLink"
import {
  formatNumber,
  formatRelative,
  formatTokens,
  formatUsd,
} from "@/shared/helpers/format"
import { useSurfaces } from "@/shared/hooks/useDeployment"

/**
 * One person: their month as Home reads it (spend, tokens, sessions,
 * refusals) with their API keys beside it, the spend rule that applies to
 * them, the models the month went to, and their sessions. A deployment that
 * records no sessions shows none of them.
 */
export function PersonPage({ personId }: { personId: string }) {
  const hasSessions = useSurfaces()("traces")
  const { window, now } = useMonth()
  const people = usePeople(window, now, { personId })
  const context = useOrganizationContext().data
  const members = useOrganizationMembers().data ?? []
  // The list `usePeople` counts from, for when a key was last used.
  const keys = useKeys()
  const [isRemoveOpen, setRemoveOpen] = useState(false)
  const [isRoleOpen, setRoleOpen] = useState(false)
  const person = people.rows.find((row) => row.id === personId)

  if (people.isLoading) return <PageLoading />
  if (!person) {
    return people.error ? (
      <>
        <BackLink to="/people" label="People" />
        <ErrorBanner error={people.error} />
      </>
    ) : (
      <>
        <BackLink to="/people" label="People" />
        <EmptyState
          title="No person at this link"
          description="They may have been removed from the organization."
        />
      </>
    )
  }

  const actions = personActions({ member: person.member, context, members })
  const canChangeRole =
    actions.changeRole !== undefined && !actions.changeRole.blocked
  const canRemove = actions.remove !== undefined && !actions.remove.blocked
  // Usage and sessions reach a member or viewer for their own workspaces only.
  const hidden = person.isOutsideView ? "—" : undefined
  const hiddenWhy = person.isOutsideView ? "Not in your workspaces" : undefined
  const teams = person.teams.length > 0 ? person.teams.join(", ") : undefined
  const lastUsed = activeKeysOf(keys.data ?? [], person.member, now)
    .map((key) => key.last_used_at)
    .filter((at): at is string => typeof at === "string")
    .sort()
    .at(-1)

  return (
    <>
      <BackLink to="/people" label="People" />
      <PageIntro
        title={person.name}
        action={
          canChangeRole || canRemove ? (
            <span className="flex flex-wrap gap-2">
              {canChangeRole ? (
                <Button onPress={() => setRoleOpen(true)}>Change role</Button>
              ) : null}
              {canRemove ? (
                <Button variant="danger" onPress={() => setRemoveOpen(true)}>
                  Remove from organization
                </Button>
              ) : null}
            </span>
          ) : undefined
        }
        detail={
          <span className="inline-flex flex-wrap items-center gap-2">
            <MemberAvatar
              userId={person.member.user_id}
              face={person.member.avatar_face}
            />
            {[
              membershipLabel(person.role),
              teams
                ? `${teams} ${person.teams.length > 1 ? "workspaces" : "workspace"}`
                : "",
              person.email,
            ]
              .filter(Boolean)
              .join(" · ")}
            {person.isInvited ? <Chip tone="accent">Invited</Chip> : null}
          </span>
        }
      />
      <ErrorBanner error={people.error} />

      <KpiStrip isEmpty={false} columns={hasSessions ? 5 : 4}>
        <KpiCell
          label="Spent this month"
          value={
            hidden ?? (person.unread.spend ? "—" : formatUsd(person.spend))
          }
          subline={hiddenWhy ?? window.label}
        />
        <KpiCell
          label="Tokens this month"
          value={
            hidden ?? (person.unread.spend ? "—" : formatTokens(person.tokens))
          }
          subline={
            hiddenWhy ??
            (person.unread.spend
              ? window.label
              : `${formatNumber(person.spendRow?.requests ?? 0)} requests`)
          }
        />
        {hasSessions ? (
          <KpiCell
            label="Sessions this month"
            value={
              hidden ??
              (person.unread.sessions ? "—" : formatNumber(person.sessions))
            }
            subline={hiddenWhy ?? window.label}
          />
        ) : null}
        <KpiCell
          label="Refused requests"
          value={
            hidden ??
            (person.unread.refusals ? "—" : formatNumber(person.refusals))
          }
          severity={
            person.refusals > 0 && !person.unread.refusals
              ? { status: "warn", word: "Review" }
              : undefined
          }
          subline={hiddenWhy ?? "This month"}
        />
        <KpiCell
          label="Active API keys"
          value={
            person.unread.keys || person.apiKeys === undefined
              ? "—"
              : formatNumber(person.apiKeys)
          }
          subline={
            person.unread.keys
              ? window.label
              : person.apiKeys === undefined
                ? "Not shown to you"
                : person.apiKeys === 0
                  ? "None active"
                  : lastUsed
                    ? `Used ${formatRelative(lastUsed)}`
                    : "Never used"
          }
          // On the line under the figure rather than as a graphic: no other
          // cell here has one, and one cell with a graphic row drops its
          // neighbors' figures off its baseline.
          delta={
            person.apiKeys && person.userId ? (
              <Link
                to="/keys"
                search={{ owner: person.userId }}
                className="shrink-0 text-link underline max-sm:inline-flex max-sm:min-h-11 max-sm:items-center"
              >
                See their API keys
              </Link>
            ) : undefined
          }
        />
      </KpiStrip>

      <PersonBudget
        person={person}
        ceilings={people.ceilings}
        isReadFailed={Boolean(people.error)}
      />

      {person.userId && !person.isOutsideView ? (
        <>
          <PersonModels userId={person.userId} window={window} />
          {hasSessions ? (
            <PersonSessions
              person={person}
              userId={person.userId}
              window={window}
            />
          ) : null}
        </>
      ) : (
        <p className="text-body text-muted">
          {person.isOutsideView
            ? hasSessions
              ? "Their models and sessions are in workspaces you are not in."
              : "Their models are in workspaces you are not in."
            : hasSessions
              ? "No sessions yet."
              : "No requests yet."}
        </p>
      )}

      {isRoleOpen ? (
        <ChangeRoleDialog person={person} onClose={() => setRoleOpen(false)} />
      ) : null}
      <RemovePersonDialog
        person={person}
        isOpen={isRemoveOpen}
        onOpenChange={setRemoveOpen}
      />
    </>
  )
}
