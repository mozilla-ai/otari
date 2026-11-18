import { Link, useNavigate } from "@tanstack/react-router"
import { useState } from "react"
import { FiUserCheck, FiUserMinus, FiUserPlus } from "react-icons/fi"
import { Button } from "@/design-system/actions/Button"
import { RowMenu, type RowMenuAction } from "@/design-system/actions/RowMenu"
import { DataTable, type DataTableColumn } from "@/design-system/data/DataTable"
import { EmptyMessage } from "@/design-system/feedback/EmptyMessage"
import { ErrorBanner } from "@/design-system/feedback/ErrorBanner"
import { useIsMobile } from "@/design-system/hooks/useIsMobile"
import { Chip } from "@/design-system/indicators/Chip"
import { PageIntro } from "@/design-system/layout/PageIntro"
import { TableScrollFrame } from "@/design-system/layout/TableScrollFrame"
import { MemberAvatar } from "@/features/account/avatar/MemberAvatar"
import { useIsOrganizationManager } from "@/features/organization/access"
import { membershipLabel } from "@/features/organization/roles"
import { ChangeRoleDialog } from "@/features/people/ChangeRoleDialog"
import { InvitePeopleDialog } from "@/features/people/InvitePeopleDialog"
import type { PersonRow } from "@/features/people/people"
import { personActions } from "@/features/people/personActions"
import { RemovePersonDialog } from "@/features/people/RemovePersonDialog"
import { useMonth, usePeople } from "@/features/people/usePeople"
import { TopModelCell } from "@/features/usage/TopModelCell"
import { useTopModels } from "@/features/usage/useTopModels"
import {
  useOrganizationContext,
  useOrganizationMembers,
} from "@/shared/api/organizations"
import { Band } from "@/shared/components/Band"
import { formatNumber, formatTokens, formatUsd } from "@/shared/helpers/format"
import { useSelectedWorkspace } from "@/shared/hooks/SelectedWorkspace"
import { useSurfaces } from "@/shared/hooks/useDeployment"

/** A figure whose read failed: unknown, not zero. */
const UNREAD = <span className="text-subtle">—</span>
const HIDDEN = <span className="text-subtle">Not in your workspaces</span>

/**
 * The people of the selected workspace, one list: the roster with what each
 * person spent this month, read the way Home reads it (spend, tokens, top
 * model). A row opens the person; an owner or admin changes a role or removes
 * someone from the row's menu. The whole organization's list is Members &
 * roles, which only owners and admins can open, and the workspace's own roster
 * (adding someone to it, their role in it) is Members.
 */
export function PeoplePage() {
  const navigate = useNavigate()
  const { window, now } = useMonth()
  const people = usePeople(window, now, { inSelectedWorkspace: true })
  const { selected } = useSelectedWorkspace()
  const canInvite = useIsOrganizationManager()
  const context = useOrganizationContext().data
  const members = useOrganizationMembers().data ?? []
  const isMobile = useIsMobile()
  // A deployment that records no sessions has none to count.
  const hasSessions = useSurfaces()("traces")
  const [isInviteOpen, setInviteOpen] = useState(false)
  const [changing, setChanging] = useState<PersonRow>()
  const [removing, setRemoving] = useState<PersonRow>()
  const spendRows = people.rows.flatMap((person) =>
    person.spendRow ? [person.spendRow] : [],
  )
  const topModels = useTopModels({
    rows: spendRows,
    cut: "people",
    window,
    workspaceId: people.workspaceId ?? "",
  })

  const menuFor = (person: PersonRow): RowMenuAction[] => {
    const actions = personActions({ member: person.member, context, members })
    return [
      ...(actions.changeRole
        ? [
            {
              id: "role",
              label: "Change role…",
              hint:
                actions.changeRole.blocked ??
                `${membershipLabel(person.role)} now`,
              icon: FiUserCheck,
              isDisabled: Boolean(actions.changeRole.blocked),
              run: () => setChanging(person),
            },
          ]
        : []),
      ...(actions.remove
        ? [
            {
              id: "remove",
              label: "Remove from organization…",
              hint:
                actions.remove.blocked ??
                (person.isInvited
                  ? "Their invitation stops working"
                  : "Takes them out of the organization"),
              icon: FiUserMinus,
              isDisabled: Boolean(actions.remove.blocked),
              isDestructive: true,
              run: () => setRemoving(person),
            },
          ]
        : []),
    ]
  }

  const columns: DataTableColumn<PersonRow>[] = [
    {
      id: "name",
      header: "Name",
      isRowHeader: true,
      // The name alone, as Home's People cut shows a person: the address is on
      // their page, and the month's columns need the width.
      cell: (person) => (
        <span className="flex min-w-0 md:max-w-[13rem] items-center gap-2">
          <MemberAvatar
            userId={person.member.user_id}
            face={person.member.avatar_face}
          />
          <span className="truncate text-body" title={person.email}>
            {person.name}
          </span>
        </span>
      ),
    },
    {
      id: "team",
      header: "Workspace",
      cell: (person) =>
        person.teams.length > 0 ? (
          person.teams.join(", ")
        ) : (
          <span className="text-subtle">None</span>
        ),
    },
    {
      id: "role",
      header: "Role",
      cell: (person) =>
        person.isInvited ? (
          <span className="flex items-center gap-2">
            {membershipLabel(person.role)}
            <Chip tone="accent">Invited</Chip>
          </span>
        ) : (
          membershipLabel(person.role)
        ),
    },
    {
      id: "keys",
      header: <span className="whitespace-nowrap">API keys</span>,
      align: "end",
      cell: (person) =>
        person.unread.keys ? (
          UNREAD
        ) : person.apiKeys === undefined ? (
          <span className="whitespace-nowrap text-subtle">Not shown</span>
        ) : (
          <span className="text-mono-caption text-foreground">
            {person.apiKeys}
          </span>
        ),
    },
    {
      id: "spend",
      header: <span className="whitespace-nowrap">Spent this month</span>,
      align: "end",
      cell: (person) =>
        person.isOutsideView ? (
          HIDDEN
        ) : person.unread.spend ? (
          UNREAD
        ) : (
          <span className="text-mono-caption text-foreground">
            {formatUsd(person.spend)}
          </span>
        ),
    },
    {
      id: "tokens",
      header: "Tokens",
      align: "end",
      cell: (person) =>
        person.isOutsideView ? null : person.unread.spend ? (
          UNREAD
        ) : (
          <span className="text-mono-caption text-foreground">
            {formatTokens(person.tokens)}
          </span>
        ),
    },
    {
      id: "model",
      header: "Top model",
      cell: (person) =>
        person.isOutsideView ? null : person.unread.spend ? (
          UNREAD
        ) : (
          <TopModelCell
            state={
              person.spendRow ? topModels.get(person.spendRow.id) : undefined
            }
            rowCost={person.spend}
            isInline={isMobile}
          />
        ),
    },
    ...(hasSessions
      ? ([
          {
            id: "sessions",
            header: "Sessions",
            align: "end",
            cell: (person) =>
              person.isOutsideView ? null : person.unread.sessions ? (
                UNREAD
              ) : (
                <span className="text-mono-caption text-foreground">
                  {formatNumber(person.sessions)}
                </span>
              ),
          },
        ] satisfies DataTableColumn<PersonRow>[])
      : []),
    {
      id: "refusals",
      header: "Refusals",
      align: "end",
      cell: (person) =>
        person.isOutsideView ? null : person.unread.refusals ? (
          UNREAD
        ) : (
          <span
            className={`text-mono-caption ${person.refusals > 0 ? "text-danger" : "text-subtle"}`}
          >
            {formatNumber(person.refusals)}
          </span>
        ),
    },
    ...(canInvite
      ? ([
          {
            id: "actions",
            header: <span className="sr-only">Actions</span>,
            align: "end",
            cell: (person) => (
              <RowMenu
                label={`Actions for ${person.name}`}
                actions={menuFor(person)}
              />
            ),
          },
        ] satisfies DataTableColumn<PersonRow>[])
      : []),
  ]

  return (
    <>
      <PageIntro
        title="People"
        action={
          canInvite ? (
            <Button variant="primary" onPress={() => setInviteOpen(true)}>
              <FiUserPlus aria-hidden className="size-4" />
              Invite people
            </Button>
          ) : undefined
        }
      />
      <ErrorBanner error={people.error} />
      <Band
        title={selected?.name ?? context?.organization.name ?? "Organization"}
        count={people.isLoading ? undefined : people.rows.length}
        description={`This month is ${window.label}. Refusals are requests the gateway refused: over a spend limit, blocked, or for a model the key may not use.${
          canInvite
            ? ""
            : ` You see spend${hasSessions ? " and sessions" : ""} in your own workspaces only.`
        }${
          people.rows.some((person) => person.apiKeys === undefined)
            ? " Each person sees only their own API keys."
            : ""
        }${
          people.isSessionCountComplete
            ? ""
            : " This month has more sessions than this page can count, so session counts are minimums."
        }`}
        isLast
      >
        <TableScrollFrame className="otari-spend-table">
          <DataTable
            ariaLabel="People"
            columns={columns}
            rows={people.isLoading ? [] : people.rows}
            isLoading={people.isLoading}
            getRowKey={(person) => person.id}
            onRowAction={(personId) =>
              void navigate({
                to: "/people/$personId",
                params: { personId },
              })
            }
            emptyContent={
              // Not null, which the phone layout reads as "No rows."
              people.error ? (
                ""
              ) : (
                <EmptyMessage>Nobody is here yet.</EmptyMessage>
              )
            }
            mobileLayout="cards"
          />
        </TableScrollFrame>
        <p className="text-caption">
          <Link
            to="/members"
            className="text-link underline max-sm:inline-flex max-sm:min-h-11 max-sm:items-center"
          >
            Workspace members
          </Link>{" "}
          adds people to {selected ? selected.name : "a workspace"} and sets
          their role in it.
          {canInvite ? (
            <>
              {" "}
              Everyone in the organization is on{" "}
              <Link
                to="/organization/members"
                className="text-link underline max-sm:inline-flex max-sm:min-h-11 max-sm:items-center"
              >
                Members &amp; roles
              </Link>
              .
            </>
          ) : null}
        </p>
      </Band>
      <InvitePeopleDialog isOpen={isInviteOpen} onOpenChange={setInviteOpen} />
      {changing ? (
        <ChangeRoleDialog
          key={changing.id}
          person={changing}
          onClose={() => setChanging(undefined)}
        />
      ) : null}
      {removing ? (
        <RemovePersonDialog
          person={removing}
          isOpen
          onOpenChange={(open) => {
            if (!open) setRemoving(undefined)
          }}
        />
      ) : null}
    </>
  )
}
