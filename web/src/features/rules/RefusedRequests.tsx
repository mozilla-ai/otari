import type { UsageEntry } from "@/client"
import { DataTable, type DataTableColumn } from "@/design-system/data/DataTable"
import { EmptyMessage } from "@/design-system/feedback/EmptyMessage"
import { ErrorBanner } from "@/design-system/feedback/ErrorBanner"
import { TableScrollFrame } from "@/design-system/layout/TableScrollFrame"
import {
  useIsOrganizationManager,
  useIsRoleSettled,
} from "@/features/organization/access"
import {
  useCurrentOwner,
  useMemberAttributionLabels,
} from "@/features/organization/attribution"
import type { Rule, RuleSubject } from "@/features/rules/rules"
import { REFUSALS_SHOWN, useRefusedRequests } from "@/features/rules/useRules"
import { useWeek } from "@/features/spend/week"
import { Band } from "@/shared/components/Band"
import { formatDateTimeMinutes } from "@/shared/helpers/format"

/**
 * The rule's subject narrowed to the caller, or nobody when the rule does not
 * reach them or their billing id is not known.
 */
function ownSubject(subject: RuleSubject, ownerId: string): RuleSubject {
  if (!ownerId || (subject.userIds && !subject.userIds.includes(ownerId))) {
    return {}
  }
  return {
    ...(subject.apiKeyIds ? { apiKeyIds: subject.apiKeyIds } : {}),
    ...(subject.workspaceId ? { workspaceId: subject.workspaceId } : {}),
    userIds: [ownerId],
  }
}

/**
 * The decision log, as far as the usage log can give one: every request the
 * gateway refused this week for whoever the rule reaches. A refusal names its
 * reason but not the rule, so a person under two limits sees both limits'
 * refusals on each. Anyone but an owner or admin sees only their own: a
 * refusal's reason can say how much of a limit someone has left.
 */
export function RefusedRequests({ rule }: { rule: Rule }) {
  const week = useWeek()
  const isSettled = useIsRoleSettled()
  const isManager = useIsOrganizationManager()
  const owner = useCurrentOwner(isSettled && !isManager)
  const subject = isManager
    ? rule.subject
    : ownSubject(rule.subject, owner.ownerId)
  // A rule that reaches no one is not read, so its table claims nothing.
  const isRead =
    isSettled &&
    Boolean(
      subject.everyone ||
        subject.userIds?.length ||
        subject.apiKeyIds?.length ||
        subject.workspaceId,
    )
  const refused = useRefusedRequests(subject, week.start, isRead)
  const read = isRead
    ? (refused.data ?? []).filter(
        (row) => isManager || row.user_id === owner.ownerId,
      )
    : []
  const people = useMemberAttributionLabels()
  const columns: DataTableColumn<UsageEntry>[] = [
    {
      id: "when",
      header: "When",
      cell: (row) => (
        <span className="whitespace-nowrap tabular-nums">
          {formatDateTimeMinutes(row.timestamp)}
        </span>
      ),
    },
    {
      id: "who",
      header: "Person",
      // As Home's People cut names someone the roster does not.
      cell: (row) =>
        row.user_id === null
          ? "No person"
          : (people.get(row.user_id) ??
            (row.user_alias || `${row.user_id.slice(0, 8)}…`)),
    },
    {
      id: "key",
      header: "API key",
      cell: (row) => row.api_key_name ?? "",
    },
    {
      id: "why",
      header: "Why",
      isRowHeader: true,
      cell: (row) => (
        <span
          className="block max-w-[32rem] truncate text-caption"
          title={row.error_message ?? "Refused"}
        >
          {row.error_message ?? "Refused"}
        </span>
      ),
    },
  ]
  return (
    <Band
      title="Refused requests"
      description={
        isManager
          ? "Requests the gateway refused this week for anyone under this rule. A refusal says why, not which rule refused it."
          : "Your own requests the gateway refused this week under this rule. Only owners and admins see everyone’s. A refusal says why, not which rule refused it."
      }
    >
      <ErrorBanner error={refused.error} />
      <TableScrollFrame className="">
        <DataTable
          ariaLabel={`Refusals for ${rule.name}`}
          columns={columns}
          rows={read.slice(0, REFUSALS_SHOWN)}
          isLoading={(isRead && refused.isLoading) || owner.isPending}
          getRowKey={(row) => row.id}
          emptyContent={
            refused.error || !isRead ? (
              ""
            ) : (
              <EmptyMessage>Nothing refused this week.</EmptyMessage>
            )
          }
          mobileLayout="cards"
        />
      </TableScrollFrame>
      {read.length > REFUSALS_SHOWN ? (
        <p className="text-caption">
          Showing the {REFUSALS_SHOWN} most recent refused requests.
        </p>
      ) : null}
    </Band>
  )
}
