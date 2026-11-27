import { useState } from "react"
import type {
  ApiKey,
  OrganizationMember,
  OrganizationSpendCeiling,
} from "@/client"
import { DataTable, type DataTableColumn } from "@/design-system/data/DataTable"
import { EmptyMessage } from "@/design-system/feedback/EmptyMessage"
import { ErrorBanner } from "@/design-system/feedback/ErrorBanner"
import { SearchField } from "@/design-system/forms/SearchField"
import { TableScrollFrame } from "@/design-system/layout/TableScrollFrame"
import { Segmented } from "@/design-system/navigation/Segmented"
import { MemberAvatar } from "@/features/account/avatar/MemberAvatar"
import { keyState } from "@/features/keys/keyList"
import { memberLabel, membershipLabel } from "@/features/organization/roles"
import { EntryList } from "@/features/rules/BandParts"
import {
  modelEntries,
  type Rule,
  ruleIdForCeiling,
  toolEntries,
} from "@/features/rules/rules"
import type { Moderation } from "@/features/rules/useModeration"
import { keyName } from "@/features/spend/names"
import { useKeys } from "@/shared/api/apiKeys"
import { useAllSpendCeilings } from "@/shared/api/budgets"
import { useOrganizationMembers } from "@/shared/api/organizations"
import { useUrlState } from "@/shared/helpers/urlState"

const URL_DEFAULTS = { by: "members", q: "" }

const BY_OPTIONS = [
  { value: "members", label: "Members" },
  { value: "keys", label: "API keys" },
]

interface MemberRow {
  id: string
  member: OrganizationMember
  name: string
  budgets: string[]
}

interface KeyRow {
  id: string
  key: ApiKey
  name: string
  budgets: string[]
}

const includesText = (haystack: readonly string[], query: string) =>
  query === "" ||
  haystack.some((one) => one.toLowerCase().includes(query.toLowerCase()))

/**
 * Who in the selected workspace is under which guardrail, for an owner or
 * admin. Members show their own budgets and the content moderation their
 * requests go through; a member's model list is the deployment operator's to
 * read, so it is not a column. API keys show the same, plus each key's lists.
 */
export function GuardrailAssignments({
  rules,
  moderation,
}: {
  rules: readonly Rule[]
  moderation: Moderation
}) {
  const url = useUrlState(URL_DEFAULTS)
  const by = url.get("by") === "keys" ? "keys" : "members"
  const query = url.get("q")
  const [now] = useState(() => Date.now())
  const members = useOrganizationMembers()
  const ceilings = useAllSpendCeilings()
  const keys = useKeys()
  const workspaceId = moderation.workspace?.id

  const budgetNames = (matches: readonly OrganizationSpendCeiling[]) =>
    matches.map(
      (ceiling) =>
        rules.find((rule) => rule.id === ruleIdForCeiling(ceiling))?.name ??
        ceiling.name ??
        "Budget",
    )
  const allCeilings = ceilings.data ?? []

  const memberRows: MemberRow[] = (members.data ?? []).flatMap((member) => {
    if (member.status !== "active") return []
    const place = (member.workspaces ?? []).find(
      (one) => one.workspace_id === workspaceId,
    )
    if (!place) return []
    const name = memberLabel(member)
    if (!includesText([name, member.email ?? ""], query)) return []
    return [
      {
        id: member.organization_member_id ?? name,
        member,
        name,
        budgets: budgetNames(
          allCeilings.filter(
            (ceiling) =>
              (ceiling.scope_type === "org_member" &&
                ceiling.scope_id === member.organization_member_id) ||
              (ceiling.scope_type === "workspace_member" &&
                ceiling.scope_id === place.workspace_member_id),
          ),
        ),
      },
    ]
  })

  const keyRows: KeyRow[] = (keys.data ?? []).flatMap((key) => {
    if (key.workspace_id !== workspaceId || keyState(key, now) !== "active") {
      return []
    }
    const name = keyName(key)
    if (!includesText([name], query)) return []
    return [
      {
        id: key.id,
        key,
        name,
        budgets: budgetNames(
          allCeilings.filter(
            (ceiling) =>
              ceiling.scope_type === "api_token" && ceiling.scope_id === key.id,
          ),
        ),
      },
    ]
  })

  // Every row here runs the workspace's moderation: it is set per workspace.
  const moderationCell = moderation.isLoading ? null : moderation.rows
      .length === 0 ? (
    <span className="text-muted">None</span>
  ) : (
    <span className="flex min-w-0 items-center gap-2">
      <code className="truncate font-mono text-foreground">
        {moderation.rows.map((row) => row.profile).join(", ")}
      </code>
      <span className="shrink-0 text-subtle">
        via {moderation.workspace?.name}
      </span>
    </span>
  )
  const budgetCell = (names: readonly string[]) =>
    names.length === 0 ? (
      <span className="text-muted">None</span>
    ) : (
      <span className="block truncate" title={names.join(", ")}>
        {names.join(", ")}
      </span>
    )

  const memberColumns: DataTableColumn<MemberRow>[] = [
    {
      id: "member",
      header: "Member",
      isRowHeader: true,
      cell: (row) => (
        <span className="flex min-w-0 items-center gap-2.5">
          <MemberAvatar
            userId={row.member.user_id}
            face={row.member.avatar_face}
            isCompact
          />
          <span className="truncate">{row.name}</span>
        </span>
      ),
    },
    {
      id: "role",
      header: "Role",
      cell: (row) => membershipLabel(row.member.role),
    },
    { id: "budget", header: "Budget", cell: (row) => budgetCell(row.budgets) },
    {
      id: "moderation",
      header: "Content moderation",
      cell: () => moderationCell,
    },
  ]

  const keyColumns: DataTableColumn<KeyRow>[] = [
    {
      id: "key",
      header: "API key",
      isRowHeader: true,
      cell: (row) => <span className="block truncate">{row.name}</span>,
    },
    { id: "budget", header: "Budget", cell: (row) => budgetCell(row.budgets) },
    {
      id: "models",
      header: "Models",
      cell: (row) => (
        <EntryList entries={modelEntries(row.key.allowed_models)} />
      ),
    },
    {
      id: "tools",
      header: "Tools",
      cell: (row) => <EntryList entries={toolEntries(row.key.allowed_tools)} />,
    },
    {
      id: "moderation",
      header: "Content moderation",
      cell: () => moderationCell,
    },
  ]

  const error = members.error ?? ceilings.error ?? keys.error
  const isLoading =
    members.isLoading ||
    ceilings.isLoading ||
    keys.isLoading ||
    moderation.isLoading
  const noMatch = (
    <EmptyMessage>
      {query
        ? `Nothing matches “${query}”.`
        : by === "keys"
          ? "No API key you can read is in this workspace."
          : "Nobody is in this workspace yet."}
    </EmptyMessage>
  )

  return (
    <div className="flex flex-col gap-4">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <Segmented
          label="Assigned to"
          options={BY_OPTIONS}
          value={by}
          onChange={(next) => url.patch({ by: next, q: "" })}
        />
        <SearchField
          label={by === "keys" ? "Search API keys" : "Search people"}
          placeholder={by === "keys" ? "Search API keys" : "Search people"}
          value={query}
          onChange={(value) => url.patch({ q: value })}
          className="w-72 max-w-full max-sm:w-full"
        />
      </div>
      {error ? <ErrorBanner error={error} /> : null}
      {by === "keys" ? (
        <TableScrollFrame className="otari-rules-table is-assigned-keys">
          <DataTable
            ariaLabel="Guardrails on each API key"
            columns={keyColumns}
            rows={isLoading ? [] : keyRows}
            isLoading={isLoading}
            getRowKey={(row) => row.id}
            emptyContent={error ? "" : noMatch}
            mobileLayout="cards"
          />
        </TableScrollFrame>
      ) : (
        <TableScrollFrame className="otari-rules-table is-assigned-members">
          <DataTable
            ariaLabel="Guardrails on each member"
            columns={memberColumns}
            rows={isLoading ? [] : memberRows}
            isLoading={isLoading}
            getRowKey={(row) => row.id}
            emptyContent={error ? "" : noMatch}
            mobileLayout="cards"
          />
        </TableScrollFrame>
      )}
      {by === "members" &&
      allCeilings.some(
        (ceiling) =>
          ceiling.scope_type === "organization" ||
          (ceiling.scope_type === "workspace" &&
            ceiling.scope_id === workspaceId),
      ) ? (
        <p className="text-caption">
          Everyone also stays under the organization’s budget and their
          workspace’s own budget, which are shared. When several budgets apply,
          a request has to fit all of them.
        </p>
      ) : null}
    </div>
  )
}
