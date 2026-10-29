import { FiEdit2, FiX } from "react-icons/fi"

import type { WorkspaceMcpServerGrant } from "@/client"
import { RowAction, RowActionRow } from "@/design-system/actions/RowAction"
import { DataTable, type DataTableColumn } from "@/design-system/data/DataTable"
import { Chip } from "@/design-system/indicators/Chip"
import { TableScrollFrame } from "@/design-system/layout/TableScrollFrame"
import { ruleAction, ruleTools } from "@/features/tools/mcpAccess"
import { grantSubject } from "@/features/tools/mcpGrantSubjects"

/** One row of a server's ACL: a rule for a key or a user, or the default policy. */
export type AclRow =
  | { kind: "rule"; grant: WorkspaceMcpServerGrant }
  | { kind: "default"; tools: readonly string[] }

/**
 * A server's ACL in the order it is matched: the rules for single keys, then
 * the rules for users, then the default policy, which is always last and can
 * be changed but not removed. Stacks into cards on a phone.
 */
export function McpGrantsTable({
  grants,
  defaultTools,
  labels,
  canManage,
  showRules = true,
  onEdit,
  onEditDefault,
  onRemove,
}: {
  grants: readonly WorkspaceMcpServerGrant[]
  /** The server's `everyone_tools`. */
  defaultTools: readonly string[]
  /** User labels by user id, from the organization roster. */
  labels: ReadonlyMap<string, string>
  canManage: boolean
  /** False when the viewer may not read the rules, so only the default shows. */
  showRules?: boolean
  onEdit: (grant: WorkspaceMcpServerGrant) => void
  onEditDefault: () => void
  onRemove: (grant: WorkspaceMcpServerGrant) => void
}) {
  const rules = showRules
    ? [
        ...grants.filter((grant) => grant.subject === "key"),
        ...grants.filter((grant) => grant.subject === "user"),
      ]
    : []
  const rows: AclRow[] = [
    ...rules.map((grant): AclRow => ({ kind: "rule", grant })),
    { kind: "default", tools: defaultTools },
  ]
  const toolsOf = (row: AclRow) =>
    row.kind === "rule" ? row.grant.tools : row.tools

  const columns: DataTableColumn<AclRow>[] = [
    {
      id: "source",
      header: "Source",
      isRowHeader: true,
      cell: (row) => {
        if (row.kind === "default") {
          return (
            <span className="flex min-w-0 flex-wrap items-center gap-x-2 gap-y-1">
              <span className="font-medium">Default policy</span>
              <span className="text-caption">any other key</span>
            </span>
          )
        }
        const { grant } = row
        const { name, detail } = grantSubject(grant, labels)
        return (
          <span className="flex min-w-0 flex-wrap items-center gap-x-2 gap-y-1">
            <span
              className={
                grant.subject === "key"
                  ? "font-mono text-sm break-all"
                  : "font-medium break-words"
              }
            >
              {name}
            </span>
            <span className="text-caption">{detail}</span>
            {grant.subject === "key" && grant.key_active === false ? (
              <Chip>inactive</Chip>
            ) : null}
          </span>
        )
      },
    },
    {
      id: "tools",
      header: "Tools",
      cell: (row) => (
        <span className="font-mono text-sm break-words">
          {ruleTools(toolsOf(row))}
        </span>
      ),
    },
    {
      id: "action",
      header: "Action",
      cell: (row) => {
        const action = ruleAction(toolsOf(row))
        return (
          <span
            className={`text-mono-caption ${action === "DENY" ? "text-danger" : "text-success"}`}
          >
            {action}
          </span>
        )
      },
    },
  ]
  if (canManage) {
    columns.push({
      id: "actions",
      header: "",
      cell: (row) => {
        if (row.kind === "default") {
          return (
            <RowActionRow>
              <RowAction
                icon={FiEdit2}
                label="Edit"
                ariaLabel="Edit the default policy"
                onPress={onEditDefault}
              />
            </RowActionRow>
          )
        }
        const { name } = grantSubject(row.grant, labels)
        return (
          <RowActionRow>
            <RowAction
              icon={FiEdit2}
              label="Edit"
              ariaLabel={`Edit the rule for ${name}`}
              onPress={() => onEdit(row.grant)}
            />
            <RowAction
              icon={FiX}
              label="Delete"
              ariaLabel={`Delete the rule for ${name}`}
              onPress={() => onRemove(row.grant)}
            />
          </RowActionRow>
        )
      },
    })
  }

  return (
    <TableScrollFrame className="otari-mcp-grants-table">
      <DataTable
        ariaLabel="ACL"
        columns={columns}
        rows={rows}
        getRowKey={(row) => (row.kind === "rule" ? row.grant.id : "default")}
        mobileLayout="cards"
      />
    </TableScrollFrame>
  )
}
