import { Link, useNavigate } from "@tanstack/react-router"
import { useState } from "react"
import { RowMenu } from "@/design-system/actions/RowMenu"
import { DataTable, type DataTableColumn } from "@/design-system/data/DataTable"
import { EmptyMessage } from "@/design-system/feedback/EmptyMessage"
import { TableScrollFrame } from "@/design-system/layout/TableScrollFrame"
import {
  useCanCreateKey,
  useCanManageWorkspace,
} from "@/features/organization/access"
import { LimitCell } from "@/features/rules/LimitCell"
import { ModeChip } from "@/features/rules/ModeChip"
import {
  type PendingAction,
  RuleActionDialog,
} from "@/features/rules/RuleActionDialog"
import { RuleName } from "@/features/rules/RuleName"
import { RulesFrame } from "@/features/rules/RulesFrame"
import { ruleMenuItems } from "@/features/rules/ruleMenu"
import {
  defaultRuleSort,
  emptyCopy,
  RULE_TYPE_LABEL,
  type Rule,
  type RuleColumn,
  type RuleSort,
  ruleActions,
  ruleColumns,
  rulesOnTab,
  ruleTabOf,
  sortRules,
} from "@/features/rules/rules"
import { ScopeCell } from "@/features/rules/ScopeCell"
import { UsedCell } from "@/features/rules/UsedCell"
import { useRules } from "@/features/rules/useRules"
import { useKeysScope } from "@/shared/api/apiKeys"
import { useOrganizationContext } from "@/shared/api/organizations"
import { formatUsd } from "@/shared/helpers/format"
import { useUrlState } from "@/shared/helpers/urlState"
import { useSurfaces } from "@/shared/hooks/useDeployment"

const URL_DEFAULTS = { type: "" }

const SORTABLE: readonly RuleSort["column"][] = [
  "name",
  "limit",
  "spent",
  "used",
]

const listWords = (list: string[] | null | undefined) =>
  list == null ? "Any" : list.join(", ")

/**
 * Every rule an agent's requests are held to, in one list, whichever store
 * keeps it today. A row opens the rule. Owners and admins see spend against
 * each limit and change it from the row; anyone else reads the limits the
 * server gives them, with one line naming the ones it does not.
 */
export function RulesPage() {
  const navigate = useNavigate()
  const { rules, isManager, isLoading, error } = useRules()
  const canManageWorkspace = useCanManageWorkspace()
  const canCreateKey = useCanCreateKey()
  const keysScope = useKeysScope()
  const context = useOrganizationContext().data
  const url = useUrlState(URL_DEFAULTS)
  const tab = ruleTabOf(url.get("type"), isManager)
  const [sort, setSort] = useState<RuleSort>()
  const [pending, setPending] = useState<PendingAction>()
  const hostsSurface = useSurfaces()
  // The workspace's guardrails page, for the owners and admins who set them.
  const workspaceGuardrails =
    hostsSurface("tools") && isManager ? (
      <Link
        to="/tools/guardrails"
        className="text-link underline max-sm:inline-flex max-sm:min-h-11 max-sm:items-center"
      >
        Workspace guardrails
      </Link>
    ) : undefined

  const activeSort = sort ?? defaultRuleSort(isManager)
  const shown = sortRules(rulesOnTab(rules, tab), activeSort)
  const actionsOf = (rule: Rule) =>
    ruleActions(rule, { isManager, canManageWorkspace })
  const hasActions = isManager || rules.some((rule) => actionsOf(rule).length)
  const empty = emptyCopy(tab, isManager, {
    isDeploymentWide: keysScope.isDeploymentWide,
    canCreateKey,
    organizationName: context?.organization?.name,
  })
  const isFailed = Boolean(error) && rules.length === 0

  const cells: Record<RuleColumn, DataTableColumn<Rule>> = {
    name: {
      id: "name",
      header: "Rule",
      isRowHeader: true,
      allowsSorting: true,
      cell: (rule) => <RuleName name={rule.name} />,
    },
    type: {
      id: "type",
      header: "Type",
      cell: (rule) => (
        <span className="text-sm text-muted">{RULE_TYPE_LABEL[rule.type]}</span>
      ),
    },
    scope: {
      id: "scope",
      header: "Applies to",
      cell: (rule) => <ScopeCell scope={rule.scope} />,
    },
    limit: {
      id: "limit",
      header: "Limit",
      allowsSorting: true,
      cell: (rule) => <LimitCell rule={rule} />,
    },
    spent: {
      id: "spent",
      header: "Spent",
      align: "end",
      allowsSorting: true,
      cell: (rule) =>
        rule.spend?.spent === undefined ? null : (
          <span className="text-mono-caption text-foreground tabular-nums">
            {formatUsd(rule.spend.spent)}
          </span>
        ),
    },
    used: {
      id: "used",
      header: "Used",
      align: "end",
      allowsSorting: true,
      cell: (rule) =>
        rule.spend?.usedSpent === undefined ? null : (
          <UsedCell
            spent={rule.spend.usedSpent}
            max={rule.spend.usedMax ?? rule.spend.max}
            label={`${rule.name} used`}
          />
        ),
    },
    models: {
      id: "models",
      header: "Models",
      cell: (rule) => (
        <span
          className="block truncate text-sm"
          title={listWords(rule.access?.models)}
        >
          {listWords(rule.access?.models)}
        </span>
      ),
    },
    tools: {
      id: "tools",
      header: "Tools",
      cell: (rule) => (
        <span
          className="block truncate text-sm"
          title={listWords(rule.access?.tools)}
        >
          {listWords(rule.access?.tools)}
        </span>
      ),
    },
    mode: {
      id: "mode",
      header: "Mode",
      cell: (rule) => <ModeChip mode={rule.mode} />,
    },
    actions: {
      id: "actions",
      header: <span className="sr-only">Actions</span>,
      align: "end",
      cell: (rule) => (
        <RowMenu
          label={`Actions for ${rule.name}`}
          actions={ruleMenuItems(rule, actionsOf(rule), (action) =>
            setPending({ action, rule }),
          )}
        />
      ),
    },
  }
  const columns = ruleColumns(tab, isManager, hasActions).map((id) => cells[id])

  return (
    <RulesFrame current={tab} error={error}>
      {isFailed ? null : (
        <TableScrollFrame
          className={`otari-rules-table is-${tab}${isManager ? "" : " is-member"}`}
        >
          <DataTable
            ariaLabel="Rules"
            columns={columns}
            // A row built before the budgets land carries a stand-in name,
            // and before the keys land, no key name.
            rows={isLoading ? [] : shown}
            isLoading={isLoading}
            getRowKey={(rule) => rule.id}
            sortDescriptor={activeSort}
            onSortChange={(next) => {
              const column = SORTABLE.find((one) => one === next.column)
              if (column) setSort({ column, direction: next.direction })
            }}
            onRowAction={(ruleId) =>
              void navigate({ to: "/rules/$ruleId", params: { ruleId } })
            }
            emptyContent={
              // Nothing is empty that could not be read. Not null, which the
              // phone layout reads as "No rows."
              error ? (
                ""
              ) : (
                <EmptyMessage>
                  <span className="flex flex-col gap-1">
                    <span className="text-body">{empty.title}</span>
                    {empty.body ? (
                      <span>
                        {empty.body}
                        {empty.link ? (
                          <>
                            {" "}
                            <Link
                              to={empty.link.to}
                              className="text-link underline"
                            >
                              {empty.link.label}
                            </Link>
                            .
                          </>
                        ) : null}
                      </span>
                    ) : null}
                  </span>
                </EmptyMessage>
              )
            }
            mobileLayout="cards"
          />
        </TableScrollFrame>
      )}
      <div className="flex flex-col gap-1 text-caption">
        {isManager ? null : (
          <p>
            Owners and admins also see organization and workspace spend limits,
            limits on one person or one API key, and guardrails.
            {/* No Data tab here (guardrails are refused this caller), and
                the workspace Guardrails page opens to owners and admins
                only, so this line links it only if that changes. */}
            {workspaceGuardrails ? <> {workspaceGuardrails}</> : null}
          </p>
        )}
        {isManager && tab === "data" && workspaceGuardrails ? (
          <p>{workspaceGuardrails}</p>
        ) : null}
        {tab === "all" ? (
          <p>
            Rules for commands and files live in each repository’s .otari
            directory and run on the machine, so they are not listed here.
          </p>
        ) : null}
      </div>
      {pending ? (
        <RuleActionDialog
          pending={pending}
          onClose={() => setPending(undefined)}
        />
      ) : null}
    </RulesFrame>
  )
}
