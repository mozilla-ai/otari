import { useNavigate } from "@tanstack/react-router"
import { useState } from "react"
import { FiPlus } from "react-icons/fi"
import { Button } from "@/design-system/actions/Button"
import { RowMenu } from "@/design-system/actions/RowMenu"
import { DataTable, type DataTableColumn } from "@/design-system/data/DataTable"
import { EmptyState } from "@/design-system/feedback/EmptyState"
import { ErrorBanner } from "@/design-system/feedback/ErrorBanner"
import { PageIntro } from "@/design-system/layout/PageIntro"
import { TableScrollFrame } from "@/design-system/layout/TableScrollFrame"
import {
  useCanCreateKey,
  useCanManageWorkspace,
} from "@/features/organization/access"
import { BandEmpty, EntryList } from "@/features/rules/BandParts"
import { GuardrailAssignments } from "@/features/rules/GuardrailAssignments"
import { LimitCell } from "@/features/rules/LimitCell"
import { ModerationBand } from "@/features/rules/ModerationBand"
import { NewRuleDialog } from "@/features/rules/NewRuleDialog"
import {
  type PendingAction,
  RuleActionDialog,
} from "@/features/rules/RuleActionDialog"
import { RuleName } from "@/features/rules/RuleName"
import { ruleMenuItems } from "@/features/rules/ruleMenu"
import {
  defaultRuleSort,
  emptyCopy,
  modelEntries,
  RULE_TYPE_LABEL,
  type Rule,
  type RuleColumn,
  type RuleSort,
  type RuleType,
  ruleActions,
  ruleColumns,
  rulesOfType,
  sortRules,
  toolEntries,
} from "@/features/rules/rules"
import { ScopeCell } from "@/features/rules/ScopeCell"
import { UsedCell } from "@/features/rules/UsedCell"
import { useModeration } from "@/features/rules/useModeration"
import { useRules } from "@/features/rules/useRules"
import { useKeysScope } from "@/shared/api/apiKeys"
import { useOrganizationContext } from "@/shared/api/organizations"
import { Band } from "@/shared/components/Band"
import { ViewTabs } from "@/shared/components/ViewTabs"
import { formatUsd } from "@/shared/helpers/format"
import { useUrlState } from "@/shared/helpers/urlState"

const URL_DEFAULTS = { view: "" }

const SORTABLE: readonly RuleSort["column"][] = [
  "name",
  "limit",
  "spent",
  "used",
]

/**
 * Every guardrail an agent's requests are held to, one band per type, each
 * with the columns its type has. Owners and admins see spend against each
 * budget and change it from the row, and the content moderation running in
 * the selected workspace; their Assignments view lists who is under which.
 * Anyone else reads the budgets and key lists the server gives them, with one
 * line naming what it does not.
 */
export function GuardrailsPage() {
  const { rules, isManager, isLoading, error } = useRules()
  const moderation = useModeration(isManager)
  const context = useOrganizationContext().data
  const url = useUrlState(URL_DEFAULTS)
  const [isNewOpen, setNewOpen] = useState(false)
  const isAssignments = isManager && url.get("view") === "assignments"
  const organizationName = context?.organization?.name

  const budgets = rulesOfType(rules, "spend")
  const access = rulesOfType(rules, "access")
  const isEmpty =
    isManager &&
    !isLoading &&
    !moderation.isLoading &&
    !error &&
    !moderation.error &&
    budgets.length === 0 &&
    access.length === 0 &&
    moderation.rows.length === 0
  const isFailed = Boolean(error) && rules.length === 0

  return (
    <div className="flex min-w-0 flex-col gap-6">
      <div className="otari-rules-intro flex flex-col">
        <PageIntro
          title="Guardrails"
          action={
            isManager ? (
              <Button variant="primary" onPress={() => setNewOpen(true)}>
                <FiPlus aria-hidden className="size-4" />
                New guardrail
              </Button>
            ) : undefined
          }
        />
        {/* One banner: a read both views share fails once, and Assignments
            carries it with its own reads. */}
        {!isAssignments && (error || moderation.error) ? (
          <div className="pb-4">
            <ErrorBanner error={error ?? moderation.error} />
          </div>
        ) : null}
        {isManager ? (
          <ViewTabs
            label="Guardrails views"
            current={isAssignments ? "assignments" : "guardrails"}
            tabs={[
              {
                id: "guardrails",
                to: "/guardrails",
                label: "Guardrails",
                search: {},
              },
              {
                id: "assignments",
                to: "/guardrails",
                label: "Assignments",
                search: { view: "assignments" },
              },
            ]}
          />
        ) : null}
      </div>

      {isAssignments ? (
        <GuardrailAssignments
          rules={rules}
          isRulesLoading={isLoading}
          pageError={error ?? moderation.error}
          moderation={moderation}
        />
      ) : isEmpty ? (
        <EmptyState
          title="No guardrails yet"
          description={`Guardrails set budgets, model and provider access, and content moderation for the people and API keys in ${organizationName ?? "your organization"}.`}
          actionLabel="New guardrail"
          onAction={() => setNewOpen(true)}
        />
      ) : isFailed ? null : (
        <div className="flex flex-col">
          <RuleBand
            type="spend"
            rules={budgets}
            isManager={isManager}
            isLoading={isLoading}
            hasError={Boolean(error)}
          />
          <RuleBand
            type="access"
            rules={access}
            isManager={isManager}
            isLoading={isLoading}
            hasError={Boolean(error)}
            isLast={!isManager}
          />
          {isManager ? <ModerationBand moderation={moderation} /> : null}
        </div>
      )}

      {isManager ? null : (
        <p className="text-caption">
          Owners and admins also see organization and workspace budgets, budgets
          on one person or one API key, and content moderation.
        </p>
      )}
      {isNewOpen ? <NewRuleDialog onClose={() => setNewOpen(false)} /> : null}
    </div>
  )
}

/** One type's band: its own columns, a row opening the guardrail. */
function RuleBand({
  type,
  rules,
  isManager,
  isLoading,
  hasError,
  isLast = false,
}: {
  type: Exclude<RuleType, "data">
  rules: readonly Rule[]
  isManager: boolean
  isLoading: boolean
  hasError: boolean
  isLast?: boolean
}) {
  const navigate = useNavigate()
  const canManageWorkspace = useCanManageWorkspace()
  const canCreateKey = useCanCreateKey()
  const keysScope = useKeysScope()
  const context = useOrganizationContext().data
  const [sort, setSort] = useState<RuleSort>()
  const [pending, setPending] = useState<PendingAction>()

  const activeSort = sort ?? defaultRuleSort(isManager)
  const shown = sortRules(rules, activeSort)
  const actionsOf = (rule: Rule) =>
    ruleActions(rule, { isManager, canManageWorkspace })
  const hasActions = rules.some((rule) => actionsOf(rule).length)
  const empty = emptyCopy(type, isManager, {
    isDeploymentWide: keysScope.isDeploymentWide,
    canCreateKey,
    organizationName: context?.organization?.name,
  })

  const cells: Record<RuleColumn, DataTableColumn<Rule>> = {
    name: {
      id: "name",
      header: type === "access" ? "API key" : "Name",
      isRowHeader: true,
      allowsSorting: type === "spend",
      cell: (rule) => (
        <RuleName
          name={
            type === "access" && rule.scope.kind === "key" && rule.scope.name
              ? rule.scope.name
              : rule.name
          }
        />
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
        rule.spend?.usedSpent === undefined ||
        rule.spend.max === null ? null : (
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
        <EntryList entries={modelEntries(rule.access?.models ?? null)} />
      ),
    },
    tools: {
      id: "tools",
      header: "Tools",
      cell: (rule) => (
        <EntryList entries={toolEntries(rule.access?.tools ?? null)} />
      ),
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
  const columns = ruleColumns(type, isManager, hasActions).map(
    (id) => cells[id],
  )

  return (
    <Band
      title={RULE_TYPE_LABEL[type]}
      count={isLoading ? undefined : rules.length}
      isLast={isLast}
    >
      <TableScrollFrame
        className={`otari-rules-table is-${type}${isManager ? "" : " is-member"}`}
      >
        <DataTable
          ariaLabel={RULE_TYPE_LABEL[type]}
          columns={columns}
          // A row built before the budgets land carries a stand-in name, and
          // before the keys land, no key name.
          rows={isLoading ? [] : shown}
          isLoading={isLoading}
          getRowKey={(rule) => rule.id}
          sortDescriptor={type === "spend" ? activeSort : undefined}
          onSortChange={(next) => {
            const column = SORTABLE.find((one) => one === next.column)
            if (column) setSort({ column, direction: next.direction })
          }}
          onRowAction={(id) =>
            void navigate({ to: "/guardrails/$id", params: { id } })
          }
          emptyContent={
            // Nothing is empty that could not be read. Not null, which the
            // phone layout reads as "No rows."
            hasError ? "" : <BandEmpty copy={empty} />
          }
          mobileLayout="cards"
        />
      </TableScrollFrame>
      {pending ? (
        <RuleActionDialog
          pending={pending}
          onClose={() => setPending(undefined)}
        />
      ) : null}
    </Band>
  )
}
