import type { ReactNode } from "react"
import { useState } from "react"
import { FiPlus } from "react-icons/fi"
import { Button } from "@/design-system/actions/Button"
import { ErrorBanner } from "@/design-system/feedback/ErrorBanner"
import { PageIntro } from "@/design-system/layout/PageIntro"
import { useIsOrganizationManager } from "@/features/organization/access"
import { NewRuleDialog } from "@/features/rules/NewRuleDialog"
import { RULE_TAB_LABEL, type RuleTab, ruleTabs } from "@/features/rules/rules"
import { ViewTabs } from "@/shared/components/ViewTabs"

/** Rules' title and tab row, over the list of rules, one tab per type. */
export function RulesFrame({
  current,
  error,
  children,
}: {
  current: RuleTab
  error?: unknown
  children: ReactNode
}) {
  const isManager = useIsOrganizationManager()
  const [isNewOpen, setNewOpen] = useState(false)

  return (
    <div className="flex min-w-0 flex-col gap-6">
      <div className="otari-rules-intro flex flex-col">
        <PageIntro
          title="Rules"
          action={
            isManager ? (
              <Button variant="primary" onPress={() => setNewOpen(true)}>
                <FiPlus aria-hidden className="size-4" />
                New rule
              </Button>
            ) : undefined
          }
        />
        {error ? (
          <div className="pb-4">
            <ErrorBanner error={error} />
          </div>
        ) : null}
        <ViewTabs
          label="Rule types"
          current={current}
          tabs={ruleTabs(isManager).map((one) => ({
            id: one,
            to: "/rules",
            label: RULE_TAB_LABEL[one],
            search: one === "all" ? {} : { type: one },
          }))}
        />
      </div>
      {children}
      {isNewOpen ? <NewRuleDialog onClose={() => setNewOpen(false)} /> : null}
    </div>
  )
}
