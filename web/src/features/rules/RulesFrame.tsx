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
import { useSurfaces } from "@/shared/hooks/useDeployment"

/** What the tab row names the view by: a type of rule, or Routing. */
export type RulesView = RuleTab | "routing"

/**
 * Rules' title and tab row, over either view: the list of rules
 * (`RulesPage.tsx`, one tab per type) or the routing policies
 * (`features/routing/RoutingPage.tsx`, at `/routing`). Two routes under one
 * tab row, as Sessions and Requests are. Routing is offered to every role, on
 * a deployment that hosts the routing surface.
 */
export function RulesFrame({
  current,
  error,
  children,
}: {
  current: RulesView
  error?: unknown
  children: ReactNode
}) {
  const isManager = useIsOrganizationManager()
  const hostsSurface = useSurfaces()
  const [isNewOpen, setNewOpen] = useState(false)
  const isRouting = current === "routing"

  return (
    <div className="flex min-w-0 flex-col gap-6">
      <div className="otari-rules-intro flex flex-col">
        <PageIntro
          title="Rules"
          action={
            // Routing's own Create policy is the action on that view.
            isManager && !isRouting ? (
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
          tabs={[
            ...ruleTabs(isManager).map((one) => ({
              id: one,
              to: "/rules",
              label: RULE_TAB_LABEL[one],
              search: one === "all" ? {} : { type: one },
            })),
            ...(hostsSurface("routing")
              ? [{ id: "routing", to: "/routing", label: "Routing" }]
              : []),
          ]}
        />
      </div>
      {children}
      {isNewOpen ? <NewRuleDialog onClose={() => setNewOpen(false)} /> : null}
    </div>
  )
}
