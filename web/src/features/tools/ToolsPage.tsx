import { useSearch } from "@tanstack/react-router"
import { ToolsFrame } from "@/features/tools/ToolsFrame"
import { ToolsGuardrailsPage } from "@/features/tools/ToolsGuardrailsPage"
import { ToolsOverview } from "@/features/tools/ToolsOverview"
import { useDeploymentOperator } from "@/shared/api/organizations"
import type { DashboardSearch } from "@/shared/helpers/search"

/**
 * `/tools`: every tool at a glance. The deployment's guardrail service
 * endpoints have no tab of their own, so the overview's operator row opens
 * them here, by `?service=guardrails`; anyone else gets the overview.
 */
export function ToolsPage() {
  const search = useSearch({ strict: false }) as DashboardSearch
  const { isOperator } = useDeploymentOperator()
  return (
    <ToolsFrame current="/tools">
      {search.service === "guardrails" && isOperator ? (
        <ToolsGuardrailsPage only="guardrails" />
      ) : (
        <ToolsOverview />
      )}
    </ToolsFrame>
  )
}
