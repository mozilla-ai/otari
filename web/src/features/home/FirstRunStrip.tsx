import { Button } from "@heroui/react"
import { useNavigate } from "@tanstack/react-router"
import { useMemo } from "react"
import { Dot } from "@/design-system/indicators/Dot"
import { Section } from "@/design-system/layout/Section"
import { canManage } from "@/features/organization/roles"
import { useHostedProviders } from "@/shared/api/hostedProviders"
import { useModels } from "@/shared/api/models"
import {
  useDeploymentOperator,
  useOrganizationContext,
} from "@/shared/api/organizations"
import { useProviders } from "@/shared/api/providers"
import { useUsageLogs } from "@/shared/api/usage"
import { useSelectedWorkspace } from "@/shared/hooks/SelectedWorkspace"

/**
 * Whether anything has been served in the selected workspace: imported OTLP
 * usage lands in the usage tables through a budget-exempt key and no provider
 * config, so "nothing to route to" alone does not mean "nothing has
 * happened". Asked only while that is in doubt.
 */
function useHasAnyUsage(enabled: boolean) {
  const { selected: workspace } = useSelectedWorkspace()
  const filters = useMemo(
    () => ({ workspace_id: workspace?.workspace_id }),
    [workspace],
  )
  const logs = useUsageLogs(filters, 0, 1, enabled)
  return {
    isKnown: logs.isSuccess,
    hasUsage: (logs.data?.length ?? 0) > 0,
  }
}

/**
 * Home's first step on a gateway with nothing to route to and nothing served:
 * add a provider. The first-request guide (`SetupGuide`) holds back in exactly
 * that state, so this is the one thing telling a fresh install what to do.
 *
 * The operator is counted on the deployment's providers, configured or
 * hosted, and offered the button that adds one; anyone else on the models
 * their workspace may call, with an owner or admin sent to the organization's
 * provider keys and anyone else told whom to ask. Each read waits for its
 * answer, so the band never flashes in on a gateway that is set up.
 */
export function FirstRunStrip() {
  const { answer } = useDeploymentOperator()
  return answer === "not-operator" ? (
    <OrganizationFirstRun />
  ) : answer === "pending" ? null : (
    <OperatorFirstRun />
  )
}

function OperatorFirstRun() {
  const providers = useProviders()
  const hosted = useHostedProviders()
  // A failed providers query leaves the setup state unknown, so it stays
  // quiet. A failed hosted read counts none rather than blocking the band.
  const providerCount = providers.isSuccess
    ? providers.data.providers.length + (hosted.data?.data?.length ?? 0)
    : 0
  const needsSetup = providers.isSuccess && providerCount === 0
  const usage = useHasAnyUsage(needsSetup)
  return needsSetup && usage.isKnown && !usage.hasUsage ? (
    <GetStartedStrip />
  ) : null
}

function OrganizationFirstRun() {
  const managesSpend = canManage(useOrganizationContext().data)
  const { selected } = useSelectedWorkspace()
  const scope = selected?.workspace_id
  // What this caller could route a request to. Not asked until a workspace is
  // selected, since the band is about one.
  const models = useModels(scope !== undefined, scope)
  const hasNoModels = models.isSuccess && models.data.data.length === 0
  const usage = useHasAnyUsage(hasNoModels)
  return hasNoModels && usage.isKnown && !usage.hasUsage ? (
    <GetStartedStrip
      description={
        managesSpend
          ? "Add a provider key to start serving models, then create an API key and send your first request."
          : "Ask an owner or admin of this organization to add a provider key. Once models are available, you can create an API key and send a request."
      }
      action={managesSpend ? "organization-provider-key" : null}
    />
  ) : null
}

function GetStartedStrip({
  description = "Add a provider to begin serving models. Once it is configured, this page will show your gateway’s traffic, spend, and health.",
  action = "provider",
}: {
  description?: string
  /** The button, or none for a caller who may not act on it. */
  action?: "provider" | "organization-provider-key" | null
}) {
  const navigate = useNavigate()

  return (
    <Section
      className="border-y border-border py-5"
      contentClassName="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between"
    >
      <div className="flex items-start gap-3">
        <Dot className="mt-2.5 bg-accent" />
        <div>
          <h2 className="text-display-sub">Get started with Otari</h2>
          <p className="mt-1 text-sm text-muted">{description}</p>
        </div>
      </div>
      {action === null ? null : (
        <Button
          variant="primary"
          className="shrink-0"
          onPress={() =>
            navigate({
              to:
                // Only an operator is offered "provider" (`FirstRunStrip`), so
                // it goes to the deployment's own providers.
                action === "provider"
                  ? "/admin/hosted-providers"
                  : "/organization/provider-keys",
            })
          }
        >
          {action === "provider"
            ? "Add your first provider"
            : "Add a provider key"}
        </Button>
      )}
    </Section>
  )
}
