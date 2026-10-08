import type { ReactNode } from "react"
import type {
  UpdateWorkspaceCodeExecutionPolicyRequest,
  WorkspaceCodeExecutionPolicy,
} from "@/client"
import { Button } from "@/design-system/actions/Button"
import { ErrorBanner } from "@/design-system/feedback/ErrorBanner"
import { Toggle } from "@/design-system/forms/Toggle"
import { SettingRow } from "@/design-system/layout/SettingRow"
import { SettingsGroup } from "@/design-system/layout/SettingsGroup"
import { canManageWorkspace } from "@/features/organization/roles"
import { usePolicyWriter } from "@/features/tools/usePolicyWriter"
import { WorkspaceToolStatus } from "@/features/tools/WorkspaceToolStatus"
import { useOrganizationContext } from "@/shared/api/organizations"
import {
  useClearWorkspaceCodeExecutionPolicy,
  useSetWorkspaceCodeExecutionPolicy,
  useWorkspaceCodeExecutionPolicy,
} from "@/shared/api/tools"
import { useSelectedWorkspace } from "@/shared/hooks/SelectedWorkspace"
import { useAutosave } from "@/shared/hooks/useAutosave"

// The stored policy has three states (allowed, blocked, none) and can also
// narrow limits, the image and the tool kinds. The card shows only the switch;
// the narrowing fields are set through the API.
//
// What "none" means depends on the deployment. Standalone, no policy narrows
// nothing, so it reads as on, and switching on deletes a policy that narrows
// nothing else. Hosted, the control plane's resolver treats a workspace with no
// policy as off (code execution is something a workspace turns on), so the
// switch reads the stored `enabled` alone and only ever writes it.

/** Whether a stored policy narrows anything beyond allowing or blocking. */
function narrowsAnything(policy: WorkspaceCodeExecutionPolicy): boolean {
  return [
    policy.default_purpose_hint,
    policy.max_iterations,
    policy.exec_timeout_s,
    policy.image,
    policy.tools,
    policy.executor,
  ].some((value) => value != null)
}

/**
 * Which parts of a stored policy name an image or tool kind this deployment no
 * longer offers. Admission refuses such a policy, so the switch alone would read
 * "on" over a workspace whose requests all fail.
 */
function staleParts(policy: WorkspaceCodeExecutionPolicy, isHosted: boolean) {
  return {
    // Hosted, `allowed_images` is the control plane's own list, not the
    // platform sandbox's, so it cannot say a pin was withdrawn.
    image:
      !isHosted &&
      policy.image != null &&
      !policy.allowed_images.includes(policy.image),
    tools: (policy.tools ?? []).some(
      (name) => !policy.available_tools.includes(name),
    ),
  }
}

/**
 * Whether requests billed to this workspace may run generated code.
 *
 * A policy can only narrow what the deployment above allows; it never grants a
 * sandbox the deployment has not configured.
 */
export function WorkspaceCodeExecutionPolicyCard({
  leading,
  isHosted = false,
}: {
  /** Rows above the switch, so the tool and its switch read as one card. */
  leading?: ReactNode
  /**
   * A hosted control plane, where no policy means off and the sandbox is the
   * platform's rather than this process's, so `sandbox_configured` says nothing.
   */
  isHosted?: boolean
}) {
  const { selected, isLoading: workspaceLoading } = useSelectedWorkspace()
  const context = useOrganizationContext()
  // The client half of the gate the service enforces, and it gates the *read*
  // too: the policy is the workspace's posture rather than one member's
  // allowance, so a member who cannot manage the workspace cannot see it
  // either, and asking would earn a 403 banner over a form they cannot use.
  const manages = canManageWorkspace(context.data, selected?.role)
  const workspaceId = selected && manages ? selected.workspace_id : null
  const query = useWorkspaceCodeExecutionPolicy(workspaceId)
  const setPolicy = useSetWorkspaceCodeExecutionPolicy()
  const clearPolicy = useClearWorkspaceCodeExecutionPolicy()
  const save = useAutosave()
  // A PUT replaces the whole policy, so the switch writes through the same
  // serialized writer the narrowing fields would, carrying them unchanged.
  const write = usePolicyWriter({
    server: query.data,
    resetKey: selected?.workspace_id ?? "",
    toBody: (stored) => ({
      enabled: stored.enabled,
      default_purpose_hint: stored.default_purpose_hint,
      max_iterations: stored.max_iterations,
      exec_timeout_s: stored.exec_timeout_s,
      image: stored.image,
      tools: stored.tools,
      executor: stored.executor,
    }),
    put: (body: UpdateWorkspaceCodeExecutionPolicyRequest) =>
      setPolicy.mutateAsync({
        workspaceId: selected?.workspace_id as string,
        body,
      }),
  })

  if (!selected) {
    return (
      <SettingsGroup isBounded>
        {leading}
        <SettingRow
          label="Workspace access"
          help={
            workspaceLoading
              ? "Reading the workspaces you belong to."
              : "Per-workspace code execution is set on a workspace you belong to. An owner or admin can add you to one on the Workspaces page."
          }
          control={null}
        />
      </SettingsGroup>
    )
  }

  if (!manages) {
    return (
      <WorkspaceToolStatus
        tool="code_execution"
        leading={leading}
        workspace={selected}
      />
    )
  }

  const policy = query.data
  const isOn = isHosted
    ? Boolean(policy?.configured && policy.enabled)
    : !(policy?.configured && !policy.enabled)
  const hasSandbox = isHosted || (policy?.sandbox_configured ?? true)
  const organizationName = context.data?.organization.name
  const stale = policy
    ? staleParts(policy, isHosted)
    : { image: false, tools: false }
  const needsReset = isOn && (stale.image || stale.tools)
  // Disabled until the read has succeeded. Without that the switch sits on over
  // a workspace that may well have a stored policy, and one change issues the
  // write that drops it.
  const isUnreadable = query.isLoading || query.isError || !policy

  const setAllowed = (allowed: boolean) =>
    save.run(() =>
      isHosted || !allowed || (policy && narrowsAnything(policy))
        ? write({ enabled: allowed })
        : clearPolicy.mutateAsync({ workspaceId: selected.workspace_id }),
    )
  // Drops only what is stale, so a valid limit beside it survives.
  const clearStale = () =>
    write({
      enabled: true,
      ...(stale.image ? { image: null } : {}),
      ...(stale.tools ? { tools: null } : {}),
    })

  return (
    <SettingsGroup isBounded>
      {leading}
      {query.error ? (
        <div className="px-4 py-3">
          <ErrorBanner error={query.error} />
        </div>
      ) : null}

      <SettingRow
        label={`Allow in ${selected.name}`}
        help={
          organizationName
            ? `Workspace in ${organizationName}.`
            : "This workspace's requests."
        }
        error={save.error}
        note={
          !hasSandbox ? (
            <p className="text-caption text-subtle">
              This deployment has no sandbox configured, so code execution is
              off everywhere.
            </p>
          ) : needsReset ? (
            <div className="flex flex-col items-start gap-2">
              <p className="text-caption text-warning">
                This workspace's policy names an image or tool this deployment
                no longer offers, so its requests are refused.
              </p>
              <Button
                size="sm"
                isPending={save.isSaving}
                isDisabled={isUnreadable}
                onPress={() => {
                  // Never queued behind the switch's own write, whose
                  // `enabled` this one would otherwise overwrite.
                  if (save.isSaving || isUnreadable) return
                  void save.run(clearStale)
                }}
              >
                Clear it
              </Button>
            </div>
          ) : null
        }
        control={
          <Toggle
            label="Allow code execution"
            isSelected={hasSandbox && isOn}
            onChange={(next) => void setAllowed(next)}
            isDisabled={isUnreadable || !hasSandbox || save.isSaving}
          />
        }
      />
    </SettingsGroup>
  )
}
