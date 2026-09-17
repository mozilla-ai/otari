import type { GuardrailFallback, GuardrailMode } from "@/client"
import { ErrorBanner } from "@/design-system/feedback/ErrorBanner"
import { MultiSelect } from "@/design-system/forms/MultiSelect"
import { Select } from "@/design-system/forms/Select"
import { FormSectionRule } from "@/features/tools/FormSectionRule"

// How a stored definition is enforced, which is about the row rather than about
// the guardrail: an operator decides it once and it does not change when they
// swap one vendor for another.
//
// Two scope controls rather than one list of workspaces where "none" means
// "all": an empty list is an ordinary mistake, and it must not read as the
// widest possible scope.

/** The enforcement choices, held together because they are stored and sent together. */
export interface GuardrailEnforcement {
  mode: GuardrailMode
  onUnavailable: GuardrailFallback
  everywhere: boolean
  workspaceIds: string[]
}

/** The choices as the create and update requests carry them. */
export function enforcementFields(value: GuardrailEnforcement) {
  return {
    mode: value.mode,
    on_unavailable: value.onUnavailable,
    applies_to_all_workspaces: value.everywhere,
    workspace_ids: value.everywhere ? [] : value.workspaceIds,
  }
}

export function GuardrailEnforcementFields({
  value,
  onChange,
  workspaces,
  isLoadingWorkspaces,
  workspacesError,
  isDisabled,
}: {
  value: GuardrailEnforcement
  onChange: (next: GuardrailEnforcement) => void
  workspaces: readonly { id: string; name: string }[]
  /** The list is still on its way and nothing cached stands in for it. */
  isLoadingWorkspaces: boolean
  /** Why the list is missing, when it is. */
  workspacesError: unknown
  isDisabled: boolean
}) {
  const set = (patch: Partial<GuardrailEnforcement>) =>
    onChange({ ...value, ...patch })
  // A monitoring definition serves the request whether or not the guardrail
  // answered, so the fallback has nothing to decide until the mode is block.
  // Disabled rather than hidden: the choice is kept for when it is.
  const monitoring = value.mode === "monitor"
  return (
    <>
      <FormSectionRule label="How it runs" />
      <Select
        label="When it flags a request"
        value={value.mode}
        onChange={(next) => set({ mode: next as GuardrailMode })}
        options={[
          { value: "block", label: "Block the request" },
          { value: "monitor", label: "Report only, let it through" },
        ]}
        isDisabled={isDisabled}
        description="Blocking never calls the model. Reporting is how you watch a check before you trust it."
        reserveMessage
      />
      <Select
        label="When it cannot answer"
        value={value.onUnavailable}
        onChange={(next) => set({ onUnavailable: next as GuardrailFallback })}
        options={[
          { value: "block", label: "Block the request" },
          { value: "allow", label: "Let it through" },
        ]}
        isDisabled={isDisabled || monitoring}
        // Not the same as an inconclusive verdict, which is the guardrail
        // answering and never blocks. This is nobody answering at all.
        description={
          monitoring
            ? "Not while it only reports: the request is served and the missing verdict is reported."
            : "Covers a vendor outage, a timeout, and an answer Otari cannot read."
        }
        reserveMessage
      />
      <Select
        label="Where it runs"
        value={value.everywhere ? "all" : "chosen"}
        onChange={(next) => set({ everywhere: next === "all" })}
        options={[
          { value: "all", label: "Every workspace" },
          { value: "chosen", label: "Chosen workspaces" },
        ]}
        isDisabled={isDisabled}
        description={
          value.everywhere
            ? "Including a workspace created later."
            : "Only the workspaces you pick. A new one inherits nothing."
        }
        reserveMessage
      />
      {value.everywhere ? null : (
        <>
          <ErrorBanner error={workspacesError} />
          <MultiSelect
            label="Workspaces"
            value={value.workspaceIds}
            onChange={(next) => set({ workspaceIds: next })}
            options={workspaces.map((workspace) => ({
              id: workspace.id,
              label: workspace.name,
            }))}
            emptyMessage={
              isLoadingWorkspaces
                ? "Loading workspaces…"
                : workspacesError
                  ? "The workspaces could not be loaded."
                  : "This deployment has no workspaces yet."
            }
            countNoun={{ one: "workspace", other: "workspaces" }}
            isInvalid={value.workspaceIds.length === 0}
            errorMessage={
              value.workspaceIds.length === 0
                ? "Pick at least one workspace."
                : undefined
            }
            reserveMessage
          />
        </>
      )}
    </>
  )
}
