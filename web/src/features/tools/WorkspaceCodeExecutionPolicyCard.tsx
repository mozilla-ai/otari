import type { UpdateWorkspaceCodeExecutionPolicyRequest } from "@/client"
import { canManageWorkspace } from "@/features/organization/roles"
import {
  ceilingParser,
  PolicyRow,
  parsePhrase,
} from "@/features/tools/PolicyRow"
import { usePolicyWriter } from "@/features/tools/usePolicyWriter"
import { useOrganizationContext } from "@/shared/api/organizations"
import {
  useClearWorkspaceCodeExecutionPolicy,
  useSetWorkspaceCodeExecutionPolicy,
  useWorkspaceCodeExecutionPolicy,
} from "@/shared/api/tools"
import { ErrorBanner } from "@/shared/components/feedback/ErrorBanner"
import { InfoBanner } from "@/shared/components/feedback/InfoBanner"
import { Checkbox } from "@/shared/components/forms/Checkbox"
import { SettingRow } from "@/shared/components/layout/SettingRow"
import { SettingsGroup } from "@/shared/components/layout/SettingsGroup"
import { FilterSelect } from "@/shared/components/navigation/FilterSelect"
import { useSelectedWorkspace } from "@/shared/hooks/SelectedWorkspace"
import { useAutosave } from "@/shared/hooks/useAutosave"

// The layer above the deployment-wide sandbox settings this group sits under:
// the settings above say where code runs, this says which workspaces may ask
// for it and how far. A policy can only narrow, so there is no control here
// that turns anything on the deployment has not configured.
//
// Three states, not two, which is why the first control is a select rather than
// a toggle: a workspace can be allowed, blocked, or carry no policy at all.
// "Deployment default" is the last of those and is a delete, not a saved
// `enabled: true`. While it is chosen there is no policy to narrow, so the rows
// below it have nothing to write and are disabled.
//
// The image and tool controls are built from `allowed_images` and
// `available_tools` on the policy itself rather than from constants here. The
// image list is the operator's supply-chain allow-list and the server refuses
// anything outside it, so a free-text field would be offering what the write
// rejects.

type Stance = "default" | "allowed" | "blocked"

// The sentinel for "no workspace image", which is a real choice and not an
// absent one: the workspace runs whatever the deployment runs.
const DEPLOYMENT_IMAGE = ""

// The server's own ceilings (`workspace_code_execution_policy_service`): a value
// above either could never take effect, so it is refused rather than stored.
export const MAX_ITERATIONS = 25
export const MAX_EXEC_TIMEOUT_S = 60

/**
 * Whether requests billed to this workspace may run generated code, and the
 * limits they run under.
 *
 * A policy can only narrow what the deployment above allows; it never grants a
 * sandbox the deployment has not configured.
 */
export function WorkspaceCodeExecutionPolicyCard({
  docsHref,
}: {
  docsHref: string
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
  const stanceSave = useAutosave()
  const imageSave = useAutosave()
  const toolsSave = useAutosave()
  // One writer for the group: a PUT replaces the whole policy, so two rows
  // saving at once would each carry the other's pre-save value.
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
    }),
    put: (body: UpdateWorkspaceCodeExecutionPolicyRequest) =>
      setPolicy.mutateAsync({
        workspaceId: selected?.workspace_id as string,
        body,
      }),
  })

  if (!selected) {
    return (
      <InfoBanner>
        {workspaceLoading
          ? "Reading the workspaces you belong to."
          : "Per-workspace code execution is set on a workspace you belong to. An owner or admin can add you to one on the Workspaces page."}
      </InfoBanner>
    )
  }

  if (!manages) {
    return (
      <InfoBanner>
        Code execution for {selected.name} is set by an owner or admin of the
        workspace, or of the organization.
      </InfoBanner>
    )
  }

  const policy = query.data
  const stance: Stance = !policy?.configured
    ? "default"
    : policy.enabled
      ? "allowed"
      : "blocked"

  // Disabled until the read has succeeded. Without that the rows sit at
  // "Deployment default" over a workspace that may well have a stored policy,
  // and one change issues the write that drops it.
  const unreadable = query.isLoading || query.isError || !policy
  const narrowingDisabled = unreadable || stance === "default"

  const allowedImages = policy?.allowed_images ?? []
  const availableTools = policy?.available_tools ?? []
  // A pin the operator has since withdrawn is still stored, and the server
  // refuses it on the next request and on the next save. `FilterSelect` falls
  // back to showing an unmatched value bare, so the pin would appear as its own
  // image string with nothing saying it is refused, and a save would earn a 400
  // naming a value the screen presented as ordinary. Carry it as an option
  // instead, said out loud, so the withdrawal is visible and picking something
  // else is one click.
  const withdrawnImage =
    policy?.image && !allowedImages.includes(policy.image) ? policy.image : null
  // The same shape one field over: tool kinds the stored policy names that this
  // deployment no longer serves. Admission is already refusing such a policy
  // (`_pipeline` intersects against `SERVED_TOOL_NAMES`), so the card must show
  // it rather than quietly drop it: a stale entry silently removed on an
  // unrelated save turns a refusal into permission.
  const staleTools = (policy?.tools ?? []).filter(
    (name) => !availableTools.includes(name),
  )
  // What the checkboxes cover: what is served, plus whatever stale kinds the
  // policy still names, so unticking one is how an operator retires it.
  const listedTools = [...availableTools, ...staleTools]
  const storedTools = policy?.tools

  // `enabled` is the one field a patch always restates: the stance select is
  // the only control that changes it, and every other row must not flip it.
  const commitField = (
    patch: Partial<UpdateWorkspaceCodeExecutionPolicyRequest>,
  ) => write({ enabled: stance !== "blocked", ...patch })

  // An unset list ticks every box, because that is what it means: the workspace
  // gets whatever the backend serves. Unticking one is therefore a narrowing
  // from the full set, not from nothing, and unticking the last leaves nothing
  // to store, which the server refuses, so that end returns to unset.
  //
  // A stored list narrows nothing only when it covers everything served *and*
  // names nothing else. Comparing lengths instead would read a stale
  // `["bash_code_execution"]` against a served `["code_execution"]` as the full
  // set, and save `null` over a policy admission is currently refusing, which
  // would grant code execution to a workspace whose row denies it.
  const toggleTool = (tool: string, isSelected: boolean) => {
    const current = storedTools ?? availableTools
    const next = isSelected
      ? listedTools.filter((name) => current.includes(name) || name === tool)
      : current.filter((name) => name !== tool)
    const narrowsNothing =
      next.length === 0 ||
      (availableTools.every((name) => next.includes(name)) &&
        next.every((name) => availableTools.includes(name)))
    void toolsSave.run(() =>
      commitField({ tools: narrowsNothing ? null : next }),
    )
  }

  return (
    <SettingsGroup
      bounded
      title="This workspace"
      description={`Narrows what the deployment allows for requests billed to ${selected.name}. Never widens it, and grants no sandbox the deployment has not configured.`}
      docsHref={docsHref}
    >
      {query.error ? (
        <div className="px-4 py-3">
          <ErrorBanner error={query.error} />
        </div>
      ) : null}
      {policy && !policy.sandbox_configured ? (
        <div className="px-4 py-3">
          <InfoBanner>
            This deployment has no sandbox configured, so code execution is
            unavailable here whatever this workspace's policy says. The sandbox
            URL is set above.
          </InfoBanner>
        </div>
      ) : null}

      <SettingRow
        label="Code execution"
        help="Allow or block the otari_code_execution tool for requests billed here."
        error={stanceSave.error}
        control={
          <FilterSelect
            ariaLabel="Code execution for this workspace"
            value={stance}
            onChange={(next) =>
              void stanceSave.run(() =>
                next === "default"
                  ? clearPolicy.mutateAsync({
                      workspaceId: selected.workspace_id,
                    })
                  : commitField({ enabled: next === "allowed" }),
              )
            }
            options={[
              { value: "default", label: "Deployment default" },
              { value: "allowed", label: "Allowed" },
              { value: "blocked", label: "Blocked" },
            ]}
            disabled={unreadable || stanceSave.isSaving}
          />
        }
      />

      <PolicyRow
        key={`hint-${selected.workspace_id}`}
        label="Prompt hint"
        help="Used when a request declares otari_code_execution without a hint of its own."
        placeholder="Show your working"
        committed={policy?.default_purpose_hint ?? ""}
        parse={parsePhrase}
        commit={(default_purpose_hint) => commitField({ default_purpose_hint })}
        disabled={narrowingDisabled}
      />
      <PolicyRow
        key={`iterations-${selected.workspace_id}`}
        label="Max tool-loop iterations"
        help="Lowers the number of model-to-tool rounds. It never raises one."
        placeholder="10"
        numeric
        committed={
          policy?.max_iterations == null ? "" : String(policy.max_iterations)
        }
        parse={ceilingParser(MAX_ITERATIONS, "rounds")}
        commit={(max_iterations) => commitField({ max_iterations })}
        disabled={narrowingDisabled}
      />
      <PolicyRow
        key={`timeout-${selected.workspace_id}`}
        label="Execution timeout"
        help="Lowers how long one execution may run, in seconds. It never raises it."
        placeholder="30"
        numeric
        committed={
          policy?.exec_timeout_s == null ? "" : String(policy.exec_timeout_s)
        }
        parse={ceilingParser(MAX_EXEC_TIMEOUT_S, "seconds")}
        commit={(exec_timeout_s) => commitField({ exec_timeout_s })}
        disabled={narrowingDisabled}
      />

      <SettingRow
        label="Sandbox image"
        help="The image this workspace's code runs in, from the list the operator has approved."
        error={imageSave.error}
        note={
          withdrawnImage ? (
            <p className="text-caption text-warning">
              This workspace is pinned to an image the operator no longer
              approves, so its requests are refused. Pick another, or ask an
              operator to restore it.
            </p>
          ) : null
        }
        control={
          allowedImages.length > 0 || withdrawnImage ? (
            <FilterSelect
              ariaLabel="Sandbox image for this workspace"
              value={policy?.image ?? DEPLOYMENT_IMAGE}
              onChange={(next) =>
                void imageSave.run(() =>
                  commitField({
                    image: next === DEPLOYMENT_IMAGE ? null : next,
                  }),
                )
              }
              options={[
                { value: DEPLOYMENT_IMAGE, label: "Deployment default" },
                ...allowedImages.map((allowed) => ({
                  value: allowed,
                  label: allowed,
                })),
                ...(withdrawnImage
                  ? [
                      {
                        value: withdrawnImage,
                        label: `${withdrawnImage} (no longer approved)`,
                      },
                    ]
                  : []),
              ]}
              disabled={narrowingDisabled || imageSave.isSaving}
            />
          ) : (
            <span className="text-caption text-subtle">
              None approved, so this workspace runs whatever the sandbox runs
            </span>
          )
        }
      />

      <SettingRow
        label="Code-execution tools"
        help="Which of the tools this deployment's sandbox serves the workspace may use. All ticked narrows nothing."
        error={toolsSave.error}
        note={
          staleTools.length > 0 ? (
            <p className="text-caption text-warning">
              This workspace's policy names {staleTools.join(", ")}, which this
              deployment's sandbox no longer serves, so its requests are
              refused. Untick it and pick what should be allowed, or set the
              stance to Deployment default to drop the policy.
            </p>
          ) : null
        }
        control={
          // With one tool served there is no subset to choose: ticking and
          // unticking the single box would both mean "narrow nothing", and a
          // control that cannot express anything is worse than a sentence
          // saying so. The checkboxes appear the day a backend serves a second.
          listedTools.length > 1 ? (
            <fieldset className="flex flex-col gap-1.5">
              <legend className="sr-only">Code-execution tools</legend>
              {listedTools.map((tool) => (
                <Checkbox
                  key={tool}
                  isSelected={storedTools == null || storedTools.includes(tool)}
                  isDisabled={narrowingDisabled || toolsSave.isSaving}
                  onChange={(isSelected) => toggleTool(tool, isSelected)}
                >
                  {staleTools.includes(tool)
                    ? `${tool} (no longer served)`
                    : tool}
                </Checkbox>
              ))}
            </fieldset>
          ) : (
            <span className="text-caption text-subtle">
              {availableTools.length === 1
                ? `${availableTools[0]} only`
                : "None served"}
            </span>
          )
        }
      />
    </SettingsGroup>
  )
}
