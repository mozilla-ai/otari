import { Button, Card, Chip } from "@heroui/react"
import { useEffect, useState } from "react"

import { canManageWorkspace } from "@/features/organization/roles"
import {
  useClearWorkspaceCodeExecutionPolicy,
  useOrganizationContext,
  useSetWorkspaceCodeExecutionPolicy,
  useWorkspaceCodeExecutionPolicy,
} from "@/shared/api/hooks"
import { Field } from "@/shared/components/Field"
import {
  Checkbox,
  ErrorBanner,
  errorMessage,
  FilterSelect,
  InfoBanner,
} from "@/shared/components/ui"
import { useSelectedWorkspace } from "@/shared/hooks/SelectedWorkspace"

// The layer above the deployment-wide sandbox settings this card sits under: the
// settings above say where code runs, this says which workspaces may ask for it
// and how far. A policy can only narrow, so there is no control here that turns
// anything on that the deployment has not configured; when it has configured
// nothing, the banner says so rather than letting the form imply otherwise.
//
// Three states, not two, which is why the first control is a select rather than
// a toggle: a workspace can be allowed, blocked, or carry no policy at all.
// "Deployment default" is the last of those and is a delete, not a saved
// `enabled: true`.
//
// The image and tool controls are built from `allowed_images` and
// `available_tools` on the policy itself rather than from constants here. The
// image list is the operator's supply-chain allow-list and the server refuses
// anything outside it, so a free-text field would be offering what the write
// rejects; when the operator has curated nothing, the picker is replaced by a
// line saying so.

type Stance = "default" | "allowed" | "blocked"

// The sentinel for "no workspace image", which is a real choice and not an
// absent one: the workspace runs whatever the deployment runs.
const DEPLOYMENT_IMAGE = ""

// The server's own ceilings (`workspace_code_execution_policy_service`): a value
// above either could never take effect, so it is refused rather than stored.
export const MAX_ITERATIONS = 25
export const MAX_EXEC_TIMEOUT_S = 60

function parseCeiling(
  raw: string,
  max: number,
): { value: number | null; valid: boolean } {
  const trimmed = raw.trim()
  if (trimmed === "") return { value: null, valid: true }
  // Digits only, so `0x10` and `1e1` are refused rather than silently read as
  // 16 and 10 by `Number`.
  if (!/^\d+$/.test(trimmed)) return { value: null, valid: false }
  const parsed = Number(trimmed)
  if (!Number.isSafeInteger(parsed) || parsed <= 0 || parsed > max) {
    return { value: null, valid: false }
  }
  return { value: parsed, valid: true }
}

export function WorkspaceCodeExecutionPolicyCard({
  onSaved,
}: {
  onSaved: (message: string) => void
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

  const [stance, setStance] = useState<Stance>("default")
  const [hint, setHint] = useState("")
  const [maxIterations, setMaxIterations] = useState("")
  const [execTimeout, setExecTimeout] = useState("")
  const [image, setImage] = useState(DEPLOYMENT_IMAGE)
  // `null` is "narrow nothing", which is not the same as an empty selection:
  // the server refuses an empty list, so unticking the last tool restores the
  // unnarrowed state rather than saving one nothing can run.
  const [tools, setTools] = useState<string[] | null>(null)
  const [error, setError] = useState("")

  const policy = query.data
  // Hydrate from whatever the server last said, including after a save or a
  // clear, so the form never drifts from the stored policy.
  useEffect(() => {
    if (!policy) return
    setStance(
      !policy.configured ? "default" : policy.enabled ? "allowed" : "blocked",
    )
    setHint(policy.default_purpose_hint ?? "")
    setMaxIterations(
      policy.max_iterations !== null ? String(policy.max_iterations) : "",
    )
    setExecTimeout(
      policy.exec_timeout_s !== null ? String(policy.exec_timeout_s) : "",
    )
    setImage(policy.image ?? DEPLOYMENT_IMAGE)
    setTools(policy.tools)
  }, [policy])

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

  // Disabled while the policy is in flight, so a save cannot race the load that
  // would overwrite the form under it.
  const busy = setPolicy.isPending || clearPolicy.isPending || query.isLoading
  const allowedImages = policy?.allowed_images ?? []
  const availableTools = policy?.available_tools ?? []
  // A pin the operator has since withdrawn is still stored, and the server
  // refuses it on the next request and on the next save. `FilterSelect` renders
  // a controlled native `<select>`, which shows its *first* option when `value`
  // matches none of them, so leaving it out would display "Deployment default"
  // over a policy that is nothing of the kind, and a save would earn a 400
  // naming a value never on screen. Carry it as an option instead, said out
  // loud, so the withdrawal is visible and picking something else is one click.
  const withdrawnImage =
    policy?.image && !allowedImages.includes(policy.image) ? policy.image : null

  // An unset list ticks every box, because that is what it means: the workspace
  // gets whatever the backend serves. Unticking one is therefore a narrowing
  // from the full set, not from nothing. Both ends normalize back to `null`:
  // the full set narrows nothing, and the empty set is refused by the server,
  // so neither is worth storing as a list.
  const toggleTool = (tool: string, isSelected: boolean) => {
    const current = tools ?? availableTools
    const next = isSelected
      ? availableTools.filter((name) => current.includes(name) || name === tool)
      : current.filter((name) => name !== tool)
    setTools(
      next.length === 0 || next.length === availableTools.length ? null : next,
    )
  }

  const save = () => {
    setError("")
    if (stance === "default") {
      clearPolicy.mutate(
        { workspaceId: selected.workspace_id },
        {
          onSuccess: () =>
            onSaved(`${selected.name} uses the deployment default`),
          onError: (err) => setError(errorMessage(err)),
        },
      )
      return
    }
    const iterations = parseCeiling(maxIterations, MAX_ITERATIONS)
    const timeout = parseCeiling(execTimeout, MAX_EXEC_TIMEOUT_S)
    if (!iterations.valid) {
      setError(
        `Max iterations must be a whole number from 1 to ${MAX_ITERATIONS}.`,
      )
      return
    }
    if (!timeout.valid) {
      setError(
        `Execution timeout must be a whole number of seconds from 1 to ${MAX_EXEC_TIMEOUT_S}.`,
      )
      return
    }
    setPolicy.mutate(
      {
        workspaceId: selected.workspace_id,
        body: {
          enabled: stance === "allowed",
          default_purpose_hint: hint.trim() === "" ? null : hint.trim(),
          max_iterations: iterations.value,
          exec_timeout_s: timeout.value,
          image: image === DEPLOYMENT_IMAGE ? null : image,
          tools:
            tools && tools.length > 0 && tools.length < availableTools.length
              ? tools
              : null,
        },
      },
      {
        onSuccess: () =>
          onSaved(`Code execution policy saved for ${selected.name}`),
        onError: (err) => setError(errorMessage(err)),
      },
    )
  }

  return (
    <section className="flex flex-col gap-2">
      <h2 className="text-sm font-semibold text-foreground">
        This workspace ({selected.name})
      </h2>
      <p className="text-sm text-muted">
        Whether requests billed to this workspace may use otari_code_execution,
        and the limits they run under. A workspace policy can only narrow what
        the deployment above allows; it never grants a sandbox the deployment
        has not configured.
      </p>
      <Card>
        <Card.Content className="flex flex-col gap-4 px-5 py-4">
          <ErrorBanner error={query.error} />
          {policy && !policy.sandbox_configured ? (
            <InfoBanner tone="warning">
              This deployment has no sandbox configured, so code execution is
              unavailable here whatever this workspace's policy says. The
              sandbox URL is set above.
            </InfoBanner>
          ) : null}

          <div className="flex flex-wrap items-center gap-3">
            <FilterSelect
              label="Code execution"
              value={stance}
              onChange={(next) => setStance(next as Stance)}
              options={[
                { value: "default", label: "Deployment default" },
                { value: "allowed", label: "Allowed" },
                { value: "blocked", label: "Blocked" },
              ]}
              disabled={busy}
            />
            {policy?.configured === false ? (
              <Chip size="sm" color="default">
                No policy set
              </Chip>
            ) : null}
          </div>

          <Field
            label="Default prompt hint"
            value={hint}
            onChange={setHint}
            placeholder="Leave blank to use the deployment's hint"
            description="Used only when a request declares otari_code_execution without a hint of its own."
          />
          <Field
            label="Max tool-loop iterations"
            value={maxIterations}
            onChange={setMaxIterations}
            placeholder={`Blank for the request's own limit (max ${MAX_ITERATIONS})`}
            description="Lowers the number of model-to-tool rounds. It never raises one."
          />
          <Field
            label="Execution timeout (seconds)"
            value={execTimeout}
            onChange={setExecTimeout}
            placeholder={`Blank for the deployment's ${MAX_EXEC_TIMEOUT_S}s`}
            description="Lowers how long one execution may run. It never raises it."
          />

          {allowedImages.length > 0 || withdrawnImage ? (
            <div className="flex flex-col gap-1">
              <FilterSelect
                label="Sandbox image"
                value={image}
                onChange={setImage}
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
                disabled={busy}
              />
              {withdrawnImage ? (
                <p className="text-xs text-warning">
                  This workspace is pinned to an image the operator no longer
                  approves, so its requests are refused. Pick another, or ask an
                  operator to restore it.
                </p>
              ) : (
                <p className="text-xs text-muted">
                  The image this workspace's code runs in, from the list the
                  operator has approved.
                </p>
              )}
            </div>
          ) : (
            <p className="text-xs text-muted">
              This deployment has approved no sandbox images, so this workspace
              runs whatever the sandbox runs. An operator adds them with
              sandbox_allowed_session_images.
            </p>
          )}

          {availableTools.length > 1 ? (
            <fieldset className="flex flex-col gap-2">
              <legend className="text-xs font-medium text-muted">
                Code-execution tools
              </legend>
              <p className="text-xs text-muted">
                Which of the tools this deployment's sandbox serves the
                workspace may use. Leaving them all ticked narrows nothing; use
                Blocked above to refuse code execution outright.
              </p>
              {availableTools.map((tool) => (
                <Checkbox
                  key={tool}
                  isSelected={tools === null || tools.includes(tool)}
                  isDisabled={busy}
                  onChange={(isSelected) => toggleTool(tool, isSelected)}
                >
                  {tool}
                </Checkbox>
              ))}
            </fieldset>
          ) : (
            // With one tool served there is no subset to choose: ticking and
            // unticking the single box would both mean "narrow nothing", and a
            // control that cannot express anything is worse than a sentence
            // saying so. The checkboxes appear on their own the day a backend
            // serves a second kind.
            <p className="text-xs text-muted">
              This deployment's sandbox serves{" "}
              {availableTools.length === 1 ? availableTools[0] : "no tools"}, so
              there is no tool subset to choose. Use Blocked above to refuse
              code execution for this workspace.
            </p>
          )}

          {error ? (
            <p role="alert" className="text-sm text-danger">
              {error}
            </p>
          ) : null}
          <div className="flex justify-end">
            <Button size="sm" isDisabled={busy} onPress={save}>
              Save
            </Button>
          </div>
        </Card.Content>
      </Card>
    </section>
  )
}
