/**
 * The routing policy form, in the dialog every create and edit opens in.
 *
 * Its own module rather than the page's, because it has a catalog entry: a
 * story importing it from `RoutingPage` dragged the table, the confirm dialog
 * and the page's whole hook graph into every render. The model both sides read
 * lives in `policyModel.ts`.
 */

import { Link } from "@tanstack/react-router"
import { type RefObject, useMemo, useRef, useState } from "react"

import type { PolicyGuardrail, PolicySpec } from "@/client"
import { Button } from "@/design-system/actions/Button"
import { FormDialog } from "@/design-system/feedback/FormDialog"
import { Field } from "@/design-system/forms/Field"
import { FieldAction } from "@/design-system/forms/FieldAction"
import { ControlField } from "@/design-system/forms/FieldMessages"
import { Tab, TabRow } from "@/design-system/navigation/TabRow"
import { ModelComboBox } from "@/features/models/ModelComboBox"
import { UserComboBox } from "@/features/users/UserComboBox"
import {
  useCreateAlias,
  useCreateOrganizationAlias,
  useSetOrganizationRoutingPolicy,
  useSetRoutingPolicy,
} from "@/shared/api/routing"
import { useToolSettings } from "@/shared/api/tools"
import { useUsers } from "@/shared/api/users"

import {
  defaultTargetOf,
  initialPool,
  KNN_BACKEND,
  MAX_CANDIDATES,
  type RoutingRow,
  routerBackendOf,
  sharesOf,
  WEIGHTED_BACKEND,
  weightsOf,
} from "./policyModel"

type RouterBackend = typeof KNN_BACKEND | typeof WEIGHTED_BACKEND

/** Whether a guardrails service is configured for this gateway.
 *
 *  A policy guardrail is a request to a separate service (`guardrails_url`). With
 *  no service configured there is nothing to call, so mandating a check would
 *  either fail every request through the policy (mode block, on_unavailable block)
 *  or silently do nothing. Neither is a state to let an operator build by accident,
 *  so the affordance is disabled until a service exists.
 *
 *  While the settings are still loading this returns `true`: a control that starts
 *  enabled and stays enabled is better than one that flickers from disabled to
 *  enabled, which reads as a bug.
 */
function useGuardrailsConfigured(enabled: boolean): {
  configured: boolean
  isLoading: boolean
} {
  const settings = useToolSettings(enabled)
  const field = settings.data?.fields.find(
    (entry) => entry.key === "guardrails_url",
  )
  const value = typeof field?.value === "string" ? field.value.trim() : ""
  return {
    configured: settings.isLoading || value !== "",
    isLoading: settings.isLoading,
  }
}

/** Which entry of `initialPool` serves when the router declines. */
function initialSafeIndex(spec: PolicySpec): number {
  const index = initialPool(spec).indexOf(defaultTargetOf(spec))
  return index === -1 ? 0 : index
}

/** The conditional entries, i.e. everything that is not the fallthrough. */
function conditionsOf(
  spec: PolicySpec,
): { threshold: number; target: string }[] {
  return spec.select
    .filter(
      (entry) =>
        entry.when?.budget_used_pct?.gte !== undefined &&
        entry.target !== undefined,
    )
    .map((entry) => ({
      threshold: entry.when!.budget_used_pct!.gte!,
      target: entry.target!,
    }))
}

/** Who a policy applies to. Same control and wording as the alias scope picker,
 *  because it is the same decision. */
function ScopePicker({
  userId,
  onChange,
  enabled,
}: {
  userId: string | null
  onChange: (userId: string | null) => void
  /**
   * Whether the dialog holding this is open.
   *
   * This picker renders inside the modal, which is `null` while closed, so the
   * roster never fetches with the dialog shut whatever this says. What the gate
   * buys is the exit: the subtree lives for the ~100ms fade, and a refetch
   * landing in it would be work for a form already leaving.
   */
  enabled: boolean
}) {
  const users = useUsers(enabled)
  const scoped = userId !== null

  const modeButton = (value: boolean, label: string) => (
    <Tab
      key={label}
      isActive={scoped === value}
      onPress={() => onChange(value ? "" : null)}
    >
      {label}
    </Tab>
  )

  return (
    <div className="flex flex-col gap-3">
      <ControlField
        label="Applies to"
        description="A global policy resolves for every caller. A user-scoped one resolves only for that user, and takes precedence over a global policy of the same name."
      />
      <TabRow>
        {modeButton(false, "Every caller")}
        {modeButton(true, "One user")}
      </TabRow>
      {scoped ? (
        <UserComboBox
          label="User"
          value={userId ?? ""}
          onChange={onChange}
          users={users.data ?? []}
          placeholder="Pick a user…"
          description="Only this user resolves the policy."
          unknownHint={
            <span className="text-danger">
              No such user. Pick an existing one.
            </span>
          }
        />
      ) : null}
    </div>
  )
}

const MODE_VALUES = ["block", "monitor"] as const

/** A two-value mode switch. The codebase has no Select component and four
 *  hand-rolled `aria-pressed` groups, so this follows that pattern rather than
 *  introducing a fifth idiom. */
function ModeToggle({
  label,
  hint,
  value,
  onChange,
}: {
  label: string
  hint?: string
  value: "block" | "monitor"
  onChange: (value: "block" | "monitor") => void
}) {
  return (
    <div className="flex flex-col gap-1">
      <span className="text-body">{label}</span>
      <TabRow>
        {MODE_VALUES.map((mode) => (
          <Tab
            key={mode}
            isActive={value === mode}
            onPress={() => onChange(mode)}
          >
            {mode}
          </Tab>
        ))}
      </TabRow>
      {hint === undefined ? null : <span className="text-caption">{hint}</span>}
    </div>
  )
}

/** Create or edit a policy.
 *
 *  Reading order mirrors the schema so the form and the YAML teach the same
 *  model: name and scope, then what serves a normal request, then what happens on
 *  failure, then what always runs. The failure and guardrail sections are absent
 *  until summoned rather than collapsed-and-empty, which keeps naming one model a
 *  three-field task.
 */
export function PolicyForm({
  existing,
  initialTarget = "",
  isOpen = true,
  returnFocusRef,
  workspaceId,
  onClose,
}: {
  existing: RoutingRow | null
  initialTarget?: string
  /**
   * Whether the dialog is open, for a caller that keeps this mounted.
   *
   * The create path passes it and stays mounted while it goes false, so the
   * frame can play its exit with the content intact; HeroUI drives that off
   * `data-exiting`, which needs the component still there. The edit path mounts
   * on the row it is editing, keyed on what identifies that row (a policy has
   * no id, so the page keys on kind, scope and name), so it takes the default.
   */
  isOpen?: boolean
  /** Passed through: see `FormDialog`'s own prop. The create path's empty-state
   *  trigger is gone by the time the dialog closes. */
  returnFocusRef?: RefObject<HTMLElement | null>
  /**
   * The workspace a tenant admin's write lands in, or null for an operator.
   *
   * Null is the deployment-wide surface, which defaults the workspace itself;
   * a string is the tenant-scoped one, which requires it named and refuses a
   * user scope. Both mutation pairs are always created, as hooks must be, and
   * only the pair this says is ever mutated.
   */
  workspaceId: string | null
  onClose: () => void
}) {
  const save = useSetRoutingPolicy()
  const saveAlias = useCreateAlias()
  const saveOrgPolicy = useSetOrganizationRoutingPolicy()
  const saveOrgAlias = useCreateOrganizationAlias()
  const tenantScoped = workspaceId !== null
  const editing = existing !== null
  // Editing an alias writes back through the alias API: it is still a row in
  // model_aliases, and silently rewriting it as a policy would leave the original
  // behind under the same name.
  const editingAlias = existing?.kind === "alias"
  // Gated on the dialog being open, and this is the read where that matters:
  // it sits in the form's own body rather than inside the modal, so without the
  // gate it would fetch for every render of the page behind a closed dialog.
  const guardrails_ = useGuardrailsConfigured(isOpen)

  const [name, setName] = useState(existing?.name ?? "")
  const [userId, setUserId] = useState<string | null>(existing?.user_id ?? null)
  const [target, setTarget] = useState(
    existing ? defaultTargetOf(existing.spec) : initialTarget,
  )
  const [chain, setChain] = useState<string[]>(existing?.spec.on_failure ?? [])
  const [conditions, setConditions] = useState(
    existing ? conditionsOf(existing.spec) : [],
  )
  const [guardrails, setGuardrails] = useState<PolicyGuardrail[]>(
    existing?.spec.guardrails ?? [],
  )
  // The learned router's pool, and which of its models serves when the router
  // declines. One list rather than a pool plus a separate "Serves" field: the
  // fallback is always one of the models the router may choose, so asking for it
  // twice made an operator name the strong model in two places and invited them to
  // disagree with themselves. This mirrors what the gateway does with the spec,
  // where the default target joins the pool if it was left out.
  const [candidates, setCandidates] = useState<string[]>(
    existing ? initialPool(existing.spec) : [],
  )
  const [safeIndex, setSafeIndex] = useState<number>(
    existing ? initialSafeIndex(existing.spec) : 0,
  )
  // Which backend orders the pool. The two share the pool control, because both are
  // "these models, one of them per request"; they differ in what decides and in
  // whether a share sits next to each entry.
  const [backend, setBackend] = useState<RouterBackend>(
    existing && routerBackendOf(existing.spec) === WEIGHTED_BACKEND
      ? WEIGHTED_BACKEND
      : KNN_BACKEND,
  )
  // Parallel to `candidates`, so a weight follows its model when one is removed.
  // Held as the text the operator typed rather than as a number: re-rendering a
  // parsed number swallows a half-typed decimal ("7." parses to 7 and renders back
  // as "7") and turns a cleared field into a silent 0. Parsed once, below.
  const [weights, setWeights] = useState<string[]>(() => {
    if (existing === null) return []
    const declared = weightsOf(existing.spec)
    return initialPool(existing.spec).map((selector) =>
      String(declared[selector] ?? 0),
    )
  })
  const routed = candidates.length > 0
  const weighted = routed && backend === WEIGHTED_BACKEND
  // An empty field parses to NaN rather than 0, so a share the operator cleared is
  // unfinished rather than a drain they did not ask for. "Infinity" and a negative
  // are rejected here too, matching what the API refuses.
  const weightValues = weights.map((text) =>
    text.trim() === "" ? Number.NaN : Number(text),
  )
  const weightsWellFormed = weightValues.every(
    (value) => Number.isFinite(value) && value >= 0,
  )
  const shares = sharesOf(
    weightValues.map((value) =>
      Number.isFinite(value) ? Math.max(0, value) : 0,
    ),
  )
  // With a router, the fallthrough is the marked model; without one it is the single
  // "Serves" field.
  const effectiveTarget = routed ? (candidates[safeIndex] ?? "") : target

  const nameHasDelimiter = /[:/]/.test(name)
  // A policy's name is its key, so a rename is a move rather than an edit: the API
  // takes it as `rename_from` on the same write as the spec. Aliases have no such
  // verb, so their name stays fixed here.
  const previousName = existing?.name ?? ""
  const renaming =
    editing &&
    !editingAlias &&
    name.trim() !== "" &&
    name.trim() !== previousName
  const scopeReady = userId === null || userId.trim() !== ""
  const conditionsReady = conditions.every(
    (c) => c.target.trim() !== "" && c.threshold > 0 && c.threshold < 100,
  )
  const guardrailsReady = guardrails.every((g) => g.profile.trim() !== "")
  // A model named twice is refused by the API, and on a weighted policy it would
  // also collapse in the weight map: two rows, one key, so the split submitted is
  // not the split the form showed. Checked over the named rows only, so a pair of
  // still-empty rows reads as unfinished rather than as a duplicate.
  const namedCandidates = candidates
    .map((entry) => entry.trim())
    .filter((entry) => entry !== "")
  const duplicateCandidate =
    new Set(namedCandidates).size !== namedCandidates.length
  // Two, not one: ranking a single model is not a decision, and the API refuses it.
  const candidatesReady =
    !routed ||
    (candidates.length >= 2 &&
      candidates.every((entry) => entry.trim() !== "") &&
      !duplicateCandidate &&
      effectiveTarget.trim() !== "")
  // An all-zero split would select nothing and the policy would always serve its
  // default, so the API refuses it. Caught here so the form cannot author it.
  const splitReady =
    !weighted || (weightsWellFormed && weightValues.some((value) => value > 0))
  // The server caps the compiled plan at MAX_CANDIDATES, counting the routed pool
  // plus the failure chain. Enforced here too so the form cannot author a policy it
  // then fails to save: a rule the UI knows about should not arrive as a 400.
  const plannedCandidates = (candidates.length || 1) + chain.length
  const atCandidateCap = plannedCandidates >= MAX_CANDIDATES
  const overCandidateCap = plannedCandidates > MAX_CANDIDATES
  const canSubmit =
    name.trim() !== "" &&
    effectiveTarget.trim() !== "" &&
    !nameHasDelimiter &&
    scopeReady &&
    conditionsReady &&
    guardrailsReady &&
    candidatesReady &&
    splitReady &&
    !overCandidateCap &&
    chain.every((entry) => entry.trim() !== "")

  // Built in plan order, with the fallthrough last, which is what the schema
  // requires: an entry after the default could never be reached.
  const spec: PolicySpec = useMemo(
    () => ({
      select: [
        ...conditions.map((condition) => ({
          when: { budget_used_pct: { gte: condition.threshold } },
          target: condition.target.trim(),
        })),
        // After the conditions, before the fallthrough: an explicit tier-down is
        // the operator overriding the router, and the router is what runs when no
        // condition applies.
        ...(routed
          ? [
              {
                router: backend,
                candidates: candidates.map((entry) => entry.trim()),
                // Keyed by selector, which is how the server reads it. Only for the
                // weighted backend: a weight map on a knn entry is refused, because
                // it would read as a split and do nothing.
                ...(weighted
                  ? {
                      weights: Object.fromEntries(
                        candidates.map((entry, index) => {
                          const value = weightValues[index] ?? 0
                          return [
                            entry.trim(),
                            Number.isFinite(value) ? Math.max(0, value) : 0,
                          ]
                        }),
                      ),
                    }
                  : {}),
              },
            ]
          : []),
        { default: effectiveTarget.trim() },
      ],
      ...(chain.length > 0
        ? { on_failure: chain.map((entry) => entry.trim()) }
        : {}),
      ...(guardrails.length > 0 ? { guardrails } : {}),
    }),
    [
      conditions,
      candidates,
      routed,
      backend,
      weighted,
      weightValues,
      effectiveTarget,
      chain,
      guardrails,
    ],
  )

  // Everything the operator can change, against what it was seeded with. A guard
  // that watched only the name would be worse than none on a form this long,
  // where a stray Escape can land ten minutes into building a fallback chain.
  const draft = JSON.stringify({
    name,
    userId,
    target,
    chain,
    conditions,
    guardrails,
    candidates,
    safeIndex,
    backend,
    weights,
  })
  const seeded = useRef(draft)
  const isDirty = draft !== seeded.current

  // An alias has exactly one target, so growing one a chain, a condition, or a
  // guardrail makes it a policy. Saving it as a policy alone would leave the alias
  // row in place under the same name, and the API refuses that collision, so the
  // form keeps an alias an alias and points the operator at the way across.
  const outgrewAlias =
    editingAlias &&
    (chain.length > 0 ||
      conditions.length > 0 ||
      guardrails.length > 0 ||
      candidates.length > 0)
  const pending =
    save.isPending ||
    saveAlias.isPending ||
    saveOrgPolicy.isPending ||
    saveOrgAlias.isPending

  const submit = () => {
    if (!canSubmit || outgrewAlias) return
    const scope = userId === null ? null : userId.trim()
    if (editingAlias) {
      if (workspaceId !== null) {
        saveOrgAlias.mutate(
          {
            name: name.trim(),
            target: effectiveTarget.trim(),
            workspace_id: workspaceId,
          },
          { onSuccess: onClose },
        )
        return
      }
      saveAlias.mutate(
        { name: name.trim(), target: effectiveTarget.trim(), user_id: scope },
        { onSuccess: onClose },
      )
      return
    }
    if (workspaceId !== null) {
      saveOrgPolicy.mutate(
        {
          name: name.trim(),
          spec,
          workspace_id: workspaceId,
          ...(renaming ? { rename_from: previousName } : {}),
        },
        { onSuccess: onClose },
      )
      return
    }
    save.mutate(
      {
        name: name.trim(),
        spec,
        user_id: scope,
        ...(renaming ? { rename_from: previousName } : {}),
      },
      { onSuccess: onClose },
    )
  }

  return (
    <FormDialog
      isOpen={isOpen}
      returnFocusRef={returnFocusRef}
      onOpenChange={(open) => {
        if (!open) onClose()
      }}
      // `lg` on every routing dialog: this form grows a fallback chain, a
      // condition tier, a weighted split and a guardrail list as they are asked
      // for, and a frame that changed width while an operator built one up
      // would read as a different dialog each time.
      size="lg"
      title={
        editing
          ? `Edit ${existing.kind === "alias" ? "alias" : "policy"}`
          : "New policy"
      }
      // The policy's identity, in the mono face that says it is a value rather
      // than prose. Here rather than in the title because `title` is a string.
      description={
        editing ? (
          <>
            <code>{existing.name}</code>
            {existing.user_id ? (
              <>
                {" "}
                for user <code>{existing.user_id}</code>
              </>
            ) : null}
          </>
        ) : (
          <>
            What callers send as <code>model</code>, and the model that answers
            it.
          </>
        )
      }
      submitLabel={editing ? "Save" : "Create policy"}
      onSubmit={submit}
      isPending={pending}
      isSubmitDisabled={!canSubmit || outgrewAlias}
      isDirty={isDirty}
      // All four writers, not two: an organization-scoped save fails through its
      // own mutation, and with those two missing the refusal was swallowed and
      // the panel just sat there.
      error={
        save.error ??
        saveAlias.error ??
        saveOrgPolicy.error ??
        saveOrgAlias.error
      }
      footerStart={
        <p className="text-caption">In effect for new requests within 30s.</p>
      }
    >
      <div className="grid gap-4 sm:grid-cols-2">
        {editingAlias ? (
          <div className="flex flex-col gap-1">
            <span className="text-body">Alias name</span>
            <code className="text-sm text-muted">{previousName}</code>
            <span className="text-xs text-muted">
              An alias name is its key and cannot be changed here. Delete and
              recreate to change it.
            </span>
          </div>
        ) : (
          <Field
            label="Policy name"
            value={name}
            onChange={setName}
            placeholder="fast"
            isRequired
            // Only on create. Dropping an operator who clicked Edit to change a
            // target into the name box invites a typo in the one field that is
            // the policy's identity.
            autoFocus={!editing}
            description={
              nameHasDelimiter ? (
                <span className="text-danger">
                  A policy name cannot contain “:” or “/”.
                </span>
              ) : renaming ? (
                <span>
                  Renames <code>{previousName}</code> on save. Callers have to
                  send the new name from then on, and usage already recorded
                  keeps the old one.
                </span>
              ) : editing ? (
                <>
                  What callers send as <code>model</code>. Change it to rename
                  the policy.
                </>
              ) : (
                <>
                  What callers send as <code>model</code>.
                </>
              )
            }
          />
        )}
        {routed ? (
          <div className="flex flex-col gap-1">
            <span className="text-body">Serves</span>
            <span className="text-sm text-foreground">
              {effectiveTarget.trim() === "" ? (
                <span className="text-muted">
                  whichever model you mark below
                </span>
              ) : (
                <code>{effectiveTarget}</code>
              )}
            </span>
            <span className="text-xs text-muted">
              {weighted
                ? "The split picks per request, so this policy has no single target. The model marked below is what serves a caller who opts out."
                : "A router picks per request, so this policy has no single target. The model marked below is what serves when the router does not choose."}
            </span>
          </div>
        ) : (
          <ModelComboBox
            label="Serves"
            value={target}
            onChange={setTarget}
            isRequired
            description="The model that serves a normal request. Callers never see it."
          />
        )}
      </div>

      {editing ? (
        <p className="text-caption">
          Who this applies to is the other half of the key. It cannot be changed
          here: delete and recreate to move it between scopes.
        </p>
      ) : tenantScoped ? (
        // Withheld rather than disabled: a user id is a deployment-wide
        // identifier, so the tenant-scoped writer refuses one outright and
        // an organization's entries are workspace-wide. Offering the picker
        // here would take a value the API is going to reject.
        <p className="text-caption">
          This applies to everyone in the selected workspace.
        </p>
      ) : (
        <ScopePicker userId={userId} onChange={setUserId} enabled={isOpen} />
      )}

      {/* Conditional tier-down */}
      {conditions.length > 0 ? (
        <div className="flex flex-col gap-3 border border-control-border p-3">
          <ControlField
            label="Instead, when the budget fills up"
            description="Checked before the model above. A threshold must be under 100: the budget gate refuses a request before selection once the cap is reached, so a rule at 100 could never fire."
          />
          {conditions.map((condition, index) => (
            <div key={index} className="flex flex-wrap items-end gap-3">
              <Field
                label="Budget used at least (%)"
                value={String(condition.threshold)}
                onChange={(value) =>
                  setConditions((prev) =>
                    prev.map((c, i) =>
                      i === index ? { ...c, threshold: Number(value) || 0 } : c,
                    ),
                  )
                }
                description={
                  condition.threshold >= 100 ? (
                    <span className="text-danger">Must be under 100.</span>
                  ) : condition.threshold <= 0 ? (
                    // Clearing the box coerces to 0 through the `onChange`
                    // above, which blocks the submit; without this the button
                    // dies with nothing on screen saying why.
                    <span className="text-danger">Must be over 0.</span>
                  ) : undefined
                }
              />
              <div className="min-w-56 flex-1">
                <ModelComboBox
                  label="Use instead"
                  value={condition.target}
                  onChange={(value) =>
                    setConditions((prev) =>
                      prev.map((c, i) =>
                        i === index ? { ...c, target: value } : c,
                      ),
                    )
                  }
                  isRequired
                />
              </div>
              <FieldAction>
                <Button
                  variant="ghost"
                  onPress={() =>
                    setConditions((prev) => prev.filter((_, i) => i !== index))
                  }
                >
                  Remove
                </Button>
              </FieldAction>
            </div>
          ))}
        </div>
      ) : null}

      {/* The routed pool: one control for both backends, because both are "these
              models, one of them per request". What differs is who decides, and
              whether a share sits next to each entry. */}
      {candidates.length > 0 ? (
        <div className="flex flex-col gap-3 border border-control-border p-3">
          <ControlField
            label={
              weighted ? "Split traffic between" : "The router chooses between"
            }
            description={
              weighted
                ? "Each request goes to one of these, drawn in proportion to its share. Shares are relative, so 70 and 30 mean the same as 7 and 3. No pricing needed."
                : "For each request, the cheapest of these that past scoring says is good enough. Every model here needs pricing, because the router weighs quality against cost."
            }
          />
          {candidates.map((entry, index) => (
            <div key={index} className="flex flex-wrap items-end gap-3">
              <div className="min-w-56 flex-1">
                <ModelComboBox
                  label={`Model ${index + 1}`}
                  value={entry}
                  onChange={(value) =>
                    setCandidates((prev) =>
                      prev.map((c, i) => (i === index ? value : c)),
                    )
                  }
                  isRequired
                />
              </div>
              {weighted ? (
                <div className="flex items-end gap-2">
                  <Field
                    label="Share"
                    value={weights[index] ?? ""}
                    onChange={(value) =>
                      setWeights((prev) =>
                        prev.map((weight, i) => (i === index ? value : weight)),
                      )
                    }
                    // The percentage, not the number they typed: relative weights
                    // are easy to write and hard to read, and this is the line
                    // that says a zero-weight model is drained rather than gone.
                    description={
                      !Number.isFinite(weightValues[index] ?? Number.NaN) ||
                      (weightValues[index] ?? 0) < 0
                        ? "A number, zero or more"
                        : (weightValues[index] ?? 0) > 0
                          ? `${Math.round(shares[index] ?? 0)}% of requests`
                          : "No weighted traffic; still tried if another fails"
                    }
                  />
                </div>
              ) : null}
              <FieldAction>
                <label className="flex items-center gap-2 text-body">
                  <input
                    type="radio"
                    name="router-safe-choice"
                    checked={safeIndex === index}
                    onChange={() => setSafeIndex(index)}
                  />
                  {weighted ? "Serves on opt-out" : "Serves when unsure"}
                </label>
              </FieldAction>
              <FieldAction>
                <Button
                  variant="ghost"
                  onPress={() => {
                    setCandidates((prev) => prev.filter((_, i) => i !== index))
                    setWeights((prev) => prev.filter((_, i) => i !== index))
                    // Keep the mark on the same model where possible; if the marked
                    // one went, fall back to the first, never to nothing.
                    setSafeIndex((prev) =>
                      index < prev ? prev - 1 : index === prev ? 0 : prev,
                    )
                  }}
                >
                  Remove
                </Button>
              </FieldAction>
            </div>
          ))}
          <p className="text-caption">
            {weighted ? (
              <>
                The marked model serves a caller who sends{" "}
                <code>Otari-Router: off</code>, which is the way to pin traffic
                to one provider during an incident. A model that fails before
                responding moves the request to another model in this pool, by
                the same shares, before any fallback below.
              </>
            ) : (
              <>
                The marked model serves whenever the router does not choose: too
                few scored examples, a weakly supported pick, a request carrying
                tools, or a caller sending <code>Otari-Router: off</code>. Mark
                the one you would have picked without a router.
              </>
            )}
          </p>
          {candidates.length < 2 ? (
            <p className="text-caption text-danger">
              Name at least two models.{" "}
              {weighted ? "Splitting traffic one way" : "Ranking one"} is not a
              routing decision.
            </p>
          ) : null}
          {duplicateCandidate ? (
            <p className="text-caption text-danger">
              Name each model once.{" "}
              {weighted
                ? "A model listed twice has one share, not two, so the split saved would not be the one shown."
                : "A pool that repeats a model is refused."}
            </p>
          ) : null}
          {weighted && !weightsWellFormed ? (
            <p className="text-caption text-danger">
              Every share is a number of zero or more. Use 0 to drain a model
              without removing it.
            </p>
          ) : weighted && !splitReady ? (
            <p className="text-caption text-danger">
              Give at least one model a share above zero, or this policy can
              never send traffic anywhere but its marked model.
            </p>
          ) : null}
          <div className="flex flex-wrap items-baseline gap-2">
            <button
              type="button"
              disabled={atCandidateCap}
              className={
                atCandidateCap
                  ? "cursor-not-allowed text-sm text-muted opacity-60"
                  : "text-sm text-link hover:underline"
              }
              onClick={() => {
                setCandidates((prev) => [...prev, ""])
                // Zero, not an invented share: adding a provider must not move
                // traffic onto it before the operator says how much.
                setWeights((prev) => [...prev, "0"])
              }}
            >
              + Another model
            </button>
            {atCandidateCap ? (
              <span className="text-xs text-muted">
                A policy dispatches at most {MAX_CANDIDATES} models, counting
                the fallback chain. Remove a fallback to add another.
              </span>
            ) : null}
          </div>
        </div>
      ) : null}

      {/* Failure chain */}
      {chain.length > 0 ? (
        <div className="flex flex-col gap-3 border border-control-border p-3">
          <ControlField
            label="If that fails, try"
            description="Tried in order after a retryable failure. Not tried once tokens have started streaming, or after a 400/401/403, which every provider would reject the same way."
          />
          {chain.map((entry, index) => (
            <div key={index} className="flex flex-wrap items-end gap-3">
              <div className="min-w-56 flex-1">
                <ModelComboBox
                  label={`Fallback ${index + 1}`}
                  value={entry}
                  onChange={(value) =>
                    setChain((prev) =>
                      prev.map((e, i) => (i === index ? value : e)),
                    )
                  }
                  isRequired
                />
              </div>
              <FieldAction>
                <Button
                  variant="ghost"
                  onPress={() =>
                    setChain((prev) => prev.filter((_, i) => i !== index))
                  }
                >
                  Remove
                </Button>
              </FieldAction>
            </div>
          ))}
          <div className="flex flex-wrap items-baseline gap-2">
            <button
              type="button"
              disabled={atCandidateCap}
              className={
                atCandidateCap
                  ? "cursor-not-allowed text-sm text-muted opacity-60"
                  : "text-sm text-link hover:underline"
              }
              onClick={() => setChain((prev) => [...prev, ""])}
            >
              + Another fallback
            </button>
            {atCandidateCap ? (
              <span className="text-xs text-muted">
                A policy dispatches at most {MAX_CANDIDATES} models in total.
              </span>
            ) : null}
          </div>
        </div>
      ) : null}

      {/* Guardrails */}
      {guardrails.length > 0 ? (
        <div className="flex flex-col gap-3 border border-control-border p-3">
          <div>
            <span className="text-body">Always check</span>
            <p className="text-caption">
              Runs on every request through this policy. Callers can add their
              own guardrails but cannot weaken these.
            </p>
            {guardrails_.configured ? null : (
              <p className="mt-1 text-caption text-warning">
                No guardrails service is configured, so these cannot run. With{" "}
                <code>if the service is down</code> set to block, every request
                through this policy is refused until one is configured.{" "}
                <Link to="/tools" className="underline">
                  Set one up
                </Link>
                , or remove the guardrail.
              </p>
            )}
          </div>
          {guardrails.map((guardrail, index) => (
            <div key={index} className="flex flex-col gap-3">
              <div className="flex flex-wrap items-end gap-3">
                <Field
                  label="Profile"
                  value={guardrail.profile}
                  onChange={(value) =>
                    setGuardrails((prev) =>
                      prev.map((g, i) =>
                        i === index ? { ...g, profile: value } : g,
                      ),
                    )
                  }
                  placeholder="prompt-injection"
                  isRequired
                  description="A profile configured on the guardrails service."
                />
                <ModeToggle
                  label="Mode"
                  value={guardrail.mode}
                  onChange={(mode) =>
                    setGuardrails((prev) =>
                      prev.map((g, i) => (i === index ? { ...g, mode } : g)),
                    )
                  }
                  hint="block rejects a flagged request; monitor records it and serves anyway."
                />
                <ModeToggle
                  label="If the service is down"
                  value={guardrail.on_unavailable ?? "block"}
                  onChange={(mode) =>
                    setGuardrails((prev) =>
                      prev.map((g, i) =>
                        i === index ? { ...g, on_unavailable: mode } : g,
                      ),
                    )
                  }
                  hint="block fails closed, so a guardrails outage refuses every request through this policy."
                />
                <FieldAction>
                  <Button
                    variant="ghost"
                    onPress={() =>
                      setGuardrails((prev) =>
                        prev.filter((_, i) => i !== index),
                      )
                    }
                  >
                    Remove
                  </Button>
                </FieldAction>
              </div>
              {guardrail.mode === "block" &&
              (guardrail.on_unavailable ?? "block") === "block" ? (
                <div className="text-caption text-warning">
                  With both set to block, a guardrails-service outage rejects
                  every request through this policy, ahead of any fallback
                  above.
                </div>
              ) : null}
            </div>
          ))}
        </div>
      ) : null}

      {/* Complexity is summoned, never presented: naming one model stays a
              three-field task. */}
      <div className="flex flex-wrap gap-3 text-sm">
        {conditions.length === 0 ? (
          <button
            type="button"
            className="text-link hover:underline"
            onClick={() => setConditions([{ threshold: 80, target: "" }])}
          >
            + Tier down when the budget fills up
          </button>
        ) : null}
        {chain.length === 0 ? (
          <button
            type="button"
            className="text-link hover:underline"
            onClick={() => setChain([""])}
          >
            + Add a fallback chain
          </button>
        ) : null}
        {candidates.length === 0 ? (
          <button
            type="button"
            className="text-link hover:underline"
            // Seeded with the policy's own target, marked as the safe choice, so
            // the pool starts from the model this policy already serves and the
            // operator adds the cheaper one rather than restating everything.
            onClick={() => {
              setBackend(KNN_BACKEND)
              setCandidates([target.trim() || "", ""])
              setWeights([])
              setSafeIndex(0)
            }}
          >
            + Let a router pick the cheapest good-enough model
          </button>
        ) : null}
        {candidates.length === 0 ? (
          <button
            type="button"
            className="text-link hover:underline"
            // An even split of the policy's own target with one more provider:
            // the neutral starting point for load balancing, which the operator
            // then skews. Seeding 90/10 would be guessing at a canary.
            onClick={() => {
              setBackend(WEIGHTED_BACKEND)
              setCandidates([target.trim() || "", ""])
              setWeights(["50", "50"])
              setSafeIndex(0)
            }}
          >
            + Split traffic across providers by weight
          </button>
        ) : null}
        {guardrails.length === 0 ? (
          // Disabled rather than hidden, and never disabled silently: a hidden
          // control teaches nothing, and a greyed-out one with no explanation
          // is worse. The reason sits next to it with the route to fixing it,
          // as text rather than a tooltip so it is readable on touch and by a
          // screen reader.
          <span className="flex flex-wrap items-baseline gap-2">
            <button
              type="button"
              disabled={!guardrails_.configured}
              aria-describedby={
                guardrails_.configured ? undefined : "guardrails-unavailable"
              }
              className={
                guardrails_.configured
                  ? "text-link hover:underline"
                  : "cursor-not-allowed text-muted opacity-60"
              }
              onClick={() =>
                setGuardrails([
                  { profile: "", mode: "block", on_unavailable: "block" },
                ])
              }
            >
              + Add guardrails
            </button>
            {guardrails_.configured ? null : (
              <span id="guardrails-unavailable" className="text-caption">
                No guardrails service is configured, so there would be nothing
                to call.{" "}
                <Link to="/tools" className="text-link hover:underline">
                  Set one up in Tools &amp; Guardrails
                </Link>
                .
              </span>
            )}
          </span>
        ) : null}
      </div>

      {/* Each of these explains a mode chosen above it, so it belongs beside
          that choice. The footer's caption is the one sentence about the save
          itself. */}
      {routed && !weighted ? (
        <p className="text-caption">
          A new router serves the model above until it has scored examples.
          Recording them is an API job for now (
          <code>POST /api/v1/routing/preferences/rank</code>); open{" "}
          <b>Examples</b> on the row afterwards to watch it warm up.
        </p>
      ) : null}
      {weighted ? (
        <p className="text-caption">
          Each request is drawn independently, so the shares hold over traffic
          rather than over any ten requests, and they behave the same behind any
          number of replicas.
        </p>
      ) : null}
      {outgrewAlias ? (
        <p className="text-warning text-xs">
          An alias holds one target. To add a fallback, a condition, or a
          guardrail, delete this alias and create a policy with the same name.
        </p>
      ) : null}
    </FormDialog>
  )
}
