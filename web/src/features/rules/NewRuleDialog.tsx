import { useState } from "react"
import { FormDialog } from "@/design-system/feedback/FormDialog"
import { FieldMessages } from "@/design-system/forms/FieldMessages"
import { Segmented } from "@/design-system/navigation/Segmented"
import { RuleTargetPicker } from "@/features/rules/RuleTargetPicker"
import {
  asTargetKind,
  buildTargets,
  emptyListWords,
  KIND_COPY,
  limitStatus,
  searchPlaceholder,
  TARGET_KINDS,
  type TargetKind,
  targetsOfKind,
} from "@/features/rules/ruleTargets"
import { limitDialogCopy } from "@/features/spend/limitCopy"
import type { LimitTarget } from "@/features/spend/limits"
import { SetLimitDialog } from "@/features/spend/SetLimitDialog"
import { readState } from "@/features/spend/useSpend"
import { useKeys } from "@/shared/api/apiKeys"
import { useAllSpendCeilings } from "@/shared/api/budgets"
import {
  useOrganizationContext,
  useOrganizationMembers,
} from "@/shared/api/organizations"
import { useWorkspaces } from "@/shared/api/workspaces"

/**
 * New rule: who it applies to, then the limit. A spend rule is the kind this
 * page can make, through the same budget and ceiling writes as everywhere else
 * a limit is set; the other kinds are made where they live today.
 *
 * Who is two steps, the kind and then one of that kind, so no list mixes
 * workspaces, people and keys. The second step's row is there from the start
 * and every state of it is the same height, so the dialog does not move.
 */
export function NewRuleDialog({ onClose }: { onClose: () => void }) {
  const organization = useOrganizationContext().data?.organization
  const workspaces = useWorkspaces()
  const members = useOrganizationMembers()
  const keys = useKeys()
  const ceilings = useAllSpendCeilings()
  // One clock for the dialog's life, as the Rules page keeps one.
  const [now] = useState(() => Date.now())
  const [kind, setKind] = useState<TargetKind>()
  const [choice, setChoice] = useState("")
  const [target, setTarget] = useState<LimitTarget>()

  const targets = buildTargets({
    organization,
    workspaces: workspaces.data ?? [],
    members: members.data ?? [],
    keys: keys.data ?? [],
    ceilings: ceilings.data ?? [],
    now,
  })
  const ofKind = kind ? targetsOfKind(targets, kind) : []
  const picked =
    kind === "everyone" ? ofKind[0] : ofKind.find((one) => one.value === choice)
  const reads = {
    workspace: readState(workspaces),
    person: readState(members),
    key: readState(keys),
  }
  const limits = readState(ceilings)

  if (target) {
    return (
      <SetLimitDialog
        target={target}
        {...limitDialogCopy(target)}
        onClose={onClose}
      />
    )
  }

  return (
    <FormDialog
      isOpen
      onOpenChange={(open) => {
        if (!open) onClose()
      }}
      title="New rule"
      description="A spend rule: who it applies to, then how much and how often it resets."
      submitLabel={picked?.own ? "Edit limit" : "Next"}
      isPending={false}
      error={ceilings.error ?? workspaces.error ?? members.error ?? keys.error}
      isDirty={picked !== undefined}
      // Until the ceilings are read, a target's own limit is unknown, and the
      // limit dialog would add a second one instead of changing it.
      isSubmitDisabled={!picked || !ceilings.isSuccess}
      onSubmit={() => {
        if (!picked) return
        setTarget({
          scope: picked.scope,
          scopeId: picked.scopeId,
          name: picked.name,
          own: picked.own,
        })
      }}
    >
      <div className="otari-rule-kinds flex flex-col gap-1">
        {/* The track names itself for assistive technology; this is the
            same name for the eye. */}
        <span aria-hidden className="text-body">
          Applies to
        </span>
        <Segmented
          label="Applies to"
          size="md"
          options={TARGET_KINDS}
          value={kind ?? ""}
          onChange={(next) => {
            setKind(asTargetKind(next))
            // A choice belongs to its kind, so it does not outlive it.
            setChoice("")
          }}
        />
      </div>
      {kind === "everyone" ? (
        <div className="flex min-w-0 flex-col gap-1">
          <span className="text-body">Everyone</span>
          <p className="otari-rule-everyone flex items-center text-body text-muted">
            <span className="truncate">
              {picked
                ? `Covers everyone in ${picked.name}.`
                : "Covers the whole organization."}
            </span>
          </p>
          <FieldMessages>
            {picked ? (
              <p className="truncate text-muted tabular-nums">
                {limitStatus(picked, limits)}
              </p>
            ) : null}
          </FieldMessages>
        </div>
      ) : kind ? (
        <RuleTargetPicker
          // A new field per kind, so what was typed for one is not searched
          // in the next.
          key={kind}
          label={KIND_COPY[kind].label}
          placeholder={searchPlaceholder(kind, ofKind.length, reads[kind])}
          targets={ofKind}
          value={choice}
          onChange={setChoice}
          emptyWords={(query) =>
            emptyListWords(kind, query, reads[kind], ofKind.length === 0)
          }
          limits={limits}
        />
      ) : (
        <RuleTargetPicker
          label="Who"
          placeholder="Pick a kind first"
          targets={[]}
          value=""
          onChange={setChoice}
          emptyWords={() => ""}
          isDisabled
        />
      )}
      <p className="text-caption">
        Model access is set on Members &amp; roles, and guardrails on the
        organization’s Guardrails page. Rules for commands and files live in
        each repository’s <code className="font-mono">.otari</code> directory.
      </p>
    </FormDialog>
  )
}
