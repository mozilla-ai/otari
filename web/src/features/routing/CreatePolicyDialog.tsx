/**
 * The dialog a new policy is created in: first what it is for, then a short
 * form for that alone. Combining several (a split with a backup chain, a
 * guardrail) happens on the policy's own page once it exists.
 */

import { type RefObject, useState } from "react"

import type { User } from "@/client"
import { Button } from "@/design-system/actions/Button"
import { errorMessage } from "@/design-system/feedback/errorMessage"
import { FormDialog } from "@/design-system/feedback/FormDialog"
import { RadioGroup } from "@/design-system/forms/RadioGroup"
import { Disclosure } from "@/design-system/navigation/Disclosure"
import { OfferingPicker } from "@/features/models/OfferingPicker"
import { useMemberAttributionLabels } from "@/features/organization/attribution"
import { userOptionText } from "@/features/users/userOptions"
import {
  useSetOrganizationRoutingPolicy,
  useSetRoutingPolicy,
} from "@/shared/api/routing"
import { useUsers } from "@/shared/api/users"

import { BackupList } from "./BackupList"
import { nameProblemOf, PolicyNameField } from "./PolicyNameField"
import { PolicyPreview } from "./PolicyPreview"
import {
  backupProblems,
  buildSpec,
  fallthroughOf,
  INTENTS,
  type Intent,
  isAtModelCap,
  isDraftReady,
  isFallbackReady,
  isRouted,
  seedDraft,
} from "./policyDraft"
import { describePartialScopeSave, MAX_CANDIDATES } from "./policyModel"
import { RoutedPoolSection } from "./RoutedPoolSection"
import { ScopePicker } from "./ScopePicker"
import { TierDownSection } from "./TierDownSection"

const TITLES: Record<Intent, string> = {
  fallback: "New fallback",
  weighted: "New traffic split",
  knn: "New smart router",
  priority: "New spill-over policy",
  tierdown: "New budget tier-down",
}

const DESCRIPTIONS: Record<Intent, string> = {
  fallback: "the model that serves it, and the backups tried when it fails.",
  weighted: "and the models its traffic is split across.",
  knn: "and the models a router picks between.",
  priority: "and the models tried in order as each fills up.",
  tierdown: "its main model, and the cheaper one used once the budget fills.",
}

/** What to call a user id in prose, matching what the picker shows. */
function useUserLabel(users: User[]): (userId: string) => string {
  const memberLabels = useMemberAttributionLabels()
  const byId = new Map(users.map((entry) => [entry.user_id, entry]))
  return (userId) => {
    const user = byId.get(userId)
    return user === undefined
      ? userId
      : userOptionText(user, memberLabels).label
  }
}

export function CreatePolicyDialog({
  isOpen,
  startAt,
  initialTarget = "",
  returnFocusRef,
  isDeploymentWide,
  workspaceId,
  onClose,
}: {
  /**
   * The page keeps this mounted while it goes false, so the frame plays its
   * exit with the content intact, and keys it per open for a fresh draft.
   */
  isOpen: boolean
  /** Skip the choice and open this form; the choice is one Back away. */
  startAt?: Intent
  /** A `provider:model` the main model starts as, from a deep link. */
  initialTarget?: string
  /** See `FormDialog`'s own prop. */
  returnFocusRef?: RefObject<HTMLElement | null>
  /**
   * Which surface the write goes to: the deployment-wide one for an operator,
   * the tenant-scoped one, which refuses a user scope, for an admin.
   */
  isDeploymentWide: boolean
  /** The workspace the write lands in; null only for a caller in none. */
  workspaceId: string | null
  onClose: () => void
}) {
  const save = useSetRoutingPolicy()
  const saveOrgPolicy = useSetOrganizationRoutingPolicy()
  const isTenantScoped = !isDeploymentWide
  // `/users` is a deployment-wide operator read, and a tenant admin is offered
  // no user scope, so only an operator asks for it, and only while open.
  const users = useUsers(isOpen && !isTenantScoped)
  const labelForUser = useUserLabel(users.data ?? [])

  const [intent, setIntent] = useState<Intent | undefined>(startAt)
  const [choice, setChoice] = useState<Intent>(startAt ?? "fallback")
  const [name, setName] = useState("")
  const [draft, setDraft] = useState(() =>
    seedDraft(startAt ?? "fallback", initialTarget),
  )
  // Null is every caller; a list is one policy per person in it.
  const [userIds, setUserIds] = useState<string[] | null>(null)
  // The scopes already written, with the payload they were written under, so
  // a retry after a part-written save writes only what is missing, and a
  // corrected payload writes everyone again.
  const [written, setWritten] = useState<{ key: string; userIds: string[] }>({
    key: "",
    userIds: [],
  })
  const [partialFailure, setPartialFailure] = useState<Error>()
  const [isWritingScopes, setIsWritingScopes] = useState(false)

  const choose = (next: Intent) => {
    setIntent(next)
    setChoice(next)
    setDraft(seedDraft(next, initialTarget))
  }

  const spec = buildSpec(draft)
  const nameProblem = nameProblemOf(name)
  // "Scoped, nobody chosen" would submit as a global policy, the opposite of
  // what the tab says.
  const scopeReady = userIds === null || userIds.length > 0
  const canSubmit =
    name.trim() !== "" &&
    nameProblem === "" &&
    scopeReady &&
    isDraftReady(draft) &&
    (intent !== "fallback" || isFallbackReady(draft))
  const isDirty =
    intent !== undefined &&
    (name.trim() !== "" ||
      userIds !== null ||
      JSON.stringify(draft) !==
        JSON.stringify(seedDraft(intent, initialTarget)))
  const isPending = save.isPending || saveOrgPolicy.isPending || isWritingScopes

  /** Write one policy per chosen person, one at a time, and report both halves. */
  const writeUserScopes = async (scopes: string[]) => {
    const key = JSON.stringify([name.trim(), spec])
    setIsWritingScopes(true)
    const done = written.key === key ? [...written.userIds] : []
    const failed: { userId: string; reason: string }[] = []
    for (const scope of scopes) {
      if (done.includes(scope)) continue
      try {
        await save.mutateAsync({
          name: name.trim(),
          spec,
          user_id: scope,
          workspace_id: workspaceId,
        })
        done.push(scope)
      } catch (caught) {
        failed.push({ userId: scope, reason: errorMessage(caught) })
      }
    }
    setWritten({ key, userIds: done })
    setIsWritingScopes(false)
    if (failed.length === 0) {
      onClose()
      return
    }
    setPartialFailure(
      new Error(describePartialScopeSave(done, failed, labelForUser)),
    )
  }

  const submit = () => {
    if (intent === undefined) {
      choose(choice)
      return
    }
    if (!canSubmit) return
    // Cleared on every path, or it would outrank a fresh refusal below.
    setPartialFailure(undefined)
    if (isTenantScoped && workspaceId !== null) {
      saveOrgPolicy.mutate(
        { name: name.trim(), spec, workspace_id: workspaceId },
        { onSuccess: onClose },
      )
      return
    }
    if (userIds !== null) {
      void writeUserScopes(userIds)
      return
    }
    save.mutate(
      { name: name.trim(), spec, user_id: null, workspace_id: workspaceId },
      { onSuccess: onClose },
    )
  }

  const scopeSummary =
    userIds === null
      ? "Every caller"
      : userIds.length === 0
        ? "Specific users"
        : userIds.length === 1
          ? "1 user"
          : `${userIds.length} users`

  return (
    <FormDialog
      isOpen={isOpen}
      returnFocusRef={returnFocusRef}
      onOpenChange={(open) => {
        if (!open) onClose()
      }}
      // One width for both steps, so choosing does not resize the frame.
      size="lg"
      title={intent === undefined ? "New policy" : TITLES[intent]}
      description={
        intent === undefined ? (
          "Pick what it does. You can combine more on its page once it exists."
        ) : (
          <>
            A name callers send as <code>model</code>, {DESCRIPTIONS[intent]}
          </>
        )
      }
      submitLabel={intent === undefined ? "Continue" : "Create policy"}
      onSubmit={submit}
      isPending={isPending}
      isSubmitDisabled={intent !== undefined && !canSubmit}
      isDirty={isDirty}
      error={
        intent === undefined
          ? undefined
          : (partialFailure ?? save.error ?? saveOrgPolicy.error)
      }
      footerStart={
        intent === undefined ? null : (
          <Button
            variant="ghost"
            isDisabled={isPending}
            onPress={() => setIntent(undefined)}
          >
            Back
          </Button>
        )
      }
    >
      {intent === undefined ? (
        <RadioGroup
          label="What should it do?"
          value={choice}
          onChange={(value) => setChoice(value as Intent)}
          options={INTENTS}
        />
      ) : (
        <>
          <PolicyNameField value={name} onChange={setName} autoFocus />

          {intent === "fallback" || intent === "tierdown" ? (
            // Reads the catalog, not discovery, which lists only the
            // process-wide providers and misses organization keys.
            <OfferingPicker
              id="main-model"
              label="Main model"
              value={draft.target}
              onChange={(target) => setDraft((prev) => ({ ...prev, target }))}
              isRequired
              description={
                intent === "tierdown"
                  ? "Serves until the budget fills up."
                  : "Serves every request first. Callers never see it."
              }
            />
          ) : (
            <RoutedPoolSection
              pool={draft.pool}
              onChange={(pool) => setDraft((prev) => ({ ...prev, pool }))}
              isAtCap={isAtModelCap(draft)}
            />
          )}

          {intent === "fallback" ? (
            <BackupList
              backups={draft.backups}
              onChange={(backups) => setDraft((prev) => ({ ...prev, backups }))}
              // Only here: an existing policy, on its own page, may have none.
              isRequired
              problems={backupProblems(draft.backups, draft.target)}
              isAtCap={isAtModelCap(draft)}
              maxModels={MAX_CANDIDATES}
            />
          ) : null}

          {intent === "tierdown" ? (
            <TierDownSection
              tierDowns={draft.tierDowns}
              onChange={(tierDowns) =>
                setDraft((prev) => ({ ...prev, tierDowns }))
              }
            />
          ) : null}

          <PolicyPreview
            name={name}
            target={fallthroughOf(draft)}
            backups={draft.backups}
            pool={
              isRouted(draft)
                ? {
                    backend: draft.pool.backend,
                    size: draft.pool.models.length,
                  }
                : undefined
            }
            tierDowns={draft.tierDowns}
          />

          {isTenantScoped ? (
            // Withheld rather than disabled: the tenant-scoped writer refuses
            // a user id, so an organization's entries are workspace-wide.
            <p className="text-caption">
              This applies to everyone in the selected workspace.
            </p>
          ) : (
            <Disclosure
              heading={
                <span className="flex flex-col items-start text-left">
                  <span>Applies to</span>{" "}
                  <span className="text-caption">{scopeSummary}</span>
                </span>
              }
            >
              <ScopePicker
                userIds={userIds}
                users={users.data ?? []}
                onChange={setUserIds}
                // Any landed write fixes the scope: nothing here can take one
                // back.
                isSettled={written.userIds.length > 0}
              />
            </Disclosure>
          )}
        </>
      )}
    </FormDialog>
  )
}
