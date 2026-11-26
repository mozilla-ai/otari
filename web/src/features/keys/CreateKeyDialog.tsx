import { type RefObject, useEffect, useId, useRef, useState } from "react"
import type { CreateKeyResponse, KeyScope } from "@/client"
import { Button } from "@/design-system/actions/Button"
import { ErrorBanner } from "@/design-system/feedback/ErrorBanner"
import { errorMessage } from "@/design-system/feedback/errorMessage"
import { FormDialog } from "@/design-system/feedback/FormDialog"
import { Field } from "@/design-system/forms/Field"
import { Disclosure } from "@/design-system/navigation/Disclosure"
import { Segmented } from "@/design-system/navigation/Segmented"
import { BudgetExemptToggle } from "@/features/keys/BudgetExemptToggle"
import {
  ExistingBudgetSelect,
  OWN_LIMIT,
} from "@/features/keys/ExistingBudgetSelect"
import { ExpiryFields } from "@/features/keys/ExpiryFields"
import { EMPTY_KEY_BUDGET, useKeyBudgets } from "@/features/keys/KeyBudgetField"
import {
  DEFAULT_KEY_SCOPES,
  heldScopes,
  KeyScopeFields,
} from "@/features/keys/KeyScopeFields"
import { KeySecretStep } from "@/features/keys/KeySecretStep"
import { OwnerAccessNote } from "@/features/keys/OwnerAccessNote"
import {
  NO_SERVICE_KEY,
  ServiceKeyFields,
  serviceKeyBody,
  useServiceKeyReady,
} from "@/features/keys/ServiceKeyFields"
import {
  storedUserMismatch,
  type UserMismatchChoice,
  UserMismatchPicker,
} from "@/features/keys/UserMismatchPicker"
import { ModelScopeControl } from "@/features/models/ModelScopeControl"
import { useIsOrganizationManager } from "@/features/organization/access"
import {
  useCurrentOwnerId,
  useMemberAttributionLabels,
} from "@/features/organization/attribution"
import { useKeyBudgetChoices } from "@/features/spend/keyBudgetChoices"
import {
  LIMIT_PERIODS,
  type LimitPeriod,
  useLimitWriter,
} from "@/features/spend/limits"
import { ToolScopeControl } from "@/features/tools/ToolScopeControl"
import { organizationUsers } from "@/features/users/organizationUsers"
import { UserComboBox } from "@/features/users/UserComboBox"
import { useCreateKey, useKeys } from "@/shared/api/apiKeys"
import { useDeploymentOperator } from "@/shared/api/organizations"
import { useUsers } from "@/shared/api/users"
import {
  useNarrowingWorkspace,
  useSelectedWorkspace,
} from "@/shared/hooks/SelectedWorkspace"

const TODAY = new Intl.DateTimeFormat("en-US", {
  month: "short",
  day: "numeric",
})

/** A key left unnamed still reads as something in Sessions and on Home. */
function defaultName(): string {
  return `Key, ${TODAY.format(new Date())}`
}

/**
 * The secret, shown once. Closing is an acknowledgement and the only way out,
 * so a stray Escape cannot lose the key.
 */
function KeyCreated({
  created,
  limitError,
  budgetApplied,
  memberLabels,
  isOpen,
  onOpenChange,
  onCreateAnother,
  returnFocusRef,
}: {
  created: CreateKeyResponse
  /** Set when the key exists but its limit could not be saved. */
  limitError?: unknown
  /** Set when the key exists but the budget picked for it could not be applied. */
  budgetApplied: { error?: unknown; name?: string }
  memberLabels: ReadonlyMap<string, string>
  isOpen: boolean
  onOpenChange: (open: boolean) => void
  onCreateAnother: () => void
  returnFocusRef?: RefObject<HTMLElement | null>
}) {
  const secretRef = useRef<HTMLInputElement | HTMLTextAreaElement | null>(null)
  useEffect(() => {
    // Focus without a selection: the field is concealed, and selecting the
    // stand-in would invite a copy of bullets.
    secretRef.current?.focus()
  }, [])
  return (
    <FormDialog
      isOpen={isOpen}
      onOpenChange={(open) => {
        if (!open) onOpenChange(false)
      }}
      isDismissable={false}
      returnFocusRef={returnFocusRef}
      title="API key created"
      size="lg"
      submitLabel="I’ve saved this key"
      onSubmit={() => onOpenChange(false)}
      isPending={false}
      footerStart={
        <Button variant="ghost" onPress={onCreateAnother}>
          Create another
        </Button>
      }
    >
      <KeySecretStep
        announce="API key created"
        result={created}
        memberLabels={memberLabels}
        secretRef={secretRef}
      />
      {limitError ? (
        <div className="flex flex-col gap-1">
          <ErrorBanner error={limitError} />
          <p className="text-sm text-muted">
            The key works, but its limit was not saved. Set it from the key’s
            row on Home.
          </p>
        </div>
      ) : null}
      {budgetApplied.error ? (
        <p role="alert" className="text-sm text-danger">
          The key was created, but its budget was not applied (
          {errorMessage(budgetApplied.error)}). Edit the key to apply one.
        </p>
      ) : budgetApplied.name ? (
        <p className="text-caption">Capped by {budgetApplied.name}.</p>
      ) : null}
    </FormDialog>
  )
}

/**
 * A key in two answers: what to call it, and what it may spend. The name is
 * how its work and its cost read everywhere else; the limit is the key's own,
 * on top of its owner's, and resets every hour, day, week or month. Only
 * owners and admins may set one, so anyone else is told who does, in its place.
 * When it expires comes next. The rest of what a key can carry,
 * an existing budget included, sits behind "More options", since most keys
 * need none of it. Which agent will use it is not asked: it is
 * read from the requests.
 */
export function CreateKeyDialog({
  isOpen,
  onOpenChange,
  returnFocusRef,
  onSecretShownChange,
}: {
  isOpen: boolean
  onOpenChange: (open: boolean) => void
  /** Where focus goes on close when the control that opened it is gone. */
  returnFocusRef?: RefObject<HTMLElement | null>
  /**
   * Told when a key's secret goes on screen and when it leaves. The secret is
   * shown once, so whoever hosts the dialog must not close it from under it.
   */
  onSecretShownChange?: (isShown: boolean) => void
}) {
  const { isOperator } = useDeploymentOperator()
  const me = useCurrentOwnerId()
  const isManager = useIsOrganizationManager()
  const users = useUsers(isOperator && isOpen)
  const memberLabels = useMemberAttributionLabels()
  // Every key in the organization, workspace unset: with the member labels it
  // is what says a user belongs here, so the owner picker never offers another
  // tenant's people.
  const organizationKeys = useKeys(undefined, isOperator && isOpen)
  const ownerOptions = organizationUsers(
    users.data ?? [],
    memberLabels,
    organizationKeys.data ?? [],
  )
  const { selected: workspace, isLoading: isWorkspaceLoading } =
    useSelectedWorkspace()
  const narrowingWorkspace = useNarrowingWorkspace()
  const create = useCreateKey()
  // Limits are written by owners and admins only, so nobody else asks.
  const limits = useLimitWriter(isManager && isOpen)
  // The budgets a caller may put a key under: the deployment's for the
  // operator, the organization's for an owner or admin, none for anyone else.
  const keyBudgets = useKeyBudgets(isOperator, isOpen)
  const budgetChoices = useKeyBudgetChoices(
    keyBudgets,
    isOpen && keyBudgets.canCap,
  )
  const [budgetChoice, setBudgetChoice] = useState(OWN_LIMIT)
  const [period, setPeriod] = useState<LimitPeriod>("calendar_month")
  const [owner, setOwner] = useState("")
  const [name, setName] = useState("")
  const [limit, setLimit] = useState("")
  const [expiresAt, setExpiresAt] = useState("")
  const [allowedModels, setAllowedModels] = useState<string[] | undefined>()
  const [isScopeValid, setScopeValid] = useState(true)
  const [allowedTools, setAllowedTools] = useState<string[] | undefined>()
  const [isToolScopeValid, setToolScopeValid] = useState(true)
  const [rejectUserMismatch, setRejectUserMismatch] =
    useState<UserMismatchChoice>("inherit")
  // An owner or admin may make the organization a service key; a member is
  // refused one, so is not offered it. The operator's is the deployment's.
  const offersServiceKey = isOperator || isManager
  const serviceKeyScope = isOperator ? "deployment" : "organization"
  const [serviceKey, setServiceKey] = useState(NO_SERVICE_KEY)
  const isServiceKeyReady = useServiceKeyReady(serviceKey, serviceKeyScope)
  const [keyScopes, setKeyScopes] = useState<KeyScope[]>([
    ...DEFAULT_KEY_SCOPES,
  ])
  const [excludeFromBudget, setExcludeFromBudget] = useState(false)
  const [created, setCreated] = useState<CreateKeyResponse>()
  const limitErrorId = useId()
  // Held across the await: a second press lands before `isPending` renders.
  const isSubmitting = useRef(false)
  const hasSecret = created !== undefined
  useEffect(() => {
    onSecretShownChange?.(hasSecret)
    return () => onSecretShownChange?.(false)
  }, [hasSecret, onSecretShownChange])

  const reset = () => {
    setPeriod("calendar_month")
    setOwner("")
    setName("")
    setLimit("")
    setExpiresAt("")
    setAllowedModels(undefined)
    setScopeValid(true)
    setAllowedTools(undefined)
    setToolScopeValid(true)
    setRejectUserMismatch("inherit")
    setServiceKey(NO_SERVICE_KEY)
    setKeyScopes([...DEFAULT_KEY_SCOPES])
    setExcludeFromBudget(false)
    setBudgetChoice(OWN_LIMIT)
    setCreated(undefined)
    create.reset()
    limits.clearError()
    keyBudgets.reset()
  }
  // However it closes, Back under it included, the next open starts empty and
  // never shows the last key's secret.
  // biome-ignore lint/correctness/useExhaustiveDependencies: runs when it closes, not when `reset` changes identity
  useEffect(() => {
    if (!isOpen) reset()
  }, [isOpen])

  if (created) {
    return (
      <KeyCreated
        created={created}
        limitError={limits.error}
        budgetApplied={{
          error: keyBudgets.applyError ?? undefined,
          name: keyBudgets.applied,
        }}
        memberLabels={memberLabels}
        isOpen={isOpen}
        onOpenChange={onOpenChange}
        onCreateAnother={reset}
        returnFocusRef={returnFocusRef}
      />
    )
  }

  const amount = Number(limit)
  const hasLimit = limit.trim() !== ""
  const hasBudgetChoice = budgetChoice !== OWN_LIMIT
  // A limit typed before an existing budget was picked is set aside, not sent.
  const isLimitValid =
    hasBudgetChoice || !hasLimit || (Number.isFinite(amount) && amount > 0)
  const expiresInPast =
    expiresAt !== "" && new Date(expiresAt).getTime() < Date.now()
  const keyName = name.trim() || defaultName()
  // An operator may mint a key for anyone; unless they say otherwise it is
  // their own, as it is for everyone else.
  const keyOwner = owner.trim() || me || ""
  const isServiceKey = offersServiceKey && serviceKey.isServiceKey
  const scopes = heldScopes(keyScopes, isServiceKey)
  const shared = {
    key_name: keyName,
    workspace_id: workspace?.workspace_id,
    expires_at: expiresAt ? new Date(expiresAt).toISOString() : null,
    allowed_models: allowedModels ?? null,
    allowed_tools: allowedTools ?? null,
    scopes,
    reject_user_mismatch: storedUserMismatch(rejectUserMismatch),
  }

  const submit = async () => {
    // One press, one key: a second press before the first settles would mint
    // another key whose secret is never shown.
    if (isSubmitting.current) return
    isSubmitting.current = true
    try {
      await mint()
    } finally {
      isSubmitting.current = false
    }
  }

  const mint = async () => {
    // Only the operator's surface takes an owner and a budget exemption.
    const key = await create.mutateAsync(
      isOperator
        ? {
            ...shared,
            user_id: keyOwner,
            exclude_from_budget: excludeFromBudget,
            ...serviceKeyBody(serviceKey),
          }
        : {
            ...shared,
            ...serviceKeyBody(offersServiceKey ? serviceKey : NO_SERVICE_KEY),
          },
    )
    // The key exists either way; a limit that fails to save is said beside it.
    // An exempt key is never gated, so a limit on it would never bind.
    if (hasBudgetChoice) {
      if (!excludeFromBudget) {
        keyBudgets.apply(key.id, keyName, {
          ...EMPTY_KEY_BUDGET,
          choice: budgetChoice,
        })
      }
    } else if (hasLimit && !excludeFromBudget) {
      await limits.save(
        { scope: "api_token", scopeId: key.id, name: keyName, own: null },
        amount,
        period,
      )
    }
    setCreated(key)
  }

  return (
    <FormDialog
      isOpen={isOpen}
      onOpenChange={onOpenChange}
      returnFocusRef={returnFocusRef}
      title="Create API key"
      description={`A key is how an agent or an app shows up in ${workspace?.name ?? "this workspace"}: its sessions and its cost are filed under the key’s name.`}
      size="lg"
      submitLabel="Create API key"
      isPending={create.isPending || limits.isPending}
      error={create.error}
      isDirty={
        name.length > 0 ||
        owner.length > 0 ||
        limit.length > 0 ||
        hasBudgetChoice ||
        expiresAt.length > 0 ||
        allowedModels !== undefined ||
        allowedTools !== undefined ||
        rejectUserMismatch !== "inherit" ||
        serviceKey.isServiceKey ||
        keyScopes.join() !== DEFAULT_KEY_SCOPES.join() ||
        excludeFromBudget
      }
      isSubmitDisabled={
        isWorkspaceLoading ||
        scopes.length === 0 ||
        !isScopeValid ||
        !isToolScopeValid ||
        !isServiceKeyReady ||
        !isLimitValid ||
        (isOperator && keyOwner === "")
      }
      // A refused create is reported by the dialog from `create.error`.
      onSubmit={() => void submit().catch(() => {})}
    >
      <Field
        label="Name"
        value={name}
        onChange={setName}
        placeholder="claude-code on my-laptop"
        description="Name it for the machine, project or customer it works for. That is how its sessions and its cost read."
        autoFocus
      />
      {hasBudgetChoice ? null : isManager ? (
        <div className="flex flex-col gap-1.5">
          <div className="flex flex-wrap items-end gap-3">
            <div className="w-40">
              <Field
                label="Spend limit, in dollars"
                value={limit}
                onChange={setLimit}
                inputMode="decimal"
                placeholder="No limit"
                isInvalid={!isLimitValid}
                describedBy={isLimitValid ? undefined : limitErrorId}
                shouldReserveMessage={false}
              />
            </div>
            {/* The field keeps 4px under its box; this lifts the rest to it. */}
            <div className="flex items-center gap-3 pb-1">
              <span className="text-body text-muted">every</span>
              <Segmented
                label="Resets every"
                size="md"
                options={LIMIT_PERIODS}
                value={period}
                onChange={(next) => setPeriod(next as LimitPeriod)}
              />
            </div>
          </div>
          {/* The error spans the row: the 160px field is too narrow for it. */}
          {isLimitValid ? (
            <p className="text-sm text-muted">
              Optional. Past it, requests on this key are refused until the
              limit resets. An hourly limit stops an agent caught in a loop. Its
              owner’s limit applies as well.
            </p>
          ) : (
            <p id={limitErrorId} role="alert" className="text-sm text-danger">
              Enter a number above zero, like 12.50, or leave it empty.
            </p>
          )}
        </div>
      ) : keyBudgets.canCap ? null : (
        // Production takes a key's limit from owners and admins only, so the
        // field is not offered; this says so where it would be.
        <div className="flex flex-col gap-0.5">
          <p className="text-body">Spend limit</p>
          <p className="text-caption">
            Set by owners and admins. Any limit on you covers this key as well.
          </p>
        </div>
      )}
      <ExpiryFields
        label="Expires"
        value={expiresAt}
        onChange={setExpiresAt}
        description={
          expiresInPast ? (
            <span className="text-danger">
              That time is in the past; the key would be rejected immediately.
            </span>
          ) : (
            "Optional. The key stops working at this time."
          )
        }
      />
      <Disclosure heading="More options" isFlush>
        <div className="flex flex-col gap-5">
          <KeyScopeFields
            value={keyScopes}
            onChange={setKeyScopes}
            isServiceKey={isServiceKey}
          />
          {keyBudgets.canCap &&
          (budgetChoices.isLoading || budgetChoices.budgets.length > 0) ? (
            <ExistingBudgetSelect
              value={budgetChoice}
              onChange={setBudgetChoice}
              budgets={budgetChoices.budgets}
              isLoading={budgetChoices.isLoading}
              ownLimitLabel={
                isManager ? "A limit for this key alone" : "No budget"
              }
            />
          ) : null}
          {isOperator ? (
            <UserComboBox
              value={owner || me || ""}
              onChange={setOwner}
              users={ownerOptions}
              description="The person or service this key acts for. Their access and limits apply. It is you unless you pick someone else."
            />
          ) : null}
          {isOperator ? (
            <OwnerAccessNote userId={owner || me || ""} users={ownerOptions} />
          ) : null}
          <ModelScopeControl
            workspaceId={narrowingWorkspace}
            title="Restrict this key’s models"
            description={
              isOperator
                ? "By default this key inherits its owner’s access. Optionally narrow it to a subset; a key can never exceed its owner’s allowed models."
                : "By default this key inherits your model access. Narrow it to a subset; a key can never exceed your models."
            }
            anyLabel={
              isOperator ? "Inherit owner access" : "Inherit your access"
            }
            initial={allowedModels}
            onChange={(value, valid) => {
              setAllowedModels(value)
              setScopeValid(valid)
            }}
          />
          <ToolScopeControl
            title="Restrict this key’s search and fetch tools"
            description={
              isOperator
                ? "Which search and fetch tools this key may run on /v1/search and /v1/fetch. Left on Inherit, it takes its owner’s list. With no list on either, it follows its model access, or its owner’s when it has none: unrestricted runs every tool, restricted only a search tool named there as provider:tool (deprecated). A key can never exceed its owner’s."
                : "Which search and fetch tools this key may run on /v1/search and /v1/fetch. Left on Inherit, it takes your list. With no list on either, it follows its model access, or yours when it has none: unrestricted runs every tool, restricted only a search tool named there as provider:tool (deprecated). A key can never exceed yours."
            }
            inheritLabel={
              isOperator ? "Inherit owner’s tools" : "Inherit your tools"
            }
            initial={allowedTools}
            onChange={(value, valid) => {
              setAllowedTools(value)
              setToolScopeValid(valid)
            }}
          />
          <UserMismatchPicker
            value={rejectUserMismatch}
            onChange={setRejectUserMismatch}
            description={
              isOperator ? (
                <>
                  What happens when a request on this key names a different{" "}
                  <code>user</code> than its owner. Accept it for clients that
                  send telemetry there rather than an identity, such as Claude
                  Code. Spend counts toward this key’s owner either way.
                </>
              ) : (
                <>
                  What happens when a request on this key names a different{" "}
                  <code>user</code> than you. Accept it for clients that send
                  telemetry there, such as Claude Code. Spend is yours either
                  way.
                </>
              )
            }
          />
          {offersServiceKey ? (
            // Ruled off: it changes what the key is, not what it may do.
            <div className="border-t border-border pt-5">
              <ServiceKeyFields
                value={serviceKey}
                onChange={setServiceKey}
                scope={serviceKeyScope}
              />
            </div>
          ) : null}
          {isOperator ? (
            <BudgetExemptToggle
              checked={excludeFromBudget}
              onChange={setExcludeFromBudget}
            />
          ) : null}
        </div>
      </Disclosure>
    </FormDialog>
  )
}
