import { useState } from "react"
import type {
  ApiKey,
  KeyScope,
  UpdateKeyRequest,
  UpdateOwnKeyRequest,
} from "@/client"
import { FormDialog } from "@/design-system/feedback/FormDialog"
import { Field } from "@/design-system/forms/Field"
import { useDirtySnapshot } from "@/design-system/forms/useDirtySnapshot"
import { BudgetExemptToggle } from "@/features/keys/BudgetExemptToggle"
import { ExpiryFields } from "@/features/keys/ExpiryFields"
import {
  ceilingLabel,
  draftFor,
  EMPTY_KEY_BUDGET,
  KeyBudgetField,
  keyBudgetIncomplete,
  useKeyBudgets,
} from "@/features/keys/KeyBudgetField"
import { heldScopes, KeyScopeFields } from "@/features/keys/KeyScopeFields"
import { OwnerAccessNote } from "@/features/keys/OwnerAccessNote"
import {
  NO_SERVICE_KEY,
  ServiceKeyFields,
  serviceKeyBody,
  serviceKeyChoice,
  useServiceKeyReady,
} from "@/features/keys/ServiceKeyFields"
import {
  storedUserMismatch,
  type UserMismatchChoice,
  UserMismatchPicker,
  userMismatchChoice,
} from "@/features/keys/UserMismatchPicker"
import { ModelScopeControl } from "@/features/models/ModelScopeControl"
import { useKeyBudgetChoices } from "@/features/spend/keyBudgetChoices"
import { KeyMcpAccess } from "@/features/tools/KeyMcpAccess"
import { ToolScopeControl } from "@/features/tools/ToolScopeControl"
import { useUpdateKey } from "@/shared/api/apiKeys"
import { useUsers } from "@/shared/api/users"

// datetime-local wants "YYYY-MM-DDTHH:mm" in local time.
function toDatetimeLocal(iso: string | null): string {
  if (!iso) return ""
  const d = new Date(iso)
  if (Number.isNaN(d.getTime())) return ""
  const pad = (value: number) => String(value).padStart(2, "0")
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}T${pad(d.getHours())}:${pad(d.getMinutes())}`
}

/**
 * Everything about an existing key that its caller may change, in one dialog:
 * name, expiry, budget, model and tool access, scopes, MCP access, and the
 * operator's own fields (budget exemption, service key).
 *
 * `isDeploymentWide` is the operator's surface, which names any key's owner
 * and may exempt a key from budget; a member's body is the member surface's
 * narrower shape.
 */
export function EditKeyForm({
  isOpen,
  isDeploymentWide,
  apiKey,
  onClose,
}: {
  isOpen: boolean
  isDeploymentWide: boolean
  apiKey: ApiKey
  onClose: () => void
}) {
  const update = useUpdateKey()
  const keyBudgets = useKeyBudgets(isDeploymentWide)
  const budgetChoices = useKeyBudgetChoices(
    keyBudgets,
    isOpen && keyBudgets.canCap,
  )
  const ceiling = keyBudgets.ceilings.find((row) => row.scope_id === apiKey.id)
  // Seeded from the key's ceiling once it has been read. Until it has, the
  // field stays off: a draft seeded as "none" would save a second ceiling over
  // the one the key already has.
  const [initialBudget, setInitialBudget] = useState(() =>
    keyBudgets.ceilingsReady ? draftFor(ceiling) : undefined,
  )
  const [budgetDraft, setBudgetDraft] = useState(
    initialBudget ?? EMPTY_KEY_BUDGET,
  )
  // Operator-only: a member edits only their own key, and the note it feeds
  // names other owners.
  const users = useUsers(isDeploymentWide)
  const [keyName, setKeyName] = useState(apiKey.key_name ?? "")
  const [expiresAt, setExpiresAt] = useState(toDatetimeLocal(apiKey.expires_at))
  const [allowedModels, setAllowedModels] = useState<string[] | undefined>(
    apiKey.allowed_models ?? undefined,
  )
  const [allowedTools, setAllowedTools] = useState<string[] | undefined>(
    apiKey.allowed_tools ?? undefined,
  )
  const [toolScopeValid, setToolScopeValid] = useState(true)
  const [excludeFromBudget, setExcludeFromBudget] = useState(
    apiKey.exclude_from_budget,
  )
  const [serviceKey, setServiceKey] = useState(() => serviceKeyChoice(apiKey))
  const [keyScopes, setKeyScopes] = useState<KeyScope[]>(apiKey.scopes)
  const [rejectUserMismatch, setRejectUserMismatch] =
    useState<UserMismatchChoice>(
      userMismatchChoice(apiKey.reject_user_mismatch),
    )
  const [scopeValid, setScopeValid] = useState(true)
  // The operator's form shows the service key fields for every key; an
  // organization's form only for the organization's own service keys, and
  // there whether it is one is fixed. An operator's service key a member holds
  // is the operator's to change.
  const showsServiceKey = isDeploymentWide || apiKey.is_organization_service_key
  const serviceKeyScope = apiKey.is_organization_service_key
    ? "organization"
    : "deployment"
  const serviceKeyReady = useServiceKeyReady(
    showsServiceKey ? serviceKey : NO_SERVICE_KEY,
    serviceKeyScope,
  )
  const isServiceKey = showsServiceKey
    ? serviceKey.isServiceKey
    : apiKey.is_service_key
  const scopes = heldScopes(keyScopes, isServiceKey)
  const { isDirty, reset: reseed } = useDirtySnapshot({
    keyName,
    expiresAt,
    allowedModels,
    allowedTools,
    excludeFromBudget,
    serviceKey,
    keyScopes,
    rejectUserMismatch,
    scopeValid,
    toolScopeValid,
    budgetDraft,
  })
  // Ceilings that land after the form opened are part of its seed, not an edit.
  // The rest of the seed is the key as it was, so anything already typed still
  // reads as unsaved.
  if (initialBudget === undefined && keyBudgets.ceilingsReady) {
    const seeded = draftFor(ceiling)
    setInitialBudget(seeded)
    setBudgetDraft(seeded)
    reseed({
      keyName: apiKey.key_name ?? "",
      expiresAt: toDatetimeLocal(apiKey.expires_at),
      allowedModels: apiKey.allowed_models ?? undefined,
      allowedTools: apiKey.allowed_tools ?? undefined,
      excludeFromBudget: apiKey.exclude_from_budget,
      serviceKey: serviceKeyChoice(apiKey),
      keyScopes: apiKey.scopes,
      rejectUserMismatch: userMismatchChoice(apiKey.reject_user_mismatch),
      scopeValid: true,
      toolScopeValid: true,
      budgetDraft: seeded,
    })
  }
  const showBudget = keyBudgets.canCap && initialBudget !== undefined
  // An exempt key is never gated, so its cap is left as it is.
  const budgetChanged =
    showBudget &&
    !excludeFromBudget &&
    budgetDraft.choice !== initialBudget?.choice
  const isBlocked =
    scopes.length === 0 ||
    !scopeValid ||
    !toolScopeValid ||
    !serviceKeyReady ||
    (budgetChanged && keyBudgetIncomplete(budgetDraft))
  const isSaving = update.isPending || keyBudgets.isApplying

  const submit = () => {
    if (isSaving || isBlocked) return
    const shared = {
      key_name: keyName.trim() || null,
      expires_at: expiresAt ? new Date(expiresAt).toISOString() : null,
      allowed_models: allowedModels ?? null,
      allowed_tools: allowedTools ?? null,
      scopes,
      reject_user_mismatch: storedUserMismatch(rejectUserMismatch),
    }
    // The member surface has no budget exemption to send.
    const body: UpdateKeyRequest | UpdateOwnKeyRequest = isDeploymentWide
      ? {
          ...shared,
          exclude_from_budget: excludeFromBudget,
          ...serviceKeyBody(serviceKey),
        }
      : apiKey.is_organization_service_key
        ? { ...shared, ...serviceKeyBody(serviceKey) }
        : shared
    update.mutate(
      { id: apiKey.id, body },
      {
        onSuccess: () => {
          if (!budgetChanged) return onClose()
          // A refusal stays on the form, reported through `applyError`.
          keyBudgets
            .change({
              keyId: apiKey.id,
              keyLabel: keyName.trim() || apiKey.id,
              draft: budgetDraft,
              current: ceiling,
            })
            .then(onClose, () => {})
        },
      },
    )
  }

  return (
    <FormDialog
      isOpen={isOpen}
      onOpenChange={(open) => {
        if (!open) onClose()
      }}
      size="lg"
      title="Edit API key"
      description={<code>{apiKey.key_name ?? apiKey.id}</code>}
      submitLabel="Save"
      onSubmit={submit}
      isPending={isSaving}
      isSubmitDisabled={isBlocked}
      isDirty={isDirty}
      error={update.error ?? keyBudgets.applyError}
    >
      <Field
        label="Name"
        value={keyName}
        onChange={setKeyName}
        placeholder="ci-bot"
        autoFocus
        shouldReserveMessage={false}
      />
      <ExpiryFields
        label="Expires"
        value={expiresAt}
        onChange={setExpiresAt}
        description="Blank clears the expiry."
      />
      {showBudget ? (
        <KeyBudgetField
          value={budgetDraft}
          onChange={setBudgetDraft}
          budgets={budgetChoices.budgets}
          isLoading={budgetChoices.isLoading}
          isExempt={excludeFromBudget}
          current={
            ceiling
              ? {
                  budgetId: ceiling.budget_id,
                  label: ceilingLabel(ceiling, keyBudgets.budgets),
                }
              : undefined
          }
        />
      ) : null}
      {isDeploymentWide && apiKey.user_id ? (
        <OwnerAccessNote userId={apiKey.user_id} users={users.data ?? []} />
      ) : null}
      <ModelScopeControl
        workspaceId={apiKey.workspace_id ?? undefined}
        title="Restrict this key’s models"
        description={
          isDeploymentWide
            ? "This key inherits its owner’s access by default. Narrow it to a subset here; it can never exceed the owner’s allowed models."
            : "This key inherits your model access by default. Narrow it to a subset here; it can never exceed your allowed models."
        }
        anyLabel={
          isDeploymentWide ? "Inherit owner access" : "Inherit your access"
        }
        initial={apiKey.allowed_models ?? undefined}
        onChange={(value, isValid) => {
          setAllowedModels(value)
          setScopeValid(isValid)
        }}
      />
      <ToolScopeControl
        title="Restrict this key’s search and fetch tools"
        description={
          isDeploymentWide
            ? "Which search and fetch tools this key may run on /v1/search and /v1/fetch. Left on Inherit, it takes its owner’s list. With no list on either, it follows its model access, or its owner’s when it has none: unrestricted runs every tool, restricted only a search tool named there as provider:tool (deprecated). It can never exceed its owner’s."
            : "Which search and fetch tools this key may run on /v1/search and /v1/fetch. Left on Inherit, it takes your list. With no list on either, it follows its model access, or yours when it has none: unrestricted runs every tool, restricted only a search tool named there as provider:tool (deprecated). It can never exceed yours."
        }
        inheritLabel={
          isDeploymentWide ? "Inherit owner’s tools" : "Inherit your tools"
        }
        initial={apiKey.allowed_tools ?? undefined}
        onChange={(value, isValid) => {
          setAllowedTools(value)
          setToolScopeValid(isValid)
        }}
      />
      <KeyScopeFields
        value={keyScopes}
        onChange={setKeyScopes}
        isServiceKey={isServiceKey}
      />
      <KeyMcpAccess workspaceId={apiKey.workspace_id} apiKeyId={apiKey.id} />
      {/* Operator-only: a member editing their own key must not be able to
          take it out of their own budget. */}
      {isDeploymentWide ? (
        <BudgetExemptToggle
          checked={excludeFromBudget}
          onChange={setExcludeFromBudget}
        />
      ) : null}
      {showsServiceKey ? (
        <ServiceKeyFields
          value={serviceKey}
          onChange={setServiceKey}
          scope={serviceKeyScope}
          isLocked={!isDeploymentWide}
        />
      ) : null}
      <UserMismatchPicker
        value={rejectUserMismatch}
        onChange={setRejectUserMismatch}
        description={
          <>
            What happens when a request on this key names a different{" "}
            <code>user</code> than its owner. Accept it for clients that send
            telemetry there rather than an identity, such as Claude Code. Spend
            counts toward this key’s owner either way.
          </>
        }
      />
    </FormDialog>
  )
}
