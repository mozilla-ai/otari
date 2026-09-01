import { Button, Card, Chip, Modal } from "@heroui/react"
import { Link } from "@tanstack/react-router"
import {
  type ReactNode,
  type RefObject,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react"
import type {
  ApiKey,
  CreateKeyRequest,
  CreateKeyResponse,
  CreateOwnKeyRequest,
  UpdateKeyRequest,
  UpdateOwnKeyRequest,
  User,
} from "@/client"
import {
  accessLabel,
  ModelScopeControl,
} from "@/features/models/ModelScopeControl"
import { useMemberAttributionLabels } from "@/features/organization/attribution"
import { UserComboBox } from "@/features/users/UserComboBox"
import {
  useCreateKey,
  useDeleteKey,
  useKeys,
  useKeysScope,
  useRotateKey,
  useUpdateKey,
  useUsers,
} from "@/shared/api/hooks"
import { BulkActionBar } from "@/shared/components/BulkActionBar"
import { ConfirmDialog } from "@/shared/components/ConfirmDialog"
import { DataTable, type DataTableColumn } from "@/shared/components/DataTable"
import { Field } from "@/shared/components/Field"
import { MissingGatewayAddressNotice } from "@/shared/components/MissingGatewayAddressNotice"
import {
  Checkbox,
  CopyField,
  EmptyState,
  ErrorBanner,
  FilterSelect,
  InfoBanner,
  PageHeader,
} from "@/shared/components/ui"
import { formatDate } from "@/shared/helpers/format"
import {
  buildCurlSnippet,
  buildPythonSnippet,
  resolveSnippetBaseUrl,
  SNIPPET_MODEL_PLACEHOLDER,
} from "@/shared/helpers/requestSnippets"
import {
  resolveSelectedIds,
  useTableSelection,
} from "@/shared/helpers/tableSelection"
import { useSelectedWorkspace } from "@/shared/hooks/SelectedWorkspace"
import { useConfirmationFocus } from "@/shared/hooks/useConfirmationFocus"
import { useDeployment } from "@/shared/hooks/useDeployment"

// ---------- helpers ----------

function relative(iso: string | null): string | null {
  if (!iso) return null
  const then = new Date(iso).getTime()
  if (Number.isNaN(then)) return null
  const diffSec = Math.round((then - Date.now()) / 1000)
  const abs = Math.abs(diffSec)
  const units: [Intl.RelativeTimeFormatUnit, number][] = [
    ["day", 86_400],
    ["hour", 3_600],
    ["minute", 60],
  ]
  const rtf = new Intl.RelativeTimeFormat(undefined, { numeric: "auto" })
  for (const [unit, sec] of units) {
    if (abs >= sec) return rtf.format(Math.round(diffSec / sec), unit)
  }
  return rtf.format(diffSec, "second")
}

function isExpired(key: ApiKey): boolean {
  if (!key.expires_at) return false
  const t = new Date(key.expires_at).getTime()
  return !Number.isNaN(t) && t < Date.now()
}

// datetime-local wants "YYYY-MM-DDTHH:mm" in local time; build it from an ISO value.
function toDatetimeLocal(iso: string | null): string {
  if (!iso) return ""
  const d = new Date(iso)
  if (Number.isNaN(d.getTime())) return ""
  const pad = (n: number) => String(n).padStart(2, "0")
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}T${pad(d.getHours())}:${pad(d.getMinutes())}`
}

const isVirtualUser = (userId: string | null): boolean =>
  (userId ?? "").startsWith("apikey-")

const label = (k: ApiKey): string => k.key_name ?? k.id

// Stable row-key getter so DataTable's per-row cache holds across re-renders.
const getKeyRowKey = (k: ApiKey): string => k.id

// ---------- one-time reveal ----------

// The highest-stakes moment: the plaintext key is shown once, so the dialog
// refuses a backdrop press and Esc and closes only on an explicit "I've saved
// this key". `ConfirmDialog` passes the same two props the other way round,
// because dismissing a confirm costs nothing and dismissing this loses the key.
// Doubles as an activation moment: a copy-paste first call.
function RevealSecretModal({
  title,
  result,
  returnFocusRef,
  onClose,
}: {
  title: string
  result: CreateKeyResponse
  returnFocusRef: RefObject<HTMLButtonElement | null>
  onClose: () => void
}) {
  const secretRef = useRef<HTMLInputElement | HTMLTextAreaElement | null>(null)
  // Where a request from this deployment belongs, which is not always the address
  // that served this page: a hosted control plane serves the dashboard and not
  // the API. Undefined when it has not said where its gateway is (otari#823).
  const baseUrl = resolveSnippetBaseUrl(useDeployment())
  const secret = result.key

  useEffect(() => {
    // Focus the secret so Ctrl/Cmd-C works at once and a stray Enter doesn't land
    // on the close button.
    secretRef.current?.focus()
    secretRef.current?.select()
  }, [])

  useEffect(
    () => () => {
      // The dialog restores focus to its own trigger, which is hidden here
      // because the reveal opens from a mutation result, so the restore has
      // nowhere to land. A frame later it has had its turn: an empty
      // document.body then means nothing caught the focus.
      requestAnimationFrame(() => {
        if (document.activeElement === document.body) {
          returnFocusRef.current?.focus()
        }
      })
    },
    [returnFocusRef],
  )

  // The same two calls the setup guide hands out with its own key; the builders
  // are shared so an operator cannot be shown two dialects of one request.
  // One value rather than two, so a deployment that named no gateway drops both
  // together and the copy under the heading cannot disagree with the fields.
  const snippets = baseUrl
    ? {
        curl: buildCurlSnippet({ baseUrl, apiKey: secret }),
        python: buildPythonSnippet({ baseUrl, apiKey: secret }),
      }
    : undefined

  return (
    <Modal isOpen>
      {/* The reveal opens from a mutation result rather than a press, and HeroUI
          warns when the trigger slot is empty; `WorkspaceSwitcher` fills it the
          same way. */}
      <Modal.Trigger className="hidden">Show the new key</Modal.Trigger>
      <Modal.Backdrop isDismissable={false} isKeyboardDismissDisabled>
        {/* `scroll="outside"` because globals.css pins `.modal__body` to
            `overflow: visible`: with the default the two snippets past the fold
            would be clipped by the dialog with nothing able to scroll them. */}
        <Modal.Container placement="center" scroll="outside">
          <Modal.Dialog>
            <Modal.Header>
              <Modal.Heading className="text-title">{title}</Modal.Heading>
            </Modal.Header>
            <Modal.Body className="flex flex-col gap-4">
              <InfoBanner tone="warning">
                Copy this key now. For security it is shown only once and cannot
                be retrieved later. If you lose it, use Regenerate to issue a
                new secret.
              </InfoBanner>
              <p className="text-xs text-muted">
                Model access: {accessLabel(result.allowed_models).text}.
              </p>
              <CopyField
                label="Secret key"
                value={secret}
                fieldRef={secretRef}
              />
              <div className="flex flex-col gap-2">
                <div>
                  <h3 className="text-title">Make your first call</h3>
                  {snippets === undefined ? (
                    <MissingGatewayAddressNotice />
                  ) : (
                    <p className="text-xs text-muted">
                      Replace <code>{SNIPPET_MODEL_PLACEHOLDER}</code> with a
                      model from the Models page.
                    </p>
                  )}
                </div>
                {snippets !== undefined ? (
                  <>
                    <CopyField label="curl" value={snippets.curl} multiline />
                    <CopyField
                      label="Python (OpenAI SDK)"
                      value={snippets.python}
                      multiline
                    />
                  </>
                ) : null}
              </div>
            </Modal.Body>
            <Modal.Footer>
              <Button variant="primary" onPress={onClose}>
                I&rsquo;ve saved this key
              </Button>
            </Modal.Footer>
          </Modal.Dialog>
        </Modal.Container>
      </Modal.Backdrop>
    </Modal>
  )
}

// ---------- inline confirm (names the target, no modal) ----------

function InlineConfirm({
  trigger,
  message,
  confirmLabel,
  isPending,
  onConfirm,
}: {
  trigger: string
  message: ReactNode
  confirmLabel: string
  isPending?: boolean
  onConfirm: () => void
}) {
  const [armed, setArmed] = useState(false)
  const { triggerRef, confirmRef } = useConfirmationFocus(armed)

  if (!armed) {
    return (
      <Button
        ref={triggerRef}
        size="sm"
        variant="danger-soft"
        onPress={() => setArmed(true)}
      >
        {trigger}
      </Button>
    )
  }

  return (
    <div className="flex flex-col items-end gap-1.5 rounded-lg border border-warning bg-warning-subtle p-2 text-right">
      <span className="max-w-xs text-xs text-warning">{message}</span>
      <span className="inline-flex gap-1">
        <Button
          ref={confirmRef}
          size="sm"
          variant="danger"
          isDisabled={isPending}
          onPress={onConfirm}
        >
          {confirmLabel}
        </Button>
        <Button
          size="sm"
          variant="ghost"
          isDisabled={isPending}
          onPress={() => setArmed(false)}
        >
          Cancel
        </Button>
      </span>
    </div>
  )
}

// ---------- create / edit forms (inline cards, matching ProvidersPage) ----------

// Shows the selected owner's model access so the operator sees the ceiling this
// key narrows within (a key can inherit it or restrict to a subset, never exceed).
function OwnerAccessNote({ userId, users }: { userId: string; users: User[] }) {
  const id = userId.trim()
  if (id === "") {
    return (
      <p className="text-xs text-muted">
        Choose an owner above to see the models this key can inherit.
      </p>
    )
  }
  const owner = users.find((u) => u.user_id === id)
  if (!owner) {
    return (
      <p className="text-xs text-muted">
        New user <code>{id}</code> starts unrestricted, so this key may allow
        any model.
      </p>
    )
  }
  const { text } = accessLabel(owner.allowed_models)
  const entries =
    owner.allowed_models && owner.allowed_models.length > 0
      ? owner.allowed_models.join(", ")
      : null
  return (
    <p className="text-xs text-muted">
      Owner <code>{id}</code> allows{" "}
      <span className="font-medium text-foreground">{text.toLowerCase()}</span>
      {entries ? (
        <>
          {" ("}
          <span className="font-mono">{entries}</span>
          {")"}
        </>
      ) : null}
      . This key inherits that, or narrows within it.
    </p>
  )
}

// Money-adjacent control: a labeled checkbox with the consequence spelled out, so
// flipping it is a deliberate act rather than a bare switch in a table cell.
function BudgetExemptToggle({
  checked,
  onChange,
}: {
  checked: boolean
  onChange: (value: boolean) => void
}) {
  return (
    <div className="flex flex-col gap-0.5 rounded-lg border border-border p-3">
      {/* The consequence sits beside the control rather than inside its label:
          a checkbox whose accessible name is a whole paragraph is read out in
          full on every focus. */}
      <Checkbox isSelected={checked} onChange={onChange}>
        <span className="font-medium text-foreground">Exempt from budget</span>
      </Checkbox>
      <p className="text-xs text-muted">
        Requests on this key are logged with their cost but never counted toward
        the owner&apos;s budget or spend, and never blocked by it.
      </p>
    </div>
  )
}

// Access-control adjacent: a three-way override of the deployment-wide
// reject_user_mismatch, so it is a picker rather than a checkbox. Same shape as
// the budget picker on the budgets page.
function UserMismatchPicker({
  value,
  onChange,
}: {
  value: boolean | null
  onChange: (value: boolean | null) => void
}) {
  const selectId = "key-reject-user-mismatch"
  return (
    <div className="flex flex-col gap-1">
      {/* The label carries markup (`user` as code), which FilterSelect's own
          `label` cannot, so it stays here with `htmlFor` on the trigger and
          `ariaLabel` naming the control for assistive tech. */}
      <label htmlFor={selectId} className="text-sm font-medium text-foreground">
        Mismatched <code>user</code> field
      </label>
      <FilterSelect
        id={selectId}
        ariaLabel="Mismatched user field"
        value={value === null ? "inherit" : value ? "reject" : "accept"}
        onChange={(next) =>
          onChange(next === "inherit" ? null : next === "reject")
        }
        options={[
          { value: "inherit", label: "Use the deployment setting (default)" },
          { value: "reject", label: "Always reject (403)" },
          { value: "accept", label: "Always accept" },
        ]}
      />
      <span className="text-xs text-muted">
        What happens when a request on this key names a different{" "}
        <code>user</code> than its owner. Accept it for clients that send
        telemetry there rather than an identity, such as Claude Code. Spend
        binds to this key&apos;s owner either way.
      </span>
    </div>
  )
}

// `isDeploymentWide` is which key surface this caller acts on (useKeysScope): an
// operator names an owner and may exempt the key from budget; a member's key is
// always their own and always enforced, so neither control is rendered and the
// body sent is the member surface's narrower shape.
function CreateKeyForm({
  isDeploymentWide,
  onClose,
  onCreated,
}: {
  isDeploymentWide: boolean
  onClose: () => void
  onCreated: (result: CreateKeyResponse) => void
}) {
  const create = useCreateKey()
  // `/v1/users` is operator-only; a member's form has no owner picker to feed.
  const users = useUsers(isDeploymentWide)
  const { selected: workspace, isLoading: workspaceLoading } =
    useSelectedWorkspace()
  const [keyName, setKeyName] = useState("")
  const [expiresAt, setExpiresAt] = useState("")
  const memberLabels = useMemberAttributionLabels()
  const [showAdvanced, setShowAdvanced] = useState(false)
  const [userId, setUserId] = useState("")
  const [allowedModels, setAllowedModels] = useState<string[] | null>(null)
  const [excludeFromBudget, setExcludeFromBudget] = useState(false)
  const [rejectUserMismatch, setRejectUserMismatch] = useState<boolean | null>(
    null,
  )
  const [scopeValid, setScopeValid] = useState(true)

  const expiresInPast =
    expiresAt !== "" && new Date(expiresAt).getTime() < Date.now()
  // User-first: an operator's key must name its owner (an existing user or a new
  // id, which the API creates as a named user). This is what keeps the dashboard
  // from minting the anonymous virtual users an omitted id would. A member's key
  // is always their own, so there is nothing to require.
  const ownerMissing = isDeploymentWide && userId.trim() === ""
  // The workspace comes from the organization context, which resolves after the
  // form paints. Submitting before it does would send no workspace and land the
  // key somewhere other than the one the switcher is showing, so the button
  // waits for that read. It waits for the read, not for a workspace: a caller
  // who belongs to none resolves to null and still submits, and what an omitted
  // workspace means is then the server's to say. The operator surface mints into
  // the organization's default; the member surface refuses with a 409 unless the
  // caller belongs to that default, which is the answer this form surfaces
  // rather than pre-empting.
  const workspaceUnresolved = workspaceLoading

  const submit = () => {
    if (create.isPending || !scopeValid || ownerMissing || workspaceUnresolved)
      return
    const shared = {
      key_name: keyName.trim() || null,
      // The workspace the shell is on. A key belongs to exactly one, and it is
      // what every request on that key is billed to, so it is decided here
      // rather than left to the server's default.
      workspace_id: workspace?.workspace_id,
      expires_at: expiresAt ? new Date(expiresAt).toISOString() : null,
      allowed_models: allowedModels,
      reject_user_mismatch: rejectUserMismatch,
    }
    // The member surface derives the owner and refuses a budget exemption, so
    // its body carries neither field rather than sending values it would ignore.
    const body: CreateKeyRequest | CreateOwnKeyRequest = isDeploymentWide
      ? {
          ...shared,
          user_id: userId.trim(),
          exclude_from_budget: excludeFromBudget,
        }
      : shared
    create.mutate(body, {
      onSuccess: (result) => {
        // Capture the secret into the reveal BEFORE closing the form, so a render
        // hiccup can never drop the one-time key.
        onCreated(result)
        onClose()
      },
    })
  }

  return (
    <Card>
      <Card.Content className="flex flex-col gap-4 p-5">
        <h2 className="text-title">Create API key</h2>
        <ErrorBanner error={create.error} />
        <div className="grid gap-4 sm:grid-cols-2">
          <Field
            label="Name"
            value={keyName}
            onChange={setKeyName}
            placeholder="ci-bot"
            autoFocus
            description="A label to recognize this key later."
          />
          <Field
            label="Expires (optional)"
            value={expiresAt}
            onChange={setExpiresAt}
            type="datetime-local"
            description={
              expiresInPast ? (
                <span className="text-danger">
                  That time is in the past; the key would be rejected
                  immediately.
                </span>
              ) : (
                "Leave blank for a key that never expires."
              )
            }
          />
        </div>
        {isDeploymentWide ? (
          <UserComboBox
            value={userId}
            onChange={setUserId}
            users={users.data ?? []}
            memberLabels={memberLabels}
          />
        ) : (
          <p className="text-xs text-muted">
            This key is yours: requests on it are billed to you and count
            against your budget.
          </p>
        )}
        <button
          type="button"
          className="self-start text-xs font-medium text-link hover:text-link-hover"
          onClick={() => setShowAdvanced((v) => !v)}
        >
          {showAdvanced ? "Hide advanced" : "Advanced"}
        </button>
        {showAdvanced ? (
          <div className="flex flex-col gap-4 rounded-lg border border-border p-4">
            {isDeploymentWide ? (
              <OwnerAccessNote userId={userId} users={users.data ?? []} />
            ) : null}
            <ModelScopeControl
              title="Restrict this key's models"
              description={
                isDeploymentWide
                  ? "By default this key inherits its owner's access. Optionally narrow it to a subset; a key can never exceed its owner's allowed models."
                  : "By default this key inherits your model access. Optionally narrow it to a subset; a key can never exceed your allowed models."
              }
              anyLabel={
                isDeploymentWide
                  ? "Inherit owner access"
                  : "Inherit your access"
              }
              initial={null}
              onChange={(value, valid) => {
                setAllowedModels(value)
                setScopeValid(valid)
              }}
            />
            {isDeploymentWide ? (
              <BudgetExemptToggle
                checked={excludeFromBudget}
                onChange={setExcludeFromBudget}
              />
            ) : null}
            <UserMismatchPicker
              value={rejectUserMismatch}
              onChange={setRejectUserMismatch}
            />
          </div>
        ) : null}
        <div className="flex gap-2">
          <Button
            variant="primary"
            isDisabled={
              create.isPending ||
              !scopeValid ||
              ownerMissing ||
              workspaceUnresolved
            }
            onPress={submit}
          >
            {create.isPending ? "Creating…" : "Create key"}
          </Button>
          <Button variant="ghost" onPress={onClose}>
            Cancel
          </Button>
        </div>
      </Card.Content>
    </Card>
  )
}

function EditKeyForm({
  isDeploymentWide,
  apiKey,
  onClose,
}: {
  isDeploymentWide: boolean
  apiKey: ApiKey
  onClose: () => void
}) {
  const update = useUpdateKey()
  // Operator-only, like the create form's picker; a member edits only their own
  // key and the note it feeds names other owners.
  const users = useUsers(isDeploymentWide)
  const [keyName, setKeyName] = useState(apiKey.key_name ?? "")
  const [expiresAt, setExpiresAt] = useState(toDatetimeLocal(apiKey.expires_at))
  const [allowedModels, setAllowedModels] = useState<string[] | null>(
    apiKey.allowed_models,
  )
  const [excludeFromBudget, setExcludeFromBudget] = useState(
    apiKey.exclude_from_budget,
  )
  const [rejectUserMismatch, setRejectUserMismatch] = useState<boolean | null>(
    apiKey.reject_user_mismatch,
  )
  const [scopeValid, setScopeValid] = useState(true)

  const submit = () => {
    if (update.isPending || !scopeValid) return
    const shared = {
      key_name: keyName.trim() || null,
      expires_at: expiresAt ? new Date(expiresAt).toISOString() : null,
      allowed_models: allowedModels,
      reject_user_mismatch: rejectUserMismatch,
    }
    // The member surface has no budget exemption to send (see CreateKeyForm).
    const body: UpdateKeyRequest | UpdateOwnKeyRequest = isDeploymentWide
      ? { ...shared, exclude_from_budget: excludeFromBudget }
      : shared
    update.mutate({ id: apiKey.id, body }, { onSuccess: onClose })
  }

  return (
    <Card>
      <Card.Content className="flex flex-col gap-4 p-5">
        <h2 className="text-title">
          Edit <code>{apiKey.key_name ?? apiKey.id}</code>
        </h2>
        <ErrorBanner error={update.error} />
        <div className="grid gap-4 sm:grid-cols-2">
          <Field
            label="Name"
            value={keyName}
            onChange={setKeyName}
            placeholder="ci-bot"
          />
          <Field
            label="Expires"
            value={expiresAt}
            onChange={setExpiresAt}
            type="datetime-local"
            description="Blank clears the expiry."
          />
        </div>
        {isDeploymentWide && apiKey.user_id ? (
          <OwnerAccessNote userId={apiKey.user_id} users={users.data ?? []} />
        ) : null}
        <ModelScopeControl
          title="Restrict this key's models"
          description={
            isDeploymentWide
              ? "This key inherits its owner's access by default. Narrow it to a subset here; it can never exceed the owner's allowed models."
              : "This key inherits your model access by default. Narrow it to a subset here; it can never exceed your allowed models."
          }
          anyLabel={
            isDeploymentWide ? "Inherit owner access" : "Inherit your access"
          }
          initial={apiKey.allowed_models}
          onChange={(value, valid) => {
            setAllowedModels(value)
            setScopeValid(valid)
          }}
        />
        {isDeploymentWide ? (
          <BudgetExemptToggle
            checked={excludeFromBudget}
            onChange={setExcludeFromBudget}
          />
        ) : null}
        <UserMismatchPicker
          value={rejectUserMismatch}
          onChange={setRejectUserMismatch}
        />
        <div className="flex gap-2">
          <Button
            variant="primary"
            isDisabled={update.isPending || !scopeValid}
            onPress={submit}
          >
            {update.isPending ? "Saving…" : "Save changes"}
          </Button>
          <Button variant="ghost" onPress={onClose}>
            Cancel
          </Button>
        </div>
      </Card.Content>
    </Card>
  )
}

// ---------- status + rows ----------

function StatusChip({ apiKey }: { apiKey: ApiKey }) {
  if (!apiKey.is_active) {
    return (
      <Chip size="sm" color="default">
        Disabled
      </Chip>
    )
  }
  if (isExpired(apiKey)) {
    return (
      <Chip size="sm" color="warning">
        Expired
      </Chip>
    )
  }
  return (
    <Chip size="sm" color="accent">
      Active
    </Chip>
  )
}

function AccessChip({ allowed }: { allowed: string[] | null }) {
  const { text, tone } = accessLabel(allowed)
  const cls =
    tone === "danger"
      ? "text-danger font-medium"
      : tone === "muted"
        ? "text-muted"
        : "text-accent font-medium"
  // Surface the exact entries on hover; the count would mislead (a wildcard is many).
  const title = allowed && allowed.length > 0 ? allowed.join(", ") : undefined
  return (
    <span className={`text-xs ${cls}`} title={title}>
      {text}
    </span>
  )
}

export function KeysPage() {
  // Scoped to the workspace the switcher is on: a key belongs to exactly one,
  // and this page is in the workspace context.
  const { selected: workspace } = useSelectedWorkspace()
  // Which surface the caller acts on decides the page's voice as well as the
  // endpoints: an operator manages every key in the organization, a member
  // manages their own (otari-ai#1941).
  const scope = useKeysScope()
  const isDeploymentWide = scope.isDeploymentWide
  const keys = useKeys(workspace?.workspace_id)
  const updateKey = useUpdateKey()
  const rotateKey = useRotateKey()
  const deleteKey = useDeleteKey()
  const memberLabels = useMemberAttributionLabels()

  const [addOpen, setAddOpen] = useState(false)
  const [editing, setEditing] = useState<string | null>(null)
  const [revealed, setRevealed] = useState<{
    title: string
    result: CreateKeyResponse
  } | null>(null)
  // Where focus lands when the reveal closes. The control that opened it is
  // either gone (the create form) or behind a confirm that has been consumed
  // (a regenerate), so the page's own primary action is the stable landing
  // spot nearest to where the operator was.
  const createButtonRef = useRef<HTMLButtonElement>(null)

  const rows = keys.data ?? []
  // An unresolved scope is a loading state: which surface the page reads, and
  // which affordances it draws, are both undecided until the organization
  // context answers, and a disabled query reports `isLoading` false.
  const loading = !scope.isReady || keys.isLoading
  const editingKey = rows.find((k) => k.id === editing) ?? null
  const showOnboarding = !loading && rows.length === 0 && !addOpen
  const selection = useTableSelection()
  const [bulkDeleteOpen, setBulkDeleteOpen] = useState(false)
  const [bulkError, setBulkError] = useState<unknown>(undefined)
  const [bulkPending, setBulkPending] = useState(false)

  const selectableKeys = rows.map((k) => k.id)
  const selectedIds = resolveSelectedIds(selection.selectedKeys, selectableKeys)
  const selectedKeys = rows.filter((k) => selectedIds.includes(k.id))

  // Stable handlers (mutate fns are referentially stable in TanStack Query) so
  // the memoized columns below survive unrelated re-renders.
  const setActive = useCallback(
    (k: ApiKey, active: boolean) =>
      updateKey.mutate({ id: k.id, body: { is_active: active } }),
    [updateKey.mutate],
  )

  const regenerate = useCallback(
    (k: ApiKey) =>
      rotateKey.mutate(k.id, {
        onSuccess: (result) =>
          setRevealed({ title: `New secret for ${label(k)}`, result }),
      }),
    [rotateKey.mutate],
  )

  const runBulk = async (
    targets: ApiKey[],
    action: (k: ApiKey) => Promise<unknown>,
    onDone?: () => void,
  ) => {
    setBulkPending(true)
    setBulkError(undefined)
    try {
      for (const key of targets) {
        await action(key)
      }
      selection.clear()
      onDone?.()
    } catch (error) {
      setBulkError(error)
    } finally {
      setBulkPending(false)
    }
  }

  // Memoized on the values the cells actually read (mutation pending flags and
  // the stable handlers) so DataTable's per-row cache holds across selection
  // clicks; see the DataTable docstring.
  const columns = useMemo<DataTableColumn<ApiKey>[]>(
    () => [
      {
        id: "name",
        header: "Name",
        isRowHeader: true,
        cell: (k) => (
          <div className="flex flex-col gap-0.5">
            <span className="font-medium text-foreground">
              {k.key_name ?? <span className="text-muted">(unnamed)</span>}
            </span>
            <div className="flex flex-wrap items-center gap-1">
              <AccessChip allowed={k.allowed_models} />
              {k.exclude_from_budget ? (
                <span
                  className="inline-flex items-center rounded-full border border-border bg-primary-subtle px-2 py-0.5 text-xs font-medium text-primary-subtle-foreground"
                  title="Requests on this key are logged with cost but never counted toward budget"
                >
                  Budget-exempt
                </span>
              ) : null}
              {k.reject_user_mismatch === null ? null : (
                <span
                  className="inline-flex items-center rounded-full border border-border bg-primary-subtle px-2 py-0.5 text-xs font-medium text-primary-subtle-foreground"
                  title={
                    k.reject_user_mismatch
                      ? "This key always rejects a request naming a different user, whatever the deployment setting says"
                      : "This key always accepts a request naming a different user; spend still binds to its owner"
                  }
                >
                  {k.reject_user_mismatch ? "Strict user" : "Lenient user"}
                </span>
              )}
            </div>
          </div>
        ),
      },
      {
        id: "status",
        header: "Status",
        cell: (k) => <StatusChip apiKey={k} />,
      },
      // Every key on a member's page is their own, so an Owner column there
      // would repeat one name down the table.
      ...(isDeploymentWide
        ? [
            {
              id: "owner",
              header: "Owner",
              // A member is named; anything else keeps the raw id, which for a
              // hand-made owner like `ci-bot` is already the readable form. The id
              // stays in the title so the value actually sent on a request is still
              // recoverable from this column.
              cell: (k: ApiKey) => {
                if (isVirtualUser(k.user_id)) {
                  return (
                    <Chip size="sm" color="default">
                      virtual
                    </Chip>
                  )
                }
                const member = k.user_id
                  ? memberLabels.get(k.user_id)
                  : undefined
                if (member) {
                  return (
                    <span
                      className="text-sm text-foreground"
                      title={k.user_id ?? ""}
                    >
                      {member}
                    </span>
                  )
                }
                return (
                  <code className="text-xs text-muted">{k.user_id ?? "—"}</code>
                )
              },
            } satisfies DataTableColumn<ApiKey>,
          ]
        : []),
      {
        id: "key",
        header: "Key",
        cell: (k) => (
          <code className="text-xs text-muted">
            {k.key_prefix ? `${k.key_prefix}…` : "—"}
          </code>
        ),
      },
      {
        id: "created",
        header: "Created",
        cell: (k) => (
          <span className="text-muted">{formatDate(k.created_at)}</span>
        ),
      },
      {
        id: "last_used",
        header: "Last used",
        cell: (k) => (
          <span className="text-muted">
            {relative(k.last_used_at) ?? "never"}
          </span>
        ),
      },
      {
        id: "expires",
        header: "Expires",
        cell: (k) => (
          <span
            className="text-muted"
            title={
              k.expires_at ? new Date(k.expires_at).toLocaleString() : undefined
            }
          >
            {k.expires_at ? formatDate(k.expires_at) : "never"}
          </span>
        ),
      },
      {
        id: "actions",
        header: "Actions",
        align: "end",
        cell: (k) => (
          <div className="flex items-center justify-end gap-1.5">
            <Button
              size="sm"
              variant="outline"
              isDisabled={updateKey.isPending}
              onPress={() => setActive(k, !k.is_active)}
            >
              {k.is_active ? "Disable" : "Enable"}
            </Button>
            <Button
              size="sm"
              variant="ghost"
              onPress={() => {
                setAddOpen(false)
                setEditing(k.id)
              }}
            >
              Edit
            </Button>
            <InlineConfirm
              trigger="Regenerate"
              confirmLabel="Regenerate"
              isPending={rotateKey.isPending}
              message={
                <>
                  Regenerate the secret for <strong>{label(k)}</strong>? The
                  current secret stops working immediately, with no grace
                  period.
                </>
              }
              onConfirm={() => regenerate(k)}
            />
            {/* Permanent delete is only offered once a key is disabled, so a live
              caller can't be broken (and its audit trail erased) in one click. */}
            {k.is_active ? null : (
              <InlineConfirm
                trigger="Delete"
                confirmLabel="Delete permanently"
                isPending={deleteKey.isPending}
                message={
                  <>
                    Permanently delete <strong>{label(k)}</strong>? This removes
                    the key and unlinks its usage history. Cannot be undone.
                  </>
                }
                onConfirm={() => deleteKey.mutate(k.id)}
              />
            )}
          </div>
        ),
      },
    ],
    [
      updateKey.isPending,
      rotateKey.isPending,
      deleteKey.isPending,
      deleteKey.mutate,
      setActive,
      regenerate,
      memberLabels,
      isDeploymentWide,
    ],
  )

  // Bulk delete targets only already-disabled keys, mirroring the per-row rule
  // that a live key must be disabled before it can be permanently deleted.
  const deletableSelected = selectedKeys.filter((k) => !k.is_active)

  return (
    <div className="flex flex-col gap-6">
      <PageHeader
        title="API keys"
        description={
          isDeploymentWide
            ? "Issue and revoke the keys that authenticate callers to this gateway. Secrets are shown once at creation."
            : "Create and manage your own keys for calling this gateway. Secrets are shown once at creation."
        }
        action={
          addOpen ? null : (
            <Button
              ref={createButtonRef}
              variant="primary"
              onPress={() => {
                setEditing(null)
                setAddOpen(true)
              }}
            >
              Create key
            </Button>
          )
        }
      />

      <ErrorBanner
        error={
          keys.error ?? updateKey.error ?? rotateKey.error ?? deleteKey.error
        }
      />

      {/* A key's owner and its spending limit are both set elsewhere now, on the
          organization rail. This page is where an operator arrives looking for
          them, so it says where they went rather than leaving the sidebar to be
          re-learned. A member's keys are their own and both destinations refuse
          them, so their page states the ownership rule instead of pointing at
          pages they cannot open. */}
      {isDeploymentWide ? (
        <p className="text-sm text-muted">
          A key spends against its owner's budget. Owners live under{" "}
          <Link
            to="/organization/members"
            className="font-medium text-link hover:text-link-hover"
          >
            Organization → Members &amp; roles
          </Link>
          , and their limits under{" "}
          <Link
            to="/budgets"
            className="font-medium text-link hover:text-link-hover"
          >
            Spend &amp; budgets
          </Link>
          .
        </p>
      ) : (
        <p className="text-sm text-muted">
          These are your keys: requests on them are billed to you and spend
          against your budget.
        </p>
      )}

      {showOnboarding ? (
        <EmptyState
          title="No API keys yet"
          description="An API key authenticates callers to this gateway. Create one to make your first request; the secret is shown once, so keep it somewhere safe."
          actionLabel="Create your first key"
          onAction={() => {
            setEditing(null)
            setAddOpen(true)
          }}
        />
      ) : null}

      {addOpen ? (
        <CreateKeyForm
          isDeploymentWide={isDeploymentWide}
          onClose={() => setAddOpen(false)}
          onCreated={(result) =>
            setRevealed({ title: "API key created", result })
          }
        />
      ) : null}
      {/* Key on the row id so switching which key is edited remounts the form:
          its fields seed from `apiKey` via useState (mount-only), so without this
          a second Edit would keep the first key's values and PATCH the wrong row. */}
      {editingKey ? (
        <EditKeyForm
          key={editingKey.id}
          isDeploymentWide={isDeploymentWide}
          apiKey={editingKey}
          onClose={() => setEditing(null)}
        />
      ) : null}

      {selectedIds.length > 0 ? (
        <BulkActionBar
          selectedCount={selectedIds.length}
          allMatching={false}
          matchingTotal={null}
          canSelectAllMatching={false}
          onSelectAllMatching={() => {}}
          onClear={selection.clear}
        >
          <Button
            size="sm"
            variant="outline"
            isDisabled={bulkPending}
            onPress={() =>
              void runBulk(selectedKeys, (k) =>
                updateKey.mutateAsync({ id: k.id, body: { is_active: false } }),
              )
            }
          >
            Disable
          </Button>
          {/* A member cannot exempt their own spend from enforcement, so the
              bulk form of the toggle is operator-only like the per-key one. */}
          {isDeploymentWide ? (
            <Button
              size="sm"
              variant="outline"
              isDisabled={bulkPending}
              onPress={() =>
                void runBulk(selectedKeys, (k) =>
                  updateKey.mutateAsync({
                    id: k.id,
                    body: { exclude_from_budget: true },
                  }),
                )
              }
            >
              Budget-exempt
            </Button>
          ) : null}
          <Button
            size="sm"
            variant="danger"
            isDisabled={deletableSelected.length === 0}
            onPress={() => setBulkDeleteOpen(true)}
          >
            Delete
          </Button>
        </BulkActionBar>
      ) : null}

      {/* Suppress the table (and its own empty message) while the onboarding
          panel owns the empty state, so a fresh gateway shows one call to action,
          not a panel stacked over a redundant "no rows" table. */}
      {showOnboarding ? null : (
        <DataTable
          ariaLabel="API keys"
          columns={columns}
          rows={rows}
          getRowKey={getKeyRowKey}
          isLoading={loading}
          emptyContent="No API keys yet. Create one to authenticate a caller."
          selectionMode="multiple"
          selectedKeys={selection.selectedKeys}
          onSelectionChange={selection.onSelectionChange}
        />
      )}

      <ConfirmDialog
        isOpen={bulkDeleteOpen}
        onOpenChange={setBulkDeleteOpen}
        heading="Delete API keys"
        body={`Permanently delete ${deletableSelected.length} disabled ${
          deletableSelected.length === 1 ? "key" : "keys"
        }? This removes them and unlinks their usage history. Cannot be undone. Active keys in the selection are skipped; disable them first.`}
        confirmLabel="Delete permanently"
        isPending={bulkPending}
        error={bulkError}
        onConfirm={() =>
          void runBulk(
            deletableSelected,
            (k) => deleteKey.mutateAsync(k.id),
            () => setBulkDeleteOpen(false),
          )
        }
      />

      {revealed ? (
        <RevealSecretModal
          title={revealed.title}
          result={revealed.result}
          returnFocusRef={createButtonRef}
          onClose={() => {
            setRevealed(null)
            // Drop the one-time secret from mutation state so reopening Create/Regenerate
            // never flashes the previous key.
            rotateKey.reset()
          }}
        />
      ) : null}
    </div>
  )
}
