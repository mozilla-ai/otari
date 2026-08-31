import { Button } from "@heroui/react"
import { Link } from "@tanstack/react-router"
import {
  type ReactNode,
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
  BLEED_INSET,
  Dot,
  FULL_BLEED,
  TableScrollFrame,
} from "@/shared/components/surface"
import {
  CopyField,
  EmptyState,
  ErrorBanner,
  FilterSelect,
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

/**
 * The highest-stakes moment: the plaintext key, shown once.
 *
 * A strip between the page header and the table rather than a modal. The modal
 * existed to make the acknowledgement unavoidable, and it bought that with a
 * focus trap, a swallowed Esc and a backdrop that ignored clicks, all of which
 * are a dialog fighting its own conventions. A strip that stays until it is
 * acknowledged does the same job without pretending to be dismissible: there is
 * one control and it is the acknowledgement. The page scrolls to it on reveal,
 * so it is never announced somewhere the operator is not looking.
 */
function RevealSecretStrip({
  title,
  result,
  onClose,
}: {
  title: string
  result: CreateKeyResponse
  onClose: () => void
}) {
  const secretRef = useRef<HTMLInputElement | HTMLTextAreaElement | null>(null)
  // Where a request from this deployment belongs, which is not always the address
  // that served this page: a hosted control plane serves the dashboard and not
  // the API. Undefined when it has not said where its gateway is (otari#823).
  const baseUrl = resolveSnippetBaseUrl(useDeployment())
  const secret = result.key

  useEffect(() => {
    // The strip is above the fold only if the page is at the top, and a reveal
    // can arrive after a scroll. Move there first, then take focus, so the
    // secret is selected somewhere the operator can see it.
    window.scrollTo({ top: 0 })
    secretRef.current?.focus()
    secretRef.current?.select()
  }, [])

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
    <section
      // `alert` rather than `status`: this is the one message on the page that
      // is gone forever if it is missed.
      role="alert"
      aria-labelledby="reveal-title"
      className={`${FULL_BLEED} ${BLEED_INSET} flex flex-col gap-4 border-y border-border py-5`}
    >
      <div className="flex items-start gap-3">
        <Dot className="mt-2 bg-danger" />
        <div className="min-w-0">
          <h2 id="reveal-title" className="text-title">
            {title}
          </h2>
          <p className="mt-1 text-sm text-muted">
            Copy this key now. For security it is shown only once and cannot be
            retrieved later. If you lose it, use Regenerate to issue a new
            secret. Model access: {accessLabel(result.allowed_models).text}.
          </p>
        </div>
      </div>
      <CopyField label="Secret key" value={secret} fieldRef={secretRef} />
      <div className="flex flex-col gap-2">
        <div>
          <div className="text-sm font-medium text-foreground">
            Make your first call
          </div>
          {snippets === undefined ? (
            <MissingGatewayAddressNotice />
          ) : (
            <p className="text-xs text-muted">
              Replace <code>{SNIPPET_MODEL_PLACEHOLDER}</code> with a model from
              the Models page.
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
      <div className="flex justify-end border-t border-border pt-4">
        <Button variant="primary" onPress={onClose}>
          I&rsquo;ve saved this key
        </Button>
      </div>
    </section>
  )
}

// ---------- row actions and the armed strip ----------

/**
 * A row action: 13px of muted text, no button, no border, no fill. A row has
 * four of these and boxing each one turned the last lane into a control panel;
 * the actions are the only interactive things in a row of static values, so
 * they need no shape to be found. Danger ink is reserved for the armed state,
 * which is the only moment one of them is about to do something irreversible.
 */
function RowAction({
  onPress,
  isDanger,
  isDisabled,
  children,
}: {
  onPress: () => void
  isDanger?: boolean
  isDisabled?: boolean
  children: ReactNode
}) {
  return (
    <button
      type="button"
      disabled={isDisabled}
      onClick={onPress}
      className={`text-[13px] transition-colors motion-reduce:transition-none disabled:opacity-50 ${
        isDanger ? "text-danger" : "text-muted hover:text-foreground"
      }`}
    >
      {children}
    </button>
  )
}

/**
 * The confirmation, as a strip spanning the whole table directly under the row
 * it is about, rather than as a panel inside one cell.
 *
 * It reads as part of that row: the row's own bottom rule is suppressed in CSS
 * so the two are one unit, and the strip carries its own rules top and bottom.
 * The message keeps its `<strong>` on the key's name, which is the one thing
 * that must not be misread when the next click is irreversible.
 */
function ArmedStrip({
  message,
  confirmLabel,
  isPending,
  onConfirm,
  onCancel,
}: {
  message: ReactNode
  confirmLabel: string
  isPending?: boolean
  onConfirm: () => void
  onCancel: () => void
}) {
  return (
    <div className="flex flex-wrap items-center gap-4 bg-background px-8 py-3.5">
      <Dot className="bg-danger" />
      <span className="min-w-0 flex-1 text-sm text-foreground">{message}</span>
      <div className="flex shrink-0 items-center gap-4">
        <RowAction onPress={onCancel} isDisabled={isPending}>
          Cancel
        </RowAction>
        <Button
          size="sm"
          variant="danger"
          isDisabled={isPending}
          onPress={onConfirm}
        >
          {confirmLabel}
        </Button>
      </div>
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
    <label className="flex items-start gap-2 rounded-lg border border-border p-3 text-sm">
      <input
        type="checkbox"
        checked={checked}
        onChange={(e) => onChange(e.target.checked)}
        className="mt-0.5 h-4 w-4 accent-accent"
        aria-label="Exempt this key from budget"
      />
      <span className="flex flex-col gap-0.5">
        <span className="font-medium text-foreground">Exempt from budget</span>
        <span className="text-xs text-muted">
          Requests on this key are logged with their cost but never counted
          toward the owner&apos;s budget or spend, and never blocked by it.
        </span>
      </span>
    </label>
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

function CreateKeyForm({
  onClose,
  onCreated,
}: {
  onClose: () => void
  onCreated: (result: CreateKeyResponse) => void
}) {
  const create = useCreateKey()
  const users = useUsers()
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
  // User-first: a key must name its owner (an existing user or a new id, which the
  // API creates as a named user). This is what keeps the dashboard from minting the
  // anonymous virtual users an omitted id would.
  const ownerMissing = userId.trim() === ""
  // The workspace comes from the organization context, which resolves after the
  // form paints. Submitting before it does would send no workspace and land the
  // key in the server's default rather than the one the switcher is showing, so
  // the button waits for that read. It waits for the read, not for a workspace:
  // a caller who belongs to none resolves to null and must still be able to
  // create a key, which the server puts in its default.
  const workspaceUnresolved = workspaceLoading

  const submit = () => {
    if (create.isPending || !scopeValid || ownerMissing || workspaceUnresolved)
      return
    const body: CreateKeyRequest = {
      key_name: keyName.trim() || null,
      // The workspace the shell is on. A key belongs to exactly one, and it is
      // what every request on that key is billed to, so it is decided here
      // rather than left to the server's default.
      workspace_id: workspace?.workspace_id,
      user_id: userId.trim(),
      expires_at: expiresAt ? new Date(expiresAt).toISOString() : null,
      allowed_models: allowedModels,
      exclude_from_budget: excludeFromBudget,
      reject_user_mismatch: rejectUserMismatch,
    }
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
    <section
      className={`${FULL_BLEED} ${BLEED_INSET} flex flex-col gap-4 border-y border-border py-5`}
    >
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
                That time is in the past; the key would be rejected immediately.
              </span>
            ) : (
              "Leave blank for a key that never expires."
            )
          }
        />
      </div>
      <UserComboBox
        value={userId}
        onChange={setUserId}
        users={users.data ?? []}
        memberLabels={memberLabels}
      />
      <button
        type="button"
        className="self-start text-xs font-medium text-link hover:text-link-hover"
        onClick={() => setShowAdvanced((v) => !v)}
      >
        {showAdvanced ? "Hide advanced" : "Advanced"}
      </button>
      {showAdvanced ? (
        <div className="flex flex-col gap-4 rounded-lg border border-border p-4">
          <OwnerAccessNote userId={userId} users={users.data ?? []} />
          <ModelScopeControl
            title="Restrict this key's models"
            description="By default this key inherits its owner's access. Optionally narrow it to a subset; a key can never exceed its owner's allowed models."
            anyLabel="Inherit owner access"
            initial={null}
            onChange={(value, valid) => {
              setAllowedModels(value)
              setScopeValid(valid)
            }}
          />
          <BudgetExemptToggle
            checked={excludeFromBudget}
            onChange={setExcludeFromBudget}
          />
          <UserMismatchPicker
            value={rejectUserMismatch}
            onChange={setRejectUserMismatch}
          />
        </div>
      ) : null}
      {/* The actions sit under a rule of their own, so the row that commits the
          form is divided from the fields rather than floating after them. */}
      <div className="flex items-center justify-end gap-3 border-t border-border pt-4">
        <Button variant="ghost" onPress={onClose}>
          Cancel
        </Button>
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
      </div>
    </section>
  )
}

function EditKeyForm({
  apiKey,
  onClose,
}: {
  apiKey: ApiKey
  onClose: () => void
}) {
  const update = useUpdateKey()
  const users = useUsers()
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
    update.mutate(
      {
        id: apiKey.id,
        body: {
          key_name: keyName.trim() || null,
          expires_at: expiresAt ? new Date(expiresAt).toISOString() : null,
          allowed_models: allowedModels,
          exclude_from_budget: excludeFromBudget,
          reject_user_mismatch: rejectUserMismatch,
        },
      },
      { onSuccess: onClose },
    )
  }

  return (
    <section
      className={`${FULL_BLEED} ${BLEED_INSET} flex flex-col gap-4 border-y border-border py-5`}
    >
      <h2 className="text-title">
        Edit{" "}
        <span className="font-mono text-[15px]">
          {apiKey.key_name ?? apiKey.id}
        </span>
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
      {apiKey.user_id ? (
        <OwnerAccessNote userId={apiKey.user_id} users={users.data ?? []} />
      ) : null}
      <ModelScopeControl
        title="Restrict this key's models"
        description="This key inherits its owner's access by default. Narrow it to a subset here; it can never exceed the owner's allowed models."
        anyLabel="Inherit owner access"
        initial={apiKey.allowed_models}
        onChange={(value, valid) => {
          setAllowedModels(value)
          setScopeValid(valid)
        }}
      />
      <BudgetExemptToggle
        checked={excludeFromBudget}
        onChange={setExcludeFromBudget}
      />
      <UserMismatchPicker
        value={rejectUserMismatch}
        onChange={setRejectUserMismatch}
      />
      <div className="flex items-center justify-end gap-3 border-t border-border pt-4">
        <Button variant="ghost" onPress={onClose}>
          Cancel
        </Button>
        <Button
          variant="primary"
          isDisabled={update.isPending || !scopeValid}
          onPress={submit}
        >
          {update.isPending ? "Saving…" : "Save changes"}
        </Button>
      </div>
    </section>
  )
}

// ---------- status + rows ----------

/**
 * A key's state: a square dot for the family and the word for the severity,
 * never a pill.
 *
 * The three are graded rather than coloured by rank. `Disabled` takes the
 * subtle rung in both channels because somebody chose it, and a deliberate
 * operator action is not a fault to be flagged. `Expired` takes the danger dot
 * and muted ink: the key stopped working, which needs noticing, but nothing was
 * breached. `Active` is a success dot with muted ink, because the normal case
 * should be legible without being loud.
 */
function StatusMark({ apiKey }: { apiKey: ApiKey }) {
  const { word, dot, ink } = !apiKey.is_active
    ? { word: "Disabled", dot: "bg-surface-subtle", ink: "text-subtle" }
    : isExpired(apiKey)
      ? { word: "Expired", dot: "bg-danger", ink: "text-muted" }
      : { word: "Active", dot: "bg-success", ink: "text-muted" }
  return (
    <span className={`flex items-center gap-2 font-mono text-[13px] ${ink}`}>
      <Dot className={dot} />
      {word}
    </span>
  )
}

/**
 * The metadata line under a key's name: what it can reach, and the two
 * behaviors that depart from the deployment default.
 *
 * Facts about the key rather than states of it, so they are set as metadata and
 * not as chips: 11px mono, uppercase, letterspaced, muted. `No models` is the
 * one that takes danger ink, because a key that can reach nothing is broken
 * rather than merely narrow. `Selected models` is muted like its siblings and
 * deliberately not link ink: it is a label, and nothing here is clickable.
 */
function KeyMetaLine({ apiKey }: { apiKey: ApiKey }) {
  const { text, tone } = accessLabel(apiKey.allowed_models)
  // Surface the exact entries on hover; the count would mislead (a wildcard is many).
  const title =
    apiKey.allowed_models && apiKey.allowed_models.length > 0
      ? apiKey.allowed_models.join(", ")
      : undefined
  const facts: { key: string; text: string; ink: string; title?: string }[] = [
    {
      key: "access",
      text,
      ink: tone === "danger" ? "text-danger" : "text-muted",
      title,
    },
  ]
  if (apiKey.exclude_from_budget) {
    facts.push({
      key: "budget",
      text: "Budget-exempt",
      ink: "text-muted",
      title:
        "Requests on this key are logged with cost but never counted toward budget",
    })
  }
  if (apiKey.reject_user_mismatch !== null) {
    facts.push({
      key: "user",
      text: apiKey.reject_user_mismatch ? "Strict user" : "Lenient user",
      ink: "text-muted",
      title: apiKey.reject_user_mismatch
        ? "This key always rejects a request naming a different user, whatever the deployment setting says"
        : "This key always accepts a request naming a different user; spend still binds to its owner",
    })
  }
  return (
    <span className="flex flex-wrap items-center gap-x-3 gap-y-1 font-mono text-[11px] tracking-[0.1em] uppercase">
      {facts.map((fact) => (
        <span key={fact.key} className={fact.ink} title={fact.title}>
          {fact.text}
        </span>
      ))}
    </span>
  )
}

export function KeysPage() {
  // Scoped to the workspace the switcher is on: a key belongs to exactly one,
  // and this page is in the workspace context.
  const { selected: workspace } = useSelectedWorkspace()
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

  const rows = keys.data ?? []
  const loading = keys.isLoading
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
  // Which row is armed, and for what. Page state rather than per-button state,
  // because the confirmation is a strip under the row now and only one may be
  // open at a time.
  const [armed, setArmed] = useState<{
    id: string
    kind: "regenerate" | "delete"
  } | null>(null)

  // the stable handlers) so DataTable's per-row cache holds across selection
  // clicks; see the DataTable docstring.
  const columns = useMemo<DataTableColumn<ApiKey>[]>(
    () => [
      {
        id: "name",
        header: "Name",
        isRowHeader: true,
        // Two lines: the name at 16px, then the metadata line 5px under it.
        cell: (k) => (
          <div className="flex flex-col gap-[5px]">
            <span className="text-base text-foreground">
              {k.key_name ?? <span className="text-muted">(unnamed)</span>}
            </span>
            <KeyMetaLine apiKey={k} />
          </div>
        ),
      },
      {
        id: "status",
        header: "Status",
        cell: (k) => <StatusMark apiKey={k} />,
      },
      {
        id: "owner",
        header: "Owner",
        // A member is named; anything else keeps the raw id, which for a
        // hand-made owner like `ci-bot` is already the readable form. The id
        // stays in the title so the value actually sent on a request is still
        // recoverable from this column.
        cell: (k) => {
          // One column, two faces, deliberately: a member is a person and takes
          // the body face, while a raw id like `ci-bot` is an identifier and
          // takes the mono one. The face is what tells the two apart, so
          // neither needs a chip to say which it is.
          if (isVirtualUser(k.user_id)) {
            return (
              <span className="font-mono text-[13px] text-subtle">virtual</span>
            )
          }
          const member = k.user_id ? memberLabels.get(k.user_id) : undefined
          if (member) {
            return (
              <span className="text-sm text-foreground" title={k.user_id ?? ""}>
                {member}
              </span>
            )
          }
          return (
            <span className="font-mono text-[13px] text-muted">
              {k.user_id ?? "—"}
            </span>
          )
        },
      },
      {
        id: "key",
        header: "Key",
        cell: (k) => (
          <span className="font-mono text-[13px] text-muted">
            {k.key_prefix ? `${k.key_prefix}…` : "—"}
          </span>
        ),
      },
      {
        id: "created",
        header: "Created",
        cell: (k) => (
          <span className="font-mono text-[13px] text-muted">
            {formatDate(k.created_at)}
          </span>
        ),
      },
      {
        id: "last_used",
        header: "Last used",
        cell: (k) => (
          <span className="font-mono text-[13px] text-muted">
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
          <div className="flex items-center justify-end gap-4">
            <RowAction
              isDisabled={updateKey.isPending}
              onPress={() => setActive(k, !k.is_active)}
            >
              {k.is_active ? "Disable" : "Enable"}
            </RowAction>
            <RowAction
              onPress={() => {
                setAddOpen(false)
                setEditing(k.id)
              }}
            >
              Edit
            </RowAction>
            <RowAction
              isDanger={armed?.id === k.id && armed.kind === "regenerate"}
              onPress={() => setArmed({ id: k.id, kind: "regenerate" })}
            >
              Regenerate
            </RowAction>
            {/* Permanent delete is only offered once a key is disabled, so a live
              caller can't be broken (and its audit trail erased) in one click. */}
            {k.is_active ? null : (
              <RowAction
                isDanger={armed?.id === k.id && armed.kind === "delete"}
                onPress={() => setArmed({ id: k.id, kind: "delete" })}
              >
                Delete
              </RowAction>
            )}
          </div>
        ),
      },
    ],
    [updateKey.isPending, setActive, memberLabels, armed],
  )

  // The armed confirmation, rendered as a strip under its own row rather than
  // inside the actions cell. Referentially stable per the DataTable docstring.
  const renderArmed = useCallback(
    (k: ApiKey) => {
      if (!armed || armed.id !== k.id) return null
      if (armed.kind === "regenerate") {
        return (
          <ArmedStrip
            confirmLabel="Regenerate"
            isPending={rotateKey.isPending}
            message={
              <>
                Regenerate the secret for <strong>{label(k)}</strong>? The
                current secret stops working immediately, with no grace period.
              </>
            }
            onConfirm={() => {
              setArmed(null)
              regenerate(k)
            }}
            onCancel={() => setArmed(null)}
          />
        )
      }
      return (
        <ArmedStrip
          confirmLabel="Delete permanently"
          isPending={deleteKey.isPending}
          message={
            <>
              Permanently delete <strong>{label(k)}</strong>? This removes the
              key and unlinks its usage history. Cannot be undone.
            </>
          }
          onConfirm={() => {
            setArmed(null)
            deleteKey.mutate(k.id)
          }}
          onCancel={() => setArmed(null)}
        />
      )
    },
    [
      armed,
      rotateKey.isPending,
      deleteKey.isPending,
      deleteKey.mutate,
      regenerate,
    ],
  )

  // Bulk delete targets only already-disabled keys, mirroring the per-row rule
  // that a live key must be disabled before it can be permanently deleted.
  const deletableSelected = selectedKeys.filter((k) => !k.is_active)

  return (
    <div className="flex flex-col">
      <header className="flex flex-col gap-4 pb-5 sm:flex-row sm:items-start sm:justify-between">
        <div className="max-w-[620px]">
          <h1 className="font-display text-[28px] leading-[34px] font-semibold tracking-[-0.01em] text-foreground">
            API keys
          </h1>
          <p className="mt-1 text-sm text-muted">
            Issue and revoke the keys that authenticate callers to this gateway.
            Secrets are shown once at creation.
          </p>
        </div>
        {addOpen ? null : (
          <Button
            variant="primary"
            className="shrink-0"
            onPress={() => {
              setEditing(null)
              setAddOpen(true)
            }}
          >
            Create key
          </Button>
        )}
      </header>

      <ErrorBanner
        error={
          keys.error ?? updateKey.error ?? rotateKey.error ?? deleteKey.error
        }
      />

      {/* A key's owner and its spending limit are both set elsewhere now, on the
          organization rail. This page is where an operator arrives looking for
          them, so it says where they went rather than leaving the sidebar to be
          re-learned. The two links here stay in link ink: they are real links
          inside a sentence, which is what that ink is for. */}
      <p className="max-w-[620px] pb-5 text-sm text-muted">
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

      {revealed ? (
        <RevealSecretStrip
          title={revealed.title}
          result={revealed.result}
          onClose={() => {
            setRevealed(null)
            // Drop the one-time secret from mutation state so reopening
            // Create/Regenerate never flashes the previous key.
            rotateKey.reset()
          }}
        />
      ) : null}

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
        <TableScrollFrame className="otari-keys-table">
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
            detailKey={armed?.id ?? null}
            renderDetail={renderArmed}
          />
        </TableScrollFrame>
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
    </div>
  )
}
