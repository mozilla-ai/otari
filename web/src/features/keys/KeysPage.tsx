import { Spinner } from "@heroui/react"
import { Link } from "@tanstack/react-router"
import {
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
import { Button } from "@/design-system/actions/Button"
import {
  CopyField,
  concealedFingerprint,
} from "@/design-system/actions/CopyField"
import { BulkActionBar } from "@/design-system/data/BulkActionBar"
import { DataTable, type DataTableColumn } from "@/design-system/data/DataTable"
import { ConfirmDialog } from "@/design-system/feedback/ConfirmDialog"
import { EmptyMessage } from "@/design-system/feedback/EmptyMessage"
import { EmptyState } from "@/design-system/feedback/EmptyState"
import { ErrorBanner } from "@/design-system/feedback/ErrorBanner"
import { FormDialog } from "@/design-system/feedback/FormDialog"
import { Checkbox } from "@/design-system/forms/Checkbox"
import { Field } from "@/design-system/forms/Field"
import { useDirtySnapshot } from "@/design-system/forms/useDirtySnapshot"
import { useConfirmationFocus } from "@/design-system/hooks/useConfirmationFocus"
import { Dot } from "@/design-system/indicators/Dot"
import { PageIntro } from "@/design-system/layout/PageIntro"
import { TableScrollFrame } from "@/design-system/layout/TableScrollFrame"
import { FilterSelect } from "@/design-system/navigation/FilterSelect"
import {
  accessLabel,
  ModelScopeControl,
} from "@/features/models/ModelScopeControl"
import { useMemberAttributionLabels } from "@/features/organization/attribution"
import { organizationUsers } from "@/features/users/organizationUsers"
import { UserComboBox } from "@/features/users/UserComboBox"
import {
  useCreateKey,
  useDeleteKey,
  useKeys,
  useKeysScope,
  useRotateKey,
  useUpdateKey,
} from "@/shared/api/apiKeys"
import { useUsers } from "@/shared/api/users"
import { MissingGatewayAddressNotice } from "@/shared/components/access/MissingGatewayAddressNotice"
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
import { KeyActionsMenu } from "./KeyActionsMenu"
import { isVirtualUser, keyFingerprint, secretCaption } from "./secretCaption"

// ---------- helpers ----------

// Once per module rather than once per row, and pinned like every other
// formatter here: an expiry reading "in 2 Tagen" beside an English column
// heading is the mixed page formatting-and-i18n.md exists to prevent. This one
// is not `formatRelative`, which is past-only by design; an expiry is the one
// place the dashboard speaks about the future.
const expiryRelative = new Intl.RelativeTimeFormat("en-US", { numeric: "auto" })

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
  const coarsest = units.find(([, sec]) => abs >= sec)
  if (!coarsest) return expiryRelative.format(diffSec, "second")
  const [unit, sec] = coarsest
  return expiryRelative.format(Math.round(diffSec / sec), unit)
}

type Layout = "wide" | "compact" | "mobile"

// `md` and below is the list; between that and 1100 the lowest-priority lanes
// fold into the row. The region's own width decides, except that a viewport
// under `md` is the list whatever the region measures.
function layoutFor(viewport: number, region = viewport): Layout {
  if (viewport < 768 || region < 600) return "mobile"
  return region < 1100 ? "compact" : "wide"
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
  const pad = (value: number) => String(value).padStart(2, "0")
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}T${pad(d.getHours())}:${pad(d.getMinutes())}`
}

const label = (apiKey: ApiKey): string => apiKey.key_name ?? apiKey.id

// Stable row-key getter so DataTable's per-row cache holds across re-renders.
const getKeyRowKey = (apiKey: ApiKey): string => apiKey.id

function renderFingerprint(apiKey: ApiKey) {
  return (
    <code className="whitespace-nowrap text-mono-caption text-muted">
      {keyFingerprint(apiKey) ?? "—"}
    </code>
  )
}

// ---------- the one-time secret ----------

/** One-time key handoff with copyable request examples. */
function KeySecretStep({
  announce,
  result,
  memberLabels,
  secretRef,
}: {
  /** The dialog's title, repeated as the alert's own name. */
  announce: string
  result: CreateKeyResponse
  memberLabels: ReadonlyMap<string, string>
  secretRef: RefObject<HTMLInputElement | HTMLTextAreaElement | null>
}) {
  // Where a request from this deployment belongs, which is not always the address
  // that served this page: a hosted control plane serves the dashboard and not
  // the API. Undefined when it has not said where its gateway is (otari#823).
  const baseUrl = resolveSnippetBaseUrl(useDeployment())
  const secret = result.key
  // The key and both snippets show one stand-in, so the credential on this
  // screen reads as one thing rather than three.
  const concealedSecret = concealedFingerprint(
    result.key_prefix ?? undefined,
    result.key_suffix ?? undefined,
  )
  // The snippets carry the same credential, so all three fields share visibility.
  const [isSecretRevealed, setIsSecretRevealed] = useState(false)

  // The same two calls the setup guide hands out with its own key; the builders
  // are shared so an operator cannot be shown two dialects of one request.
  // One value rather than two, so a deployment that named no gateway drops both
  // together and the copy under the heading cannot disagree with the fields.
  const snippets = baseUrl
    ? {
        curl: buildCurlSnippet({ baseUrl, apiKey: secret }),
        python: buildPythonSnippet({ baseUrl, apiKey: secret }),
        // The same two commands around the stand-in, which is what the
        // snippets show whenever the key is concealed. Without them concealing
        // the key would leave it in plain sight twice over, in the requests
        // that explain it.
        concealedCurl: buildCurlSnippet({ baseUrl, apiKey: concealedSecret }),
        concealedPython: buildPythonSnippet({
          baseUrl,
          apiKey: concealedSecret,
        }),
      }
    : undefined

  return (
    // `alert` rather than `status`: this is the one message in the product that
    // is gone forever if it is missed, which is also why the dialog around it
    // does not dismiss.
    <div role="alert" aria-label={announce} className="flex flex-col gap-4">
      <p className="text-sm text-muted">
        Copy this key now. For security it is shown only once and cannot be
        retrieved later. If you lose it, use Regenerate to issue a new secret.
      </p>
      {/* The name above the field rather than as its label: `CopyField`'s label
          is a real one and answers "which field is this" for a screen reader,
          which "deploy-key" does not, and these arrive in threes. */}
      <div className="flex flex-col gap-1">
        <p className="truncate text-emphasis">{result.key_name ?? result.id}</p>
        <CopyField
          label="Secret key"
          value={secret}
          concealed={concealedSecret}
          isRevealed={isSecretRevealed}
          onRevealChange={setIsSecretRevealed}
          fieldRef={secretRef}
        />
        <p className="text-caption">{secretCaption(result, memberLabels)}</p>
      </div>
      <div className="flex flex-col gap-2">
        <div>
          <div className="text-body">Make your first call</div>
          {snippets === undefined ? (
            <MissingGatewayAddressNotice />
          ) : (
            <p className="text-caption">
              Replace <code>{SNIPPET_MODEL_PLACEHOLDER}</code> with a model from
              the Models page.
            </p>
          )}
        </div>
        {snippets !== undefined ? (
          <>
            <CopyField
              label="curl"
              value={snippets.curl}
              concealed={snippets.concealedCurl}
              isRevealed={isSecretRevealed}
              onRevealChange={setIsSecretRevealed}
              isMultiline
            />
            <CopyField
              label="Python (Otari SDK)"
              value={snippets.python}
              concealed={snippets.concealedPython}
              isRevealed={isSecretRevealed}
              onRevealChange={setIsSecretRevealed}
              isMultiline
            />
          </>
        ) : null}
      </div>
    </div>
  )
}

// ---------- create / edit forms (FormDialog) ----------

// Shows the selected owner's model access so the operator sees the ceiling this
// key narrows within (a key can inherit it or restrict to a subset, never exceed).
function OwnerAccessNote({ userId, users }: { userId: string; users: User[] }) {
  const id = userId.trim()
  if (id === "") {
    return (
      <p className="text-caption">
        Choose an owner above to see the models this key can inherit.
      </p>
    )
  }
  const owner = users.find((user) => user.user_id === id)
  if (!owner) {
    return (
      <p className="text-caption">
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
    <p className="text-caption">
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
  // The consequence sits beside the control rather than inside its label: a
  // checkbox whose accessible name is a whole paragraph is read out in full on
  // every focus. Beside rather than wrapped, so there is also no second
  // <label> around the one `Checkbox` already renders.
  return (
    <div className="flex flex-col gap-0.5 border border-control-border p-3">
      <Checkbox isSelected={checked} onChange={onChange}>
        <span className="font-medium text-foreground">Exempt from budget</span>
      </Checkbox>
      <p className="text-caption">
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
      <label htmlFor={selectId} className="text-body">
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
      <span className="text-caption">
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
/**
 * Create a key, and hand back the secret, in one frame.
 *
 * Two steps rather than two surfaces: the form submits, and when it resolves the
 * same dialog swaps its title, its body and its submit label. That is what makes
 * the secret arrive where the operator is already looking, and it is why the
 * success step is not a prop on `FormDialog` but a branch here.
 *
 * `lg` on both steps, and that is the reason the size is set at all: the form is
 * five fields and would sit happily at `md`, but the secret step carries two
 * snippets, and a frame that changed width between pressing Create and reading
 * the key would read as a second dialog.
 */
function CreateKeyDialog({
  isDeploymentWide,
  isOpen,
  onOpenChange,
  memberLabels,
  returnFocusRef,
}: {
  isDeploymentWide: boolean
  isOpen: boolean
  onOpenChange: (open: boolean) => void
  memberLabels: ReadonlyMap<string, string>
  /** Where focus goes when nothing else claims it. See `close`. */
  returnFocusRef: RefObject<HTMLButtonElement | null>
}) {
  const create = useCreateKey()
  // `/users` is operator-only; a member's form has no owner picker to feed.
  // Gated on `isOpen` as well, because this dialog stays mounted while closed so
  // it can animate out, and `fetchAllUsers` walks up to 100 pages of 1000: left
  // on the deployment alone it ran that walk on every visit to the page. The
  // query's own `staleTime` makes a reopen free.
  const users = useUsers(isDeploymentWide && isOpen)
  // Every key in the caller's organization, workspace filter deliberately unset:
  // it is one half of what says a user belongs here (see `organizationUsers`),
  // and the owner offered is the organization's rather than this workspace's,
  // because that is the scope the key is billed in. Gated like the users read
  // above, and it shares `useKeys`'s cache with the page's own workspace-scoped
  // list rather than replacing it.
  const organizationKeys = useKeys(undefined, isDeploymentWide && isOpen)
  // What the picker may offer: the deployment's user list narrowed to this
  // organization. An unfiltered one named every tenant's people (otari-ai#2108).
  const ownerOptions = organizationUsers(
    users.data ?? [],
    memberLabels,
    organizationKeys.data ?? [],
  )
  const { selected: workspace, isLoading: workspaceLoading } =
    useSelectedWorkspace()
  const [keyName, setKeyName] = useState("")
  const [expiresAt, setExpiresAt] = useState("")
  const [showAdvanced, setShowAdvanced] = useState(false)
  const [userId, setUserId] = useState("")
  const [allowedModels, setAllowedModels] = useState<string[] | null>(null)
  const [excludeFromBudget, setExcludeFromBudget] = useState(false)
  const [rejectUserMismatch, setRejectUserMismatch] = useState<boolean | null>(
    null,
  )
  const [scopeValid, setScopeValid] = useState(true)
  // The secret, once there is one. Its presence is the step: unset is the form.
  const [created, setCreated] = useState<CreateKeyResponse>()
  const secretRef = useRef<HTMLInputElement | HTMLTextAreaElement | null>(null)

  useEffect(() => {
    // The secret field is what the operator came for, so it takes focus when the
    // step arrives. Focus without a selection: the field is concealed, and
    // selecting the stand-in would invite a Ctrl/Cmd-C that copies bullets.
    if (created) secretRef.current?.focus()
  }, [created])

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
  const isBlocked = !scopeValid || ownerMissing || workspaceUnresolved

  // What the form owns, handed to the guard whole rather than compared field by
  // field: the two drifted apart once already, with the guard armed for three
  // fields out of seven while "Create another" left the rest behind. Reopening
  // is a remount now (the page keys this on an open counter), so `resetForm`
  // answers for "Create another" alone, which is a reset inside an open dialog,
  // and putting every field back to its seed reads clean again without telling
  // the guard anything. `showAdvanced` is in neither: it reveals fields rather
  // than holding a value, and a disclosure left open is not unsaved work.
  // `scopeValid` is, though: a model scope typed to something invalid is work
  // the operator would lose.
  const { isDirty } = useDirtySnapshot({
    keyName,
    expiresAt,
    userId,
    allowedModels,
    excludeFromBudget,
    rejectUserMismatch,
    scopeValid,
  })

  const resetForm = () => {
    setKeyName("")
    setExpiresAt("")
    setShowAdvanced(false)
    setUserId("")
    setAllowedModels(null)
    setExcludeFromBudget(false)
    setRejectUserMismatch(null)
    setScopeValid(true)
    create.reset()
  }

  const close = () => {
    onOpenChange(false)
    // React Aria returns focus to whatever had it when the dialog opened, which
    // is right for the heading's own button and wrong for the empty state's:
    // that one is gone by now, because creating a key is what fills the table.
    // A frame later anything with a claim has had its turn, and an empty
    // document.body means nothing took it.
    requestAnimationFrame(() => {
      if (document.activeElement === document.body) {
        returnFocusRef.current?.focus()
      }
    })
  }

  const submit = () => {
    if (create.isPending || isBlocked) return
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
        setCreated(result)
      },
    })
  }

  if (created) {
    return (
      <FormDialog
        isOpen={isOpen}
        // Guarded the same way the form step is: a truthy `open` is forwarded
        // straight through, so taking `close` bare here made any request to
        // open this dialog close it instead. Nothing sends one today, since the
        // only source is the hidden trigger, but this is the step where being
        // wrong loses the key.
        onOpenChange={(open) => (open ? onOpenChange(true) : close())}
        size="lg"
        // A dismiss here loses the key for good, so there is one way out and it
        // is the acknowledgement. `FormDialog` drops its close control and its
        // Cancel to match, and `isDismissable` does not reach the line above:
        // it guards `requestClose`, not a forwarded open.
        isDismissable={false}
        title="Key created"
        // The strip's own words, not "Done": this control is an
        // acknowledgement rather than a dismissal, and it is the only way out.
        submitLabel="I’ve saved this key"
        onSubmit={close}
        isPending={false}
        footerStart={
          <Button
            variant="ghost"
            onPress={() => {
              setCreated(undefined)
              resetForm()
            }}
          >
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
      </FormDialog>
    )
  }

  return (
    <FormDialog
      isOpen={isOpen}
      onOpenChange={(open) => (open ? onOpenChange(true) : close())}
      size="lg"
      title="New key"
      description="The secret is shown once, right after you create it."
      submitLabel="Create key"
      onSubmit={submit}
      isPending={create.isPending}
      isSubmitDisabled={isBlocked}
      error={create.error}
      isDirty={isDirty}
    >
      <Field
        label="Name"
        value={keyName}
        onChange={setKeyName}
        placeholder="ci-bot"
        autoFocus
        description="A label to recognize this key later."
        shouldReserveMessage
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
        shouldReserveMessage
      />
      {isDeploymentWide ? (
        <UserComboBox
          value={userId}
          onChange={setUserId}
          users={ownerOptions}
        />
      ) : (
        <p className="text-caption">
          This key is yours: requests on it are billed to you and count against
          your budget.
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
        <div className="flex flex-col gap-4 border border-control-border p-4">
          {isDeploymentWide ? (
            <OwnerAccessNote userId={userId} users={ownerOptions} />
          ) : null}
          <ModelScopeControl
            title="Restrict this key's models"
            description={
              isDeploymentWide
                ? "By default this key inherits its owner's access. Optionally narrow it to a subset; a key can never exceed its owner's allowed models."
                : "By default this key inherits your model access. Optionally narrow it to a subset; a key can never exceed your allowed models."
            }
            anyLabel={
              isDeploymentWide ? "Inherit owner access" : "Inherit your access"
            }
            initial={null}
            onChange={(value, isValid) => {
              setAllowedModels(value)
              setScopeValid(isValid)
            }}
          />
          {/* Exempting a key from a budget is the deployment operator's call,
              never the owner's: on a member's own-key form this would let them
              opt their own traffic out of their own budget. */}
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
    </FormDialog>
  )
}

/**
 * The secret a regenerate produced, in the same frame a create's arrives in.
 *
 * No form step and no "Create another": the key already exists and this is the
 * new secret for it, so the dialog opens where the create flow ends up.
 */
function RegeneratedSecretDialog({
  title,
  result,
  memberLabels,
  onClose,
}: {
  title: string
  result: CreateKeyResponse
  memberLabels: ReadonlyMap<string, string>
  onClose: () => void
}) {
  const secretRef = useRef<HTMLInputElement | HTMLTextAreaElement | null>(null)
  useEffect(() => {
    secretRef.current?.focus()
  }, [])
  return (
    <FormDialog
      isOpen
      onOpenChange={onClose}
      size="lg"
      // A stray backdrop click here loses the key forever: this is the one
      // message in the product that cannot be shown again.
      isDismissable={false}
      title={title}
      submitLabel="I’ve saved this key"
      onSubmit={onClose}
      isPending={false}
    >
      <KeySecretStep
        announce={title}
        result={result}
        memberLabels={memberLabels}
        secretRef={secretRef}
      />
    </FormDialog>
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
  const { isDirty } = useDirtySnapshot({
    keyName,
    expiresAt,
    allowedModels,
    excludeFromBudget,
    rejectUserMismatch,
    scopeValid,
  })

  const submit = () => {
    if (update.isPending || !scopeValid) return
    const shared = {
      key_name: keyName.trim() || null,
      expires_at: expiresAt ? new Date(expiresAt).toISOString() : null,
      allowed_models: allowedModels,
      reject_user_mismatch: rejectUserMismatch,
    }
    // The member surface has no budget exemption to send (see CreateKeyDialog).
    const body: UpdateKeyRequest | UpdateOwnKeyRequest = isDeploymentWide
      ? { ...shared, exclude_from_budget: excludeFromBudget }
      : shared
    update.mutate({ id: apiKey.id, body }, { onSuccess: onClose })
  }

  return (
    <FormDialog
      isOpen
      onOpenChange={(open) => {
        if (!open) onClose()
      }}
      // `lg` as on the create form: the same key, the same frame.
      size="lg"
      title="Edit key"
      description={<code>{apiKey.key_name ?? apiKey.id}</code>}
      submitLabel="Save"
      onSubmit={submit}
      isPending={update.isPending}
      isSubmitDisabled={!scopeValid}
      isDirty={isDirty}
      error={update.error}
    >
      <Field
        label="Name"
        value={keyName}
        onChange={setKeyName}
        placeholder="ci-bot"
        autoFocus
        shouldReserveMessage={false}
      />
      <Field
        label="Expires"
        value={expiresAt}
        onChange={setExpiresAt}
        type="datetime-local"
        description="Blank clears the expiry."
        shouldReserveMessage
      />
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
        onChange={(value, isValid) => {
          setAllowedModels(value)
          setScopeValid(isValid)
        }}
      />
      {/* Operator-only, as on the create form: a member editing their own key
          must not be able to take it out of their own budget. */}
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
    </FormDialog>
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
    ? { word: "Disabled", dot: "bg-text-subtle", ink: "text-subtle" }
    : isExpired(apiKey)
      ? { word: "Expired", dot: "bg-danger", ink: "text-muted" }
      : { word: "Active", dot: "bg-success", ink: "text-muted" }
  return (
    <span className={`flex items-center gap-2 text-mono-caption ${ink}`}>
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
function KeyMetaLine({
  apiKey,
  face = "text-mono-overline",
}: {
  apiKey: ApiKey
  /**
   * The overline's uppercase mono is right beside a 16px name and wrong inside
   * the folded caption line, where it would sit between two runs of sentence-case
   * body text.
   */
  face?: string
}) {
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
    <span
      className={`inline-flex max-w-full gap-3 overflow-hidden whitespace-nowrap align-bottom ${face}`}
    >
      {facts.map((fact) => (
        <span
          key={fact.key}
          className={`truncate ${fact.ink}`}
          title={fact.title}
        >
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
  // Bumped on each open, and the create dialog is keyed on it, so the draft is
  // fresh every time and untouched through the exit: the dialog keeps its
  // content while it animates out, so clearing on the way out blanked the
  // secret step to an empty form for the length of the animation.
  //
  // No guard against a second press while it is open: react-aria takes the page
  // out of the accessibility tree behind a modal, so the heading's trigger
  // cannot be reached until the dialog has closed. Probed rather than assumed,
  // because the trigger is still on screen: with the dialog open there is
  // exactly one control named "Create key" and it is the dialog's own submit.
  const [openCount, setOpenCount] = useState(0)
  const openCreate = () => {
    setEditing(undefined)
    setOpenCount((n) => n + 1)
    setAddOpen(true)
  }
  const [editing, setEditing] = useState<string>()
  const [regenerated, setRegenerated] = useState<{
    title: string
    result: CreateKeyResponse
  }>()
  // Where focus lands when a dialog opened from a consumed confirm closes.
  // `FormDialog` returns focus to whatever had it, and a regenerate's trigger is
  // a row action that has already disarmed itself, so the page's own primary
  // action is the stable landing spot nearest to where the operator was.
  const createButtonRef = useRef<HTMLButtonElement>(null)

  const rows = keys.data ?? []
  // An unresolved scope is a loading state: which surface the page reads, and
  // which affordances it draws, are both undecided until the organization
  // context answers, and a disabled query reports `isLoading` false.
  const loading = !scope.isReady || keys.isLoading
  const editingKey = rows.find((apiKey) => apiKey.id === editing)
  const showOnboarding = !loading && rows.length === 0
  const selection = useTableSelection()
  const [bulkDeleteOpen, setBulkDeleteOpen] = useState(false)
  const [bulkError, setBulkError] = useState<unknown>(undefined)
  const [bulkPending, setBulkPending] = useState(false)

  const selectableKeys = rows.map((apiKey) => apiKey.id)
  const selectedIds = resolveSelectedIds(selection.selectedKeys, selectableKeys)
  const selectedKeys = rows.filter((apiKey) => selectedIds.includes(apiKey.id))

  // Stable handlers (mutate fns are referentially stable in TanStack Query) so
  // the memoized columns below survive unrelated re-renders.
  const setActive = useCallback(
    (k: ApiKey, active: boolean) =>
      updateKey.mutate({ id: k.id, body: { is_active: active } }),
    [updateKey.mutate],
  )

  const regenerate = useCallback(
    (apiKey: ApiKey) =>
      rotateKey.mutate(apiKey.id, {
        onSuccess: (result) => {
          setPendingRegenerate(undefined)
          setRegenerated({ title: `New secret for ${label(apiKey)}`, result })
        },
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

  const [pendingRegenerate, setPendingRegenerate] = useState<ApiKey>()
  const [pendingDelete, setPendingDelete] = useState<ApiKey>()
  const [lastAction, setLastAction] = useState<string>()
  // Still load-bearing, though `confirmRef` goes unused: a menu closes on choice,
  // so the item that was focused is unmounted before the dialog opens and
  // react-aria's own restore has no target left. Measured, not assumed: dropping
  // this strands focus in both the regenerate-cancel and edit-close tests.
  const { triggerRef: actionTriggerRef } = useConfirmationFocus(
    !!(pendingRegenerate || pendingDelete || editing || regenerated),
  )
  const tableRegion = useRef<HTMLDivElement>(null)
  // Seeded from the viewport rather than defaulting to "wide": the observer only
  // reports after the first layout, so a phone would paint the desktop table for
  // a frame and then swap to the list.
  const [layout, setLayout] = useState<Layout>(() =>
    typeof window === "undefined" ? "wide" : layoutFor(window.innerWidth),
  )
  useEffect(() => {
    const region = tableRegion.current
    if (!region) return
    const measure = (width: number) => {
      if (width > 0) setLayout(layoutFor(window.innerWidth, width))
    }
    // The observer hands back the region's width without forcing layout, so
    // remembering it is what lets the viewport listener below re-decide without
    // measuring. One read here at setup, none per event.
    let regionWidth = region.getBoundingClientRect().width
    const observer = new ResizeObserver(([entry]) => {
      regionWidth = entry.contentRect.width
      measure(regionWidth)
    })
    observer.observe(region)
    // The region can keep its width while the viewport crosses `md`, which the
    // observer alone never reports. `window.innerWidth` is not a layout read.
    const onResize = () => measure(regionWidth)
    window.addEventListener("resize", onResize)
    return () => {
      observer.disconnect()
      window.removeEventListener("resize", onResize)
    }
  }, [])

  const ownerLabel = useCallback(
    (apiKey: ApiKey) =>
      isVirtualUser(apiKey.user_id)
        ? "virtual"
        : apiKey.user_id
          ? (memberLabels.get(apiKey.user_id) ?? apiKey.user_id)
          : "—",
    [memberLabels],
  )
  const renderActions = useCallback(
    (apiKey: ApiKey) => (
      <KeyActionsMenu
        apiKey={apiKey}
        triggerRef={lastAction === apiKey.id ? actionTriggerRef : undefined}
        onAction={() => setLastAction(apiKey.id)}
        owner={isDeploymentWide ? ownerLabel(apiKey) : undefined}
        // Those lanes are off the row below `wide`, so the menu is the only
        // place left that can show them whole.
        hasDetails={layout !== "wide"}
        isPending={updateKey.isPending || rotateKey.isPending}
        onToggle={() => setActive(apiKey, !apiKey.is_active)}
        onEdit={() => {
          setAddOpen(false)
          setEditing(apiKey.id)
        }}
        onRegenerate={() => setPendingRegenerate(apiKey)}
        onDelete={() => setPendingDelete(apiKey)}
      />
    ),
    [
      lastAction,
      actionTriggerRef,
      isDeploymentWide,
      ownerLabel,
      layout,
      updateKey.isPending,
      rotateKey.isPending,
      setActive,
    ],
  )
  // Memoized on what the cells read, so DataTable's per-row cache holds across
  // selection clicks; see its docstring.
  const columns = useMemo<DataTableColumn<ApiKey>[]>(
    () => [
      {
        id: "name",
        header: "Name",
        isRowHeader: true,
        cell: (apiKey) => (
          <div className="flex min-w-0 flex-col gap-1">
            <span className="truncate text-base text-foreground">
              {apiKey.key_name ?? <span className="text-muted">(unnamed)</span>}
            </span>
            {/* Folded, the line joins the owner and the last use to the meta
              facts, so it takes one face throughout rather than setting an
              uppercase mono run between two runs of body text. */}
            <div className="truncate text-caption">
              {layout === "compact" && isDeploymentWide ? (
                <>
                  <span>{ownerLabel(apiKey)}</span>
                  {" · "}
                </>
              ) : null}
              <KeyMetaLine
                apiKey={apiKey}
                face={layout === "compact" ? "text-caption" : undefined}
              />
              {layout === "compact"
                ? ` · used ${relative(apiKey.last_used_at) ?? "never"}`
                : null}
            </div>
          </div>
        ),
      },
      {
        id: "status",
        header: "Status",
        cell: (apiKey) => <StatusMark apiKey={apiKey} />,
      },
      // Every key on a member's page is their own, so an Owner column there
      // would repeat one name down the table.
      ...(isDeploymentWide
        ? [
            {
              id: "owner",
              header: "Owner",
              // One column, two faces, deliberately: a member is a person and
              // takes the body face, while a raw id like `ci-bot` is an
              // identifier and takes the mono one. The face is what tells the two
              // apart, so neither needs a chip to say which it is. The id stays
              // in the title, so the value actually sent on a request is
              // recoverable from a truncated cell.
              cell: (apiKey: ApiKey) => {
                const member =
                  !isVirtualUser(apiKey.user_id) && apiKey.user_id
                    ? memberLabels.get(apiKey.user_id)
                    : undefined
                if (member) {
                  return (
                    <span
                      className="block truncate text-sm text-foreground"
                      title={apiKey.user_id ?? ""}
                    >
                      {member}
                    </span>
                  )
                }
                return (
                  <span
                    className={`block truncate text-mono-caption ${
                      isVirtualUser(apiKey.user_id)
                        ? "text-subtle"
                        : "text-muted"
                    }`}
                    title={apiKey.user_id ?? ""}
                  >
                    {ownerLabel(apiKey)}
                  </span>
                )
              },
            } satisfies DataTableColumn<ApiKey>,
          ]
        : []),
      {
        id: "key",
        header: "Key",
        cell: renderFingerprint,
      },
      {
        id: "created",
        header: "Created",
        cell: (apiKey) => (
          <span className="text-mono-caption text-muted">
            {formatDate(apiKey.created_at)}
          </span>
        ),
      },
      {
        id: "last_used",
        header: "Last used",
        cell: (apiKey) => (
          <span className="text-mono-caption text-muted">
            {relative(apiKey.last_used_at) ?? "never"}
          </span>
        ),
      },
      {
        id: "expires",
        header: "Expires",
        cell: (apiKey) => (
          <span
            className="text-muted"
            title={
              apiKey.expires_at
                ? new Date(apiKey.expires_at).toLocaleString()
                : undefined
            }
          >
            {apiKey.expires_at ? formatDate(apiKey.expires_at) : "never"}
          </span>
        ),
      },
      {
        id: "actions",
        header: "Actions",
        align: "end",
        cell: renderActions,
      },
    ],
    [layout, isDeploymentWide, memberLabels, ownerLabel, renderActions],
  )
  const visibleColumns = useMemo(
    () =>
      layout === "compact"
        ? columns.filter(
            (column) =>
              !["owner", "created", "last_used", "expires"].includes(column.id),
          )
        : columns,
    [columns, layout],
  )

  // Bulk delete targets only already-disabled keys, mirroring the per-row rule
  // that a live key must be disabled before it can be permanently deleted.
  const deletableSelected = selectedKeys.filter((apiKey) => !apiKey.is_active)

  return (
    <div ref={tableRegion} className="flex min-w-0 flex-col">
      <PageIntro
        title="API keys"
        action={
          <Button
            // Focus returns here when a dialog opened from a row action closes,
            // so the control has to stay addressable. It also no longer hides
            // while the create dialog is open: the dialog is over the page, and
            // a heading that loses its action while you are using it is the
            // inconsistency this page had.
            ref={createButtonRef}
            variant="primary"
            onPress={openCreate}
          >
            Create key
          </Button>
        }
      >
        {isDeploymentWide
          ? "Issue and revoke the keys that authenticate callers to this gateway. Secrets are shown once at creation."
          : "Create and manage your own keys for calling this gateway. Secrets are shown once at creation."}
      </PageIntro>

      {/* Not the deletes: each reports inside its own confirm dialog. */}
      <ErrorBanner error={keys.error ?? updateKey.error} />

      {/* A key's owner and its spending limit are both set elsewhere now, on the
          organization rail. This page is where an operator arrives looking for
          them, so it says where they went rather than leaving the sidebar to be
          re-learned. A member's keys are their own and both destinations refuse
          them, so their page states the ownership rule instead of pointing at
          pages they cannot open. The two links stay in link ink: they are real
          links inside a sentence, which is what that ink is for. */}
      {isDeploymentWide ? (
        <p className="max-w-[38.75rem] pb-5 text-sm text-muted">
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
        <p className="max-w-[38.75rem] pb-5 text-sm text-muted">
          These are your keys: requests on them are billed to you and spend
          against your budget.
        </p>
      )}

      {regenerated ? (
        <RegeneratedSecretDialog
          title={regenerated.title}
          result={regenerated.result}
          memberLabels={memberLabels}
          onClose={() => {
            setRegenerated(undefined)
            // Drop the one-time secret from mutation state so a later
            // regenerate never flashes the previous key.
            rotateKey.reset()
            createButtonRef.current?.focus()
          }}
        />
      ) : null}

      {showOnboarding ? (
        <EmptyState
          title="No API keys yet"
          description="An API key authenticates callers to this gateway. Create one to make your first request; the secret is shown once, so keep it somewhere safe."
          actionLabel="Create your first key"
          onAction={openCreate}
        />
      ) : null}

      <CreateKeyDialog
        key={openCount}
        isDeploymentWide={isDeploymentWide}
        isOpen={addOpen}
        onOpenChange={setAddOpen}
        memberLabels={memberLabels}
        returnFocusRef={createButtonRef}
      />
      {/* Keyed on the row id: the fields seed from `apiKey` on mount only, so
          the next Edit has to arrive at a fresh form rather than the last key's
          values, which would PATCH the wrong row. */}
      {editingKey ? (
        <EditKeyForm
          key={editingKey.id}
          isDeploymentWide={isDeploymentWide}
          apiKey={editingKey}
          onClose={() => setEditing(undefined)}
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
            variant="ghost"
            isDisabled={bulkPending}
            onPress={() =>
              void runBulk(selectedKeys, (apiKey) =>
                updateKey.mutateAsync({
                  id: apiKey.id,
                  body: { is_active: false },
                }),
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
              variant="ghost"
              isDisabled={bulkPending}
              onPress={() =>
                void runBulk(selectedKeys, (apiKey) =>
                  updateKey.mutateAsync({
                    id: apiKey.id,
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
      {showOnboarding ? null : layout === "mobile" ? (
        <section
          aria-label="API keys"
          className="otari-keys-list flex flex-col"
        >
          <div className="flex min-h-11 items-center gap-2 border-b border-border-subtle">
            <Checkbox
              hasTouchTarget
              ariaLabel="Select all keys"
              isSelected={rows.length > 0 && selectedIds.length === rows.length}
              onChange={(checked) =>
                selection.onSelectionChange(
                  checked ? new Set(selectableKeys) : new Set(),
                )
              }
            >
              <span className="sr-only">Select all keys</span>
            </Checkbox>
            <span className="flex-1 text-caption">
              {rows.length} {rows.length === 1 ? "key" : "keys"}
            </span>
            <span className="text-overline">Actions</span>
          </div>
          {/* The list owns loading too: falling through to the table here painted
              a seven-column skeleton on a phone and then swapped it for this. */}
          {loading ? (
            <EmptyMessage>
              <span className="inline-flex items-center gap-2">
                <Spinner size="sm" aria-hidden="true" /> Loading…
              </span>
            </EmptyMessage>
          ) : null}
          <ul className="flex flex-col">
            {rows.map((apiKey) => (
              <li
                key={apiKey.id}
                className={`flex items-start gap-2 border-b border-border-subtle py-3 ${selectedIds.includes(apiKey.id) ? "bg-primary-subtle" : ""}`}
              >
                <div className="flex min-h-11 shrink-0 items-center">
                  <Checkbox
                    hasTouchTarget
                    ariaLabel={`Select ${label(apiKey)}`}
                    isSelected={selectedIds.includes(apiKey.id)}
                    onChange={(checked) => {
                      const next = new Set(selectedIds)
                      if (checked) next.add(apiKey.id)
                      else next.delete(apiKey.id)
                      selection.onSelectionChange(next)
                    }}
                  >
                    <span className="sr-only">Select {label(apiKey)}</span>
                  </Checkbox>
                </div>
                <div className="flex min-w-0 flex-1 flex-col gap-1 pt-2">
                  <span className="truncate text-base text-foreground">
                    {apiKey.key_name ?? "(unnamed)"}
                  </span>
                  {renderFingerprint(apiKey)}
                  <div className="flex min-w-0 items-center gap-2">
                    <span className="shrink-0">
                      <StatusMark apiKey={apiKey} />
                    </span>
                    {isDeploymentWide ? (
                      <span className="truncate text-caption">
                        {ownerLabel(apiKey)}
                      </span>
                    ) : null}
                  </div>
                  <div className="truncate text-caption">
                    <KeyMetaLine apiKey={apiKey} face="text-caption" /> · used{" "}
                    {relative(apiKey.last_used_at) ?? "never"}
                  </div>
                </div>
                {renderActions(apiKey)}
              </li>
            ))}
          </ul>
        </section>
      ) : (
        <TableScrollFrame className="otari-keys-table">
          <DataTable
            ariaLabel="API keys"
            columns={visibleColumns}
            rows={rows}
            getRowKey={getKeyRowKey}
            isLoading={loading}
            emptyContent="No API keys yet. Create one to authenticate a caller."
            selectionMode="multiple"
            selectedKeys={selection.selectedKeys}
            onSelectionChange={selection.onSelectionChange}
          />
        </TableScrollFrame>
      )}

      <ConfirmDialog
        isOpen={pendingRegenerate !== undefined}
        onOpenChange={(open) => {
          if (!open) {
            setPendingRegenerate(undefined)
            rotateKey.reset()
          }
        }}
        heading="Regenerate API key"
        body={
          pendingRegenerate ? (
            <>
              Regenerate the secret for{" "}
              <strong>{label(pendingRegenerate)}</strong>? The current secret
              stops working immediately, with no grace period.
            </>
          ) : null
        }
        confirmLabel="Regenerate"
        isPending={rotateKey.isPending}
        error={rotateKey.error}
        onConfirm={() => {
          if (pendingRegenerate) regenerate(pendingRegenerate)
        }}
      />

      <ConfirmDialog
        isOpen={pendingDelete !== undefined}
        // Cleared on the way out: a refusal otherwise sits on the mutation and
        // greets the next row's confirm as if that row had failed.
        onOpenChange={(open) => {
          if (open) return
          setPendingDelete(undefined)
          deleteKey.reset()
        }}
        heading="Delete API key"
        body={
          pendingDelete ? (
            <>
              Permanently delete <strong>{label(pendingDelete)}</strong>? This
              removes the key and unlinks its usage history. Cannot be undone.
            </>
          ) : null
        }
        confirmLabel="Delete permanently"
        isPending={deleteKey.isPending}
        error={deleteKey.error}
        onConfirm={() => {
          if (!pendingDelete) return
          deleteKey.mutate(pendingDelete.id, {
            onSuccess: () => setPendingDelete(undefined),
          })
        }}
      />

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
            (apiKey) => deleteKey.mutateAsync(apiKey.id),
            () => setBulkDeleteOpen(false),
          )
        }
      />
    </div>
  )
}
