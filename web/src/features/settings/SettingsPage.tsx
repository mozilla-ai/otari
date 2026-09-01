import { AlertDialog, Button, buttonVariants, Input } from "@heroui/react"
import { useEffect, useRef, useState } from "react"
import type { ConfigField, UpdateSettingsRequest } from "@/client"
import { MailDeliveryCard } from "@/features/settings/MailDeliveryCard"
import { MaintenanceModeCard } from "@/features/settings/MaintenanceModeCard"
import { Toggle } from "@/features/settings/Toggle"
import {
  useReencryptProviderCredentials,
  useRotateMasterKey,
  useSettings,
  useStoredProviders,
  useUpdateSettings,
} from "@/shared/api/hooks"
import { PageIntro, SettingsGroup } from "@/shared/components/surface"
import {
  ErrorBanner,
  FilterSelect,
  InfoBanner,
  PageLoading,
} from "@/shared/components/ui"

// A single settable field maps onto one key of UpdateSettingsRequest. The keys
// come from the backend's `settable` marking, so cast at this one boundary.
function settableUpdate(
  key: string,
  value: boolean | number | string | null,
): UpdateSettingsRequest {
  return { [key]: value } as UpdateSettingsRequest
}

// Whether `needle` appears as an ordered subsequence of `haystack` (fuzzy match).
function isSubsequence(needle: string, haystack: string): boolean {
  let i = 0
  for (const char of haystack) {
    if (char === needle[i]) i += 1
    if (i === needle.length) return true
  }
  return needle.length === 0
}

// A field matches the query when every whitespace-separated term is either a
// substring of the field's key/description/group or a fuzzy subsequence of its
// key (so "mctts" finds "model_cache_ttl_seconds"). An empty query matches all.
export function fieldMatches(field: ConfigField, query: string): boolean {
  const q = query.trim().toLowerCase()
  if (q === "") return true
  const haystack =
    `${field.key} ${field.description ?? ""} ${field.group}`.toLowerCase()
  const key = field.key.toLowerCase().replace(/[^a-z0-9]/g, "")
  return q
    .split(/\s+/)
    .every((term) => haystack.includes(term) || isSubsequence(term, key))
}

// A numeric setting (int or float) with an explicit Save, so a mistyped value is
// not applied on every keystroke. The draft resyncs whenever the committed value
// changes (after a save round-trip).
function NumberSetting({
  field,
  onSave,
  disabled,
}: {
  field: ConfigField
  onSave: (value: number) => void
  disabled?: boolean
}) {
  const committed = typeof field.value === "number" ? field.value : 0
  const [draft, setDraft] = useState(String(committed))
  const isFloat = field.type === "float"

  useEffect(() => {
    setDraft(String(committed))
  }, [committed])

  const parsed = Number(draft)
  const wellFormed =
    draft.trim() !== "" &&
    Number.isFinite(parsed) &&
    (isFloat || Number.isInteger(parsed))
  // Gate against the field's own lower bound so a gt=0 field disables Save at 0
  // rather than round-tripping to a 422. Falls back to >= 0 when no bound is
  // declared (every current numeric field declares one).
  const ge = field.minimum ?? undefined
  const gt = field.exclusive_minimum ?? undefined
  const withinBounds =
    gt !== undefined
      ? parsed > gt
      : ge !== undefined
        ? parsed >= ge
        : parsed >= 0
  const valid = wellFormed && withinBounds
  const changed = valid && parsed !== committed

  return (
    <div className="flex items-center gap-2">
      <Input
        type="number"
        min="0"
        step={isFloat ? "any" : "1"}
        inputMode={isFloat ? "decimal" : "numeric"}
        aria-label={field.key}
        value={draft}
        disabled={disabled}
        onChange={(event) => setDraft(event.target.value)}
        className="w-28 rounded-md border border-field-border bg-field px-2 py-1 text-right text-sm tabular-nums focus:border-accent focus:outline-none disabled:opacity-50"
      />
      <Button
        size="sm"
        variant="primary"
        aria-label={`Save ${field.key}`}
        isDisabled={disabled || !changed}
        onPress={() => onSave(parsed)}
      >
        Save
      </Button>
    </div>
  )
}

// A free-text setting (a nullable string, e.g. vision_describe_model). Empty
// input clears the value to null. Saves only on Save, like the numeric control.
function TextSetting({
  field,
  onSave,
  disabled,
}: {
  field: ConfigField
  onSave: (value: string | null) => void
  disabled?: boolean
}) {
  const committed = typeof field.value === "string" ? field.value : ""
  const [draft, setDraft] = useState(committed)

  useEffect(() => {
    setDraft(committed)
  }, [committed])

  const changed = draft !== committed

  return (
    <div className="flex items-center gap-2">
      <input
        type="text"
        aria-label={field.key}
        value={draft}
        disabled={disabled}
        placeholder="unset"
        onChange={(event) => setDraft(event.target.value)}
        className="w-56 rounded-md border border-field-border bg-field px-2 py-1 text-sm focus:border-accent focus:outline-none disabled:opacity-50"
      />
      <Button
        size="sm"
        variant="primary"
        aria-label={`Save ${field.key}`}
        isDisabled={disabled || !changed}
        onPress={() => onSave(draft.trim() === "" ? null : draft)}
      >
        Save
      </Button>
    </div>
  )
}

// Format a read-only value for display, distinguishing an unset field from an
// empty list and giving booleans an on/off reading.
function formatValue(field: ConfigField): string {
  const { value } = field
  if (value === null || value === undefined) {
    return "unset"
  }
  if (typeof value === "boolean") {
    return value ? "on" : "off"
  }
  if (Array.isArray(value)) {
    return value.length ? value.join(", ") : "none"
  }
  return String(value)
}

// The right-hand control for one field: an interactive control when settable
// (hot-changeable), otherwise the value plus a "startup-only" marker.
function SettingControl({
  field,
  patch,
  disabled,
}: {
  field: ConfigField
  patch: (body: UpdateSettingsRequest) => void
  disabled: boolean
}) {
  if (!field.settable) {
    return (
      <div className="flex items-center gap-2">
        <span className="text-sm tabular-nums text-foreground">
          {formatValue(field)}
        </span>
        {/* A label, not a chip. The pill it used to be was a bordered capsule
            next to a value on a surface that no longer has boxes on it, so the
            shape was doing the work the type should. `text-overline` is exactly
            that role and already carries the size, weight, tracking, uppercase
            and muted color, so this is the role rather than a seventh hand-rolled
            spelling of it. */}
        <span className="text-overline">startup-only</span>
      </div>
    )
  }

  if (field.type === "bool") {
    return (
      <Toggle
        checked={field.value === true}
        onChange={(next) => patch(settableUpdate(field.key, next))}
        label={field.key}
        disabled={disabled}
      />
    )
  }

  if (field.options && field.options.length > 0) {
    return (
      <FilterSelect
        ariaLabel={field.key}
        value={String(field.value ?? "")}
        onChange={(next) => patch(settableUpdate(field.key, next))}
        options={field.options.map((option) => ({
          value: option,
          label: option,
        }))}
      />
    )
  }

  if (field.type === "int" || field.type === "float") {
    return (
      <NumberSetting
        field={field}
        onSave={(value) => patch(settableUpdate(field.key, value))}
        disabled={disabled}
      />
    )
  }

  // A settable string without a fixed option set: a free-text (nullable) value.
  return (
    <TextSetting
      field={field}
      onSave={(value) => patch(settableUpdate(field.key, value))}
      disabled={disabled}
    />
  )
}

function ConfigRow({
  field,
  patch,
  disabled,
}: {
  field: ConfigField
  patch: (body: UpdateSettingsRequest) => void
  disabled: boolean
}) {
  return (
    <div className="flex items-start justify-between gap-6 py-4">
      <div className="min-w-0">
        <code className="text-sm font-medium text-foreground">{field.key}</code>
        {field.description ? (
          // `max-w-prose` for the reason the page header carries one: these
          // rows became full-bleed with the rest of the page, and the longest
          // description here measured 1211px, about 175 characters to the line,
          // roughly twice a readable measure. The row still spans the page; the
          // sentence inside it does not have to.
          <p className="mt-1 max-w-prose text-sm text-muted">
            {field.description}
          </p>
        ) : null}
      </div>
      <div className="shrink-0 pt-0.5">
        <SettingControl field={field} patch={patch} disabled={disabled} />
      </div>
    </div>
  )
}

function CopyField({
  value,
  fieldRef,
}: {
  value: string
  fieldRef?: React.RefObject<HTMLInputElement | null>
}) {
  const internalRef = useRef<HTMLInputElement>(null)
  const ref = fieldRef ?? internalRef
  const [copied, setCopied] = useState(false)
  const [selectHint, setSelectHint] = useState(false)

  const copy = async () => {
    ref.current?.focus()
    ref.current?.select()
    try {
      if (navigator.clipboard?.writeText) {
        await navigator.clipboard.writeText(value)
        setCopied(true)
        setSelectHint(false)
        window.setTimeout(() => setCopied(false), 2_000)
        return
      }
    } catch {
      // Fall through to the manual-copy hint.
    }
    setSelectHint(true)
  }

  return (
    <div className="flex flex-col gap-1">
      <div className="flex items-center justify-between">
        <span className="text-xs font-medium text-muted">New master key</span>
        <Button size="sm" variant="outline" onPress={copy}>
          {copied ? "Copied" : "Copy"}
        </Button>
      </div>
      <input
        ref={ref}
        readOnly
        value={value}
        onFocus={(event) => event.currentTarget.select()}
        autoComplete="off"
        autoCorrect="off"
        autoCapitalize="off"
        spellCheck={false}
        data-1p-ignore
        data-lpignore="true"
      />
      <span aria-live="polite" className="text-xs text-success">
        {copied ? "Copied to clipboard." : ""}
      </span>
      {selectHint ? (
        <span className="text-xs text-muted">
          Selected. Press Ctrl/Cmd-C to copy.
        </span>
      ) : null}
    </div>
  )
}

function MasterKeyRotationDialog({
  masterKey,
  error,
  isPending,
  onRegenerate,
  onClose,
}: {
  masterKey?: string
  error: Error | null
  isPending: boolean
  onRegenerate: () => void
  onClose: () => void
}) {
  const keyRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    if (masterKey === undefined) return
    keyRef.current?.focus()
    keyRef.current?.select()
  }, [masterKey])

  return (
    <AlertDialog.Backdrop>
      <AlertDialog.Container placement="center" size="lg">
        <AlertDialog.Dialog>
          <AlertDialog.Header>
            <AlertDialog.Heading>
              {masterKey !== undefined
                ? "Master key regenerated"
                : "Regenerate master key?"}
            </AlertDialog.Heading>
          </AlertDialog.Header>
          <AlertDialog.Body className="flex flex-col gap-4">
            {masterKey !== undefined ? (
              <>
                <InfoBanner tone="warning">
                  Copy this key now. It is shown once and cannot be retrieved
                  again after you close this dialog.
                </InfoBanner>
                <p className="text-sm text-muted">
                  The previous master key has stopped working. This browser tab
                  now uses the new key.
                </p>
                <CopyField value={masterKey} fieldRef={keyRef} />
              </>
            ) : (
              <>
                <InfoBanner tone="warning">
                  This immediately invalidates the current dashboard master key.
                  Other signed-in dashboard sessions will need the new key to
                  continue.
                </InfoBanner>
                <p className="text-sm text-muted">
                  The replacement key will be shown once. Save it before closing
                  the next screen.
                </p>
                <ErrorBanner error={error} />
              </>
            )}
          </AlertDialog.Body>
          <AlertDialog.Footer>
            {masterKey !== undefined ? (
              <Button variant="primary" onPress={onClose}>
                I&rsquo;ve saved this key
              </Button>
            ) : (
              <>
                <Button
                  variant="ghost"
                  isDisabled={isPending}
                  onPress={onClose}
                >
                  Cancel
                </Button>
                <Button
                  variant="danger"
                  isPending={isPending}
                  onPress={onRegenerate}
                >
                  Regenerate key
                </Button>
              </>
            )}
          </AlertDialog.Footer>
        </AlertDialog.Dialog>
      </AlertDialog.Container>
    </AlertDialog.Backdrop>
  )
}

function MasterKeyRow({ source }: { source: "configured" | "generated" }) {
  const rotateMasterKey = useRotateMasterKey()
  const [dialogOpen, setDialogOpen] = useState(false)
  const [newKey, setNewKey] = useState<string | undefined>()
  const isGenerated = source === "generated"

  // Rotation revokes every dashboard session server-side and re-mints this
  // tab's session cookie on the response, so no client-side credential swap
  // is needed; the dialog only has to reveal the new key once.
  const rotate = () =>
    rotateMasterKey.mutate(undefined, {
      onSuccess: (result) => {
        setNewKey(result.master_key)
      },
    })

  const closeDialog = () => {
    setDialogOpen(false)
    setNewKey(undefined)
    rotateMasterKey.reset()
  }

  const onOpenChange = (isOpen: boolean) => {
    if (isOpen) {
      rotateMasterKey.reset()
      setDialogOpen(true)
    } else if (newKey === undefined) {
      closeDialog()
    }
  }

  return (
    <div className="flex flex-col gap-4 py-4">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="min-w-0">
          <code className="text-sm font-medium text-foreground">
            master_key
          </code>
          <p className="mt-1 max-w-3xl text-sm text-muted">
            {isGenerated
              ? "This gateway uses its first-run generated dashboard key. Regeneration invalidates the current key immediately."
              : "This gateway uses a key managed through OTARI_MASTER_KEY or config.yml. Rotate it in configuration, then restart the gateway."}
          </p>
        </div>
        <AlertDialog isOpen={dialogOpen} onOpenChange={onOpenChange}>
          {isGenerated ? (
            <AlertDialog.Trigger
              className={buttonVariants({ size: "sm", variant: "danger-soft" })}
            >
              Regenerate
            </AlertDialog.Trigger>
          ) : (
            <Button size="sm" variant="danger-soft" isDisabled>
              Managed in configuration
            </Button>
          )}
          {dialogOpen ? (
            <MasterKeyRotationDialog
              masterKey={newKey}
              error={rotateMasterKey.error}
              isPending={rotateMasterKey.isPending}
              onRegenerate={rotate}
              onClose={closeDialog}
            />
          ) : null}
        </AlertDialog>
      </div>
    </div>
  )
}

function SecretKeyRow() {
  const storedProviders = useStoredProviders()
  const reencrypt = useReencryptProviderCredentials()
  const result = reencrypt.data
  const storedCount = storedProviders.data?.length ?? 0
  const unreadableCount = (storedProviders.data ?? []).filter(
    (provider) => !provider.decryptable,
  ).length
  const hasStoredKeys = storedCount > 0

  return (
    <div className="flex flex-col gap-4 py-4">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="min-w-0">
          <code className="text-sm font-medium text-foreground">
            OTARI_SECRET_KEY
          </code>
          <p className="mt-1 max-w-3xl text-sm text-muted">
            Generate a new key with <code>uv run otari gen-secret-key</code>,
            then restart with{" "}
            <code>OTARI_SECRET_KEY=&lt;new-key&gt;,&lt;old-key&gt;</code>.
            Re-encrypt the stored provider keys, then restart with{" "}
            <code>OTARI_SECRET_KEY=&lt;new-key&gt;</code> once none are
            unreadable.
          </p>
        </div>
        <div className="shrink-0">
          <Button
            size="sm"
            variant="outline"
            isDisabled={!hasStoredKeys || reencrypt.isPending}
            onPress={() => reencrypt.mutate()}
          >
            {reencrypt.isPending
              ? "Re-encrypting…"
              : "Re-encrypt provider keys"}
          </Button>
        </div>
      </div>
      <ErrorBanner error={storedProviders.error ?? reencrypt.error} />
      {unreadableCount > 0 ? (
        <InfoBanner tone="warning">
          {unreadableCount} stored provider key
          {unreadableCount === 1 ? "" : "s"} cannot be decrypted with the
          current <code>OTARI_SECRET_KEY</code>. Restore the old secret key and
          re-encrypt, or edit each affected provider and replace its key.
        </InfoBanner>
      ) : null}
      {result ? (
        <p className="text-sm text-muted" role="status" aria-live="polite">
          Re-encrypted {result.reencrypted} provider key
          {result.reencrypted === 1 ? "" : "s"}.
          {result.unreadable > 0
            ? ` ${result.unreadable} still need replacement.`
            : " All decryptable stored keys now use the primary secret key."}
        </p>
      ) : !storedProviders.isLoading && !hasStoredKeys ? (
        <p className="text-sm text-muted">
          No stored provider keys need re-encryption.
        </p>
      ) : null}
    </div>
  )
}

function SecurityKeysSection({
  masterKeySource,
}: {
  masterKeySource: "configured" | "generated"
}) {
  return (
    <SettingsGroup title="Credential security" count={2}>
      <MasterKeyRow source={masterKeySource} />
      <SecretKeyRow />
    </SettingsGroup>
  )
}

// Group fields by their group label, preserving first-seen order. Uses a map (not
// a consecutive-run merge) so a group name is never emitted twice even if the
// fields for it are not contiguous, which would otherwise collide React keys.
function groupFields(
  fields: ConfigField[],
): { name: string; fields: ConfigField[] }[] {
  const order: { name: string; fields: ConfigField[] }[] = []
  const byName = new Map<string, { name: string; fields: ConfigField[] }>()
  for (const field of fields) {
    let group = byName.get(field.group)
    if (!group) {
      group = { name: field.group, fields: [] }
      byName.set(field.group, group)
      order.push(group)
    }
    group.fields.push(field)
  }
  return order
}

export function SettingsPage() {
  const settings = useSettings()
  const updateSettings = useUpdateSettings()

  const data = settings.data
  const pending = updateSettings.isPending

  const [search, setSearch] = useState("")
  const [settableOnly, setSettableOnly] = useState(false)
  const searchRef = useRef<HTMLInputElement>(null)

  // "/" focuses the search box (a common shortcut for filter-heavy pages),
  // unless the user is already typing in a field.
  useEffect(() => {
    function onKeyDown(event: KeyboardEvent) {
      const target = event.target as HTMLElement | null
      const typing =
        target &&
        (target.tagName === "INPUT" ||
          target.tagName === "TEXTAREA" ||
          target.tagName === "SELECT")
      if (event.key === "/" && !typing) {
        event.preventDefault()
        searchRef.current?.focus()
      }
    }
    window.addEventListener("keydown", onKeyDown)
    return () => window.removeEventListener("keydown", onKeyDown)
  }, [])

  const patch = (body: UpdateSettingsRequest) => updateSettings.mutate(body)

  const allFields = data?.config ?? []
  const filtered = allFields.filter(
    (field) =>
      (settableOnly ? field.settable : true) && fieldMatches(field, search),
  )
  const groups = groupFields(filtered)

  return (
    <div className="flex flex-col">
      <PageIntro title="Settings">
        Every effective gateway setting. Settable fields apply immediately and
        persist across restarts; startup-only fields are shown for reference and
        change only via config.yml or environment variables (then a restart).
      </PageIntro>

      <ErrorBanner error={settings.error ?? updateSettings.error} />

      <div className="flex flex-wrap items-center gap-3 pb-3">
        <input
          ref={searchRef}
          type="search"
          aria-label="Search settings"
          placeholder="Search settings (press / to focus)…"
          value={search}
          onChange={(event) => setSearch(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "Escape") setSearch("")
          }}
          className="min-w-0 flex-1 rounded-lg border border-border bg-surface-alt px-3 py-2 text-sm text-foreground focus:border-accent focus:outline-none"
        />
        <label className="flex items-center gap-2 text-sm text-muted">
          <input
            type="checkbox"
            checked={settableOnly}
            onChange={(event) => setSettableOnly(event.target.checked)}
            className="h-4 w-4 accent-accent"
          />
          Settable only
        </label>
      </div>

      {data ? (
        <p className="pb-2 text-xs text-subtle">
          Showing {filtered.length} of {allFields.length} settings
        </p>
      ) : null}

      {data && filtered.length === 0 ? (
        <p className="text-sm text-muted">No settings match your search.</p>
      ) : null}

      {settings.isLoading ? <PageLoading /> : null}

      {groups.map((group) => (
        <SettingsGroup
          key={group.name}
          title={group.name}
          count={group.fields.length}
        >
          {group.fields.map((field) => (
            <ConfigRow
              key={field.key}
              field={field}
              patch={patch}
              disabled={!data || pending}
            />
          ))}
        </SettingsGroup>
      ))}

      {data ? (
        <SecurityKeysSection masterKeySource={data.master_key_source} />
      ) : null}

      <MailDeliveryCard />

      <MaintenanceModeCard />

      {data ? (
        <p className="pt-6 text-xs text-subtle">
          Mode: {data.mode} · Version {data.version}
          {data.require_pricing ? " · require_pricing on" : ""}
        </p>
      ) : null}
    </div>
  )
}
