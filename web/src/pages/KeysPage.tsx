import { Button, Card, Chip } from "@heroui/react";
import { type KeyboardEvent as ReactKeyboardEvent, type ReactNode, useEffect, useRef, useState } from "react";

import { useCreateKey, useDeleteKey, useKeys, useRotateKey, useUpdateKey, useUsers } from "@/api/hooks";
import type { ApiKey, CreateKeyRequest, CreateKeyResponse, User } from "@/api/types";
import { Field } from "@/components/Field";
import { accessLabel, ModelScopeControl } from "@/components/ModelScopeControl";
import { UserComboBox } from "@/components/UserComboBox";
import { LoadingRow, Table, TableMessage, Td, Th, THead, Tr } from "@/components/Table";
import { ErrorBanner, InfoBanner, PageHeader } from "@/components/ui";

// ---------- helpers ----------

function absolute(iso: string | null): string {
  if (!iso) return "—";
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? "—" : d.toLocaleDateString();
}

function relative(iso: string | null): string | null {
  if (!iso) return null;
  const then = new Date(iso).getTime();
  if (Number.isNaN(then)) return null;
  const diffSec = Math.round((then - Date.now()) / 1000);
  const abs = Math.abs(diffSec);
  const units: [Intl.RelativeTimeFormatUnit, number][] = [
    ["day", 86_400],
    ["hour", 3_600],
    ["minute", 60],
  ];
  const rtf = new Intl.RelativeTimeFormat(undefined, { numeric: "auto" });
  for (const [unit, sec] of units) {
    if (abs >= sec) return rtf.format(Math.round(diffSec / sec), unit);
  }
  return rtf.format(diffSec, "second");
}

function isExpired(key: ApiKey): boolean {
  if (!key.expires_at) return false;
  const t = new Date(key.expires_at).getTime();
  return !Number.isNaN(t) && t < Date.now();
}

// datetime-local wants "YYYY-MM-DDTHH:mm" in local time; build it from an ISO value.
function toDatetimeLocal(iso: string | null): string {
  if (!iso) return "";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return "";
  const pad = (n: number) => String(n).padStart(2, "0");
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}T${pad(d.getHours())}:${pad(d.getMinutes())}`;
}

const isVirtualUser = (userId: string | null): boolean => (userId ?? "").startsWith("apikey-");

// ---------- copy field (graceful on non-HTTPS origins) ----------

// A readonly, always-selectable field with a copy button. The Clipboard API is
// undefined on the non-secure origins this dashboard is routinely served from, so
// the text is selected on click and Ctrl/Cmd-C always works even when the button
// cannot copy programmatically. "Copied" is only claimed when it truly copied.
function CopyField({
  label,
  value,
  multiline = false,
  fieldRef,
}: {
  label: string;
  value: string;
  multiline?: boolean;
  fieldRef?: React.RefObject<HTMLInputElement | HTMLTextAreaElement | null>;
}) {
  const internalRef = useRef<HTMLInputElement | HTMLTextAreaElement | null>(null);
  const ref = fieldRef ?? internalRef;
  const [copied, setCopied] = useState(false);
  const [selectHint, setSelectHint] = useState(false);

  const copy = async () => {
    ref.current?.focus();
    ref.current?.select();
    try {
      if (navigator.clipboard?.writeText) {
        await navigator.clipboard.writeText(value);
        setCopied(true);
        setSelectHint(false);
        window.setTimeout(() => setCopied(false), 2_000);
        return;
      }
    } catch {
      // fall through to the manual path
    }
    // No Clipboard API (or it threw): the text is selected, so the operator can
    // press Ctrl/Cmd-C. Never claim it was copied.
    setSelectHint(true);
  };

  const shared =
    "w-full rounded-lg border border-[var(--otari-line)] bg-[var(--otari-bg)] px-3 py-2 font-mono text-xs text-[var(--otari-ink)]";

  return (
    <div className="flex flex-col gap-1">
      <div className="flex items-center justify-between">
        <span className="text-xs font-medium text-[var(--otari-muted)]">{label}</span>
        <Button size="sm" variant="outline" onPress={copy}>
          {copied ? "Copied" : "Copy"}
        </Button>
      </div>
      {multiline ? (
        <textarea
          ref={ref as React.RefObject<HTMLTextAreaElement>}
          readOnly
          rows={value.split("\n").length}
          value={value}
          onFocus={(e) => e.currentTarget.select()}
          className={`${shared} resize-none whitespace-pre`}
        />
      ) : (
        <input
          ref={ref as React.RefObject<HTMLInputElement>}
          readOnly
          value={value}
          onFocus={(e) => e.currentTarget.select()}
          className={shared}
        />
      )}
      {/* Announce only the "Copied" event, never the secret itself. */}
      <span aria-live="polite" className="text-xs text-green-700">
        {copied ? "Copied to clipboard." : ""}
      </span>
      {selectHint ? (
        <span className="text-xs text-[var(--otari-muted)]">Selected. Press Ctrl/Cmd-C to copy.</span>
      ) : null}
    </div>
  );
}

// ---------- one-time reveal ----------

// The highest-stakes moment: the plaintext key is shown once. A focus-trapped
// dialog that cannot be dismissed by backdrop or Esc, only by an explicit "I've
// saved this key". Doubles as an activation moment: a copy-paste first call.
function RevealSecretModal({
  title,
  result,
  onClose,
}: {
  title: string;
  result: CreateKeyResponse;
  onClose: () => void;
}) {
  const dialogRef = useRef<HTMLDivElement>(null);
  const secretRef = useRef<HTMLInputElement | HTMLTextAreaElement | null>(null);
  const origin = typeof window !== "undefined" ? window.location.origin : "";
  const secret = result.key;

  useEffect(() => {
    // Focus the secret so Ctrl/Cmd-C works at once and a stray Enter doesn't land
    // on the close button.
    secretRef.current?.focus();
    secretRef.current?.select();
  }, []);

  const onKeyDown = (event: ReactKeyboardEvent<HTMLDivElement>) => {
    // Esc is intentionally ignored; closing is an explicit acknowledgement.
    if (event.key !== "Tab") return;
    const focusables = dialogRef.current?.querySelectorAll<HTMLElement>(
      'button, input, textarea, a[href], [tabindex]:not([tabindex="-1"])',
    );
    if (!focusables || focusables.length === 0) return;
    const first = focusables[0];
    const last = focusables[focusables.length - 1];
    if (event.shiftKey && document.activeElement === first) {
      event.preventDefault();
      last.focus();
    } else if (!event.shiftKey && document.activeElement === last) {
      event.preventDefault();
      first.focus();
    }
  };

  const curl = [
    `curl ${origin}/v1/chat/completions \\`,
    `  -H "Otari-Key: ${secret}" \\`,
    `  -H "Content-Type: application/json" \\`,
    `  -d '{"model": "your-model", "messages": [{"role": "user", "content": "Hello"}]}'`,
  ].join("\n");

  const python = [
    "from openai import OpenAI",
    "",
    `client = OpenAI(base_url="${origin}/v1", api_key="${secret}")`,
    "resp = client.chat.completions.create(",
    '    model="your-model",',
    '    messages=[{"role": "user", "content": "Hello"}],',
    ")",
    "print(resp.choices[0].message.content)",
  ].join("\n");

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 p-4" role="presentation">
      <div
        ref={dialogRef}
        role="dialog"
        aria-modal="true"
        aria-labelledby="reveal-title"
        onKeyDown={onKeyDown}
        className="flex max-h-[90vh] w-full max-w-2xl flex-col gap-4 overflow-y-auto rounded-xl bg-[var(--otari-surface)] p-6 shadow-xl"
      >
        <h2 id="reveal-title" className="text-lg font-semibold text-[var(--otari-ink)]">
          {title}
        </h2>
        <InfoBanner tone="warning">
          Copy this key now. For security it is shown only once and cannot be retrieved later. If you lose it, use
          Regenerate to issue a new secret.
        </InfoBanner>
        <p className="text-xs text-[var(--otari-muted)]">Model access: {accessLabel(result.allowed_models).text}.</p>
        <CopyField label="Secret key" value={secret} fieldRef={secretRef} />
        <div className="flex flex-col gap-2">
          <div>
            <div className="text-sm font-medium text-[var(--otari-ink)]">Make your first call</div>
            <p className="text-xs text-[var(--otari-muted)]">
              Replace <code>your-model</code> with a model from the Models page.
            </p>
          </div>
          <CopyField label="curl" value={curl} multiline />
          <CopyField label="Python (OpenAI SDK)" value={python} multiline />
        </div>
        <div className="flex justify-end">
          <Button variant="primary" onPress={onClose}>
            I&rsquo;ve saved this key
          </Button>
        </div>
      </div>
    </div>
  );
}

// ---------- inline confirm (names the target, no modal) ----------

function InlineConfirm({
  trigger,
  message,
  confirmLabel,
  isPending,
  onConfirm,
}: {
  trigger: string;
  message: ReactNode;
  confirmLabel: string;
  isPending?: boolean;
  onConfirm: () => void;
}) {
  const [armed, setArmed] = useState(false);

  if (!armed) {
    return (
      <Button size="sm" variant="danger-soft" onPress={() => setArmed(true)}>
        {trigger}
      </Button>
    );
  }

  return (
    <div className="flex flex-col items-end gap-1.5 rounded-lg border border-amber-200 bg-amber-50 p-2 text-right">
      <span className="max-w-xs text-xs text-amber-800">{message}</span>
      <span className="inline-flex gap-1">
        <Button size="sm" variant="danger" isDisabled={isPending} onPress={onConfirm}>
          {confirmLabel}
        </Button>
        <Button size="sm" variant="ghost" isDisabled={isPending} onPress={() => setArmed(false)}>
          Cancel
        </Button>
      </span>
    </div>
  );
}

// ---------- create / edit forms (inline cards, matching ProvidersPage) ----------

// Shows the selected owner's model access so the operator sees the ceiling this
// key narrows within (a key can inherit it or restrict to a subset, never exceed).
function OwnerAccessNote({ userId, users }: { userId: string; users: User[] }) {
  const id = userId.trim();
  if (id === "") {
    return (
      <p className="text-xs text-[var(--otari-muted)]">Choose an owner above to see the models this key can inherit.</p>
    );
  }
  const owner = users.find((u) => u.user_id === id);
  if (!owner) {
    return (
      <p className="text-xs text-[var(--otari-muted)]">
        New user <code>{id}</code> starts unrestricted, so this key may allow any model.
      </p>
    );
  }
  const { text } = accessLabel(owner.allowed_models);
  const entries = owner.allowed_models && owner.allowed_models.length > 0 ? owner.allowed_models.join(", ") : null;
  return (
    <p className="text-xs text-[var(--otari-muted)]">
      Owner <code>{id}</code> allows <span className="font-medium text-[var(--otari-ink)]">{text.toLowerCase()}</span>
      {entries ? (
        <>
          {" ("}
          <span className="font-mono">{entries}</span>
          {")"}
        </>
      ) : null}
      . This key inherits that, or narrows within it.
    </p>
  );
}

function CreateKeyForm({ onClose, onCreated }: { onClose: () => void; onCreated: (result: CreateKeyResponse) => void }) {
  const create = useCreateKey();
  const users = useUsers();
  const [keyName, setKeyName] = useState("");
  const [expiresAt, setExpiresAt] = useState("");
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [userId, setUserId] = useState("");
  const [allowedModels, setAllowedModels] = useState<string[] | null>(null);
  const [scopeValid, setScopeValid] = useState(true);

  const expiresInPast = expiresAt !== "" && new Date(expiresAt).getTime() < Date.now();
  // User-first: a key must name its owner (an existing user or a new id, which the
  // API creates as a named user). This is what keeps the dashboard from minting the
  // anonymous virtual users an omitted id would.
  const ownerMissing = userId.trim() === "";

  const submit = () => {
    if (create.isPending || !scopeValid || ownerMissing) return;
    const body: CreateKeyRequest = {
      key_name: keyName.trim() || null,
      user_id: userId.trim(),
      expires_at: expiresAt ? new Date(expiresAt).toISOString() : null,
      allowed_models: allowedModels,
    };
    create.mutate(body, {
      onSuccess: (result) => {
        // Capture the secret into the reveal BEFORE closing the form, so a render
        // hiccup can never drop the one-time key.
        onCreated(result);
        onClose();
      },
    });
  };

  return (
    <Card>
      <Card.Content className="flex flex-col gap-4 p-5">
        <div className="flex items-center justify-between">
          <div className="text-sm font-semibold text-[var(--otari-ink)]">Create API key</div>
          <Button size="sm" variant="ghost" onPress={onClose}>
            Close
          </Button>
        </div>
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
                <span className="text-red-700">That time is in the past; the key would be rejected immediately.</span>
              ) : (
                "Leave blank for a key that never expires."
              )
            }
          />
        </div>
        <UserComboBox value={userId} onChange={setUserId} users={users.data ?? []} />
        <button
          type="button"
          className="self-start text-xs font-medium text-[var(--otari-brand-dark)]"
          onClick={() => setShowAdvanced((v) => !v)}
        >
          {showAdvanced ? "Hide advanced" : "Advanced (restrict models)"}
        </button>
        {showAdvanced ? (
          <div className="flex flex-col gap-4 rounded-lg border border-[var(--otari-line)] p-4">
            <OwnerAccessNote userId={userId} users={users.data ?? []} />
            <ModelScopeControl
              title="Restrict this key's models"
              description="By default this key inherits its owner's access. Optionally narrow it to a subset; a key can never exceed its owner's allowed models."
              anyLabel="Inherit owner access"
              initial={null}
              onChange={(value, valid) => {
                setAllowedModels(value);
                setScopeValid(valid);
              }}
            />
          </div>
        ) : null}
        <div>
          <Button variant="primary" isDisabled={create.isPending || !scopeValid || ownerMissing} onPress={submit}>
            {create.isPending ? "Creating…" : "Create key"}
          </Button>
        </div>
      </Card.Content>
    </Card>
  );
}

function EditKeyForm({ apiKey, onClose }: { apiKey: ApiKey; onClose: () => void }) {
  const update = useUpdateKey();
  const users = useUsers();
  const [keyName, setKeyName] = useState(apiKey.key_name ?? "");
  const [expiresAt, setExpiresAt] = useState(toDatetimeLocal(apiKey.expires_at));
  const [allowedModels, setAllowedModels] = useState<string[] | null>(apiKey.allowed_models);
  const [scopeValid, setScopeValid] = useState(true);

  const submit = () => {
    if (update.isPending || !scopeValid) return;
    update.mutate(
      {
        id: apiKey.id,
        body: {
          key_name: keyName.trim() || null,
          expires_at: expiresAt ? new Date(expiresAt).toISOString() : null,
          allowed_models: allowedModels,
        },
      },
      { onSuccess: onClose },
    );
  };

  return (
    <Card>
      <Card.Content className="flex flex-col gap-4 p-5">
        <div className="text-sm font-semibold text-[var(--otari-ink)]">
          Edit <code>{apiKey.key_name ?? apiKey.id}</code>
        </div>
        <ErrorBanner error={update.error} />
        <div className="grid gap-4 sm:grid-cols-2">
          <Field label="Name" value={keyName} onChange={setKeyName} placeholder="ci-bot" />
          <Field
            label="Expires"
            value={expiresAt}
            onChange={setExpiresAt}
            type="datetime-local"
            description="Blank clears the expiry."
          />
        </div>
        {apiKey.user_id ? <OwnerAccessNote userId={apiKey.user_id} users={users.data ?? []} /> : null}
        <ModelScopeControl
          title="Restrict this key's models"
          description="This key inherits its owner's access by default. Narrow it to a subset here; it can never exceed the owner's allowed models."
          anyLabel="Inherit owner access"
          initial={apiKey.allowed_models}
          onChange={(value, valid) => {
            setAllowedModels(value);
            setScopeValid(valid);
          }}
        />
        <div className="flex gap-2">
          <Button variant="primary" isDisabled={update.isPending || !scopeValid} onPress={submit}>
            {update.isPending ? "Saving…" : "Save changes"}
          </Button>
          <Button variant="ghost" onPress={onClose}>
            Cancel
          </Button>
        </div>
      </Card.Content>
    </Card>
  );
}

// ---------- status + rows ----------

function StatusChip({ apiKey }: { apiKey: ApiKey }) {
  if (!apiKey.is_active) {
    return (
      <Chip size="sm" color="default">
        Disabled
      </Chip>
    );
  }
  if (isExpired(apiKey)) {
    return (
      <Chip size="sm" color="warning">
        Expired
      </Chip>
    );
  }
  return (
    <Chip size="sm" color="accent">
      Active
    </Chip>
  );
}

function AccessChip({ allowed }: { allowed: string[] | null }) {
  const { text, tone } = accessLabel(allowed);
  const cls =
    tone === "danger"
      ? "text-red-700 font-medium"
      : tone === "muted"
        ? "text-[var(--otari-muted)]"
        : "text-[var(--otari-brand-dark)] font-medium";
  // Surface the exact entries on hover; the count would mislead (a wildcard is many).
  const title = allowed && allowed.length > 0 ? allowed.join(", ") : undefined;
  return (
    <span className={`text-xs ${cls}`} title={title}>
      {text}
    </span>
  );
}

function OnboardingPanel({ onCreate }: { onCreate: () => void }) {
  return (
    <Card>
      <Card.Content className="flex flex-col gap-4 p-6">
        <div>
          <h2 className="text-lg font-semibold text-[var(--otari-ink)]">No API keys yet</h2>
          <p className="mt-1 text-sm text-[var(--otari-muted)]">
            An API key authenticates callers to this gateway. Create one to make your first request; the secret is shown
            once, so keep it somewhere safe.
          </p>
        </div>
        <div>
          <Button variant="primary" onPress={onCreate}>
            Create your first key
          </Button>
        </div>
      </Card.Content>
    </Card>
  );
}

export function KeysPage() {
  const keys = useKeys();
  const updateKey = useUpdateKey();
  const rotateKey = useRotateKey();
  const deleteKey = useDeleteKey();

  const [addOpen, setAddOpen] = useState(false);
  const [editing, setEditing] = useState<string | null>(null);
  const [revealed, setRevealed] = useState<{ title: string; result: CreateKeyResponse } | null>(null);

  const rows = keys.data ?? [];
  const loading = keys.isLoading;
  const editingKey = rows.find((k) => k.id === editing) ?? null;
  const showOnboarding = !loading && rows.length === 0 && !addOpen;

  const label = (k: ApiKey) => k.key_name ?? k.id;

  const setActive = (k: ApiKey, active: boolean) => updateKey.mutate({ id: k.id, body: { is_active: active } });

  const regenerate = (k: ApiKey) =>
    rotateKey.mutate(k.id, {
      onSuccess: (result) => setRevealed({ title: `New secret for ${label(k)}`, result }),
    });

  return (
    <div className="flex flex-col gap-6">
      <PageHeader
        title="API keys"
        description="Issue and revoke the keys that authenticate callers to this gateway. Secrets are shown once at creation."
        action={
          addOpen ? null : (
            <Button
              variant="primary"
              onPress={() => {
                setEditing(null);
                setAddOpen(true);
              }}
            >
              Create key
            </Button>
          )
        }
      />

      <ErrorBanner error={keys.error ?? updateKey.error ?? rotateKey.error ?? deleteKey.error} />

      {showOnboarding ? (
        <OnboardingPanel
          onCreate={() => {
            setEditing(null);
            setAddOpen(true);
          }}
        />
      ) : null}

      {addOpen ? (
        <CreateKeyForm
          onClose={() => setAddOpen(false)}
          onCreated={(result) => setRevealed({ title: "API key created", result })}
        />
      ) : null}
      {/* Key on the row id so switching which key is edited remounts the form:
          its fields seed from `apiKey` via useState (mount-only), so without this
          a second Edit would keep the first key's values and PATCH the wrong row. */}
      {editingKey ? <EditKeyForm key={editingKey.id} apiKey={editingKey} onClose={() => setEditing(null)} /> : null}

      <Table>
        <THead>
          <tr>
            <Th>Name</Th>
            <Th>Status</Th>
            <Th>Owner</Th>
            <Th>Key</Th>
            <Th>Created</Th>
            <Th>Last used</Th>
            <Th>Expires</Th>
            <Th className="text-right">Actions</Th>
          </tr>
        </THead>
        <tbody>
          {loading ? (
            <LoadingRow colSpan={8} />
          ) : rows.length === 0 ? (
            <TableMessage colSpan={8}>No API keys yet. Create one to authenticate a caller.</TableMessage>
          ) : (
            rows.map((k) => (
              <Tr
                key={k.id}
                selected={editing === k.id}
                onClick={() => {
                  setAddOpen(false);
                  setEditing(k.id);
                }}
              >
                <Td className="font-medium text-[var(--otari-ink)]">
                  <div className="flex flex-col gap-0.5">
                    <span>{k.key_name ?? <span className="text-[var(--otari-muted)]">(unnamed)</span>}</span>
                    <AccessChip allowed={k.allowed_models} />
                  </div>
                </Td>
                <Td>
                  <StatusChip apiKey={k} />
                </Td>
                <Td className="text-[var(--otari-muted)]">
                  {isVirtualUser(k.user_id) ? (
                    <span className="inline-flex items-center gap-1.5">
                      <Chip size="sm" color="default">
                        virtual
                      </Chip>
                    </span>
                  ) : (
                    <code className="text-xs">{k.user_id ?? "—"}</code>
                  )}
                </Td>
                <Td>
                  <code className="text-xs text-[var(--otari-muted)]">{k.key_prefix ? `${k.key_prefix}…` : "—"}</code>
                </Td>
                <Td className="text-[var(--otari-muted)]">{absolute(k.created_at)}</Td>
                <Td className="text-[var(--otari-muted)]">{relative(k.last_used_at) ?? "never"}</Td>
                <Td className="text-[var(--otari-muted)]">
                  <span title={k.expires_at ? new Date(k.expires_at).toLocaleString() : undefined}>
                    {k.expires_at ? absolute(k.expires_at) : "never"}
                  </span>
                </Td>
                <Td>
                  {/* Row click opens Edit; the action buttons stop propagation so
                      they fire their own handler instead of also opening Edit. */}
                  <div className="flex items-center justify-end gap-1.5" onClick={(e) => e.stopPropagation()}>
                    {k.is_active ? (
                      <Button size="sm" variant="outline" isDisabled={updateKey.isPending} onPress={() => setActive(k, false)}>
                        Disable
                      </Button>
                    ) : (
                      <Button size="sm" variant="outline" isDisabled={updateKey.isPending} onPress={() => setActive(k, true)}>
                        Enable
                      </Button>
                    )}
                    <Button
                      size="sm"
                      variant="ghost"
                      onPress={() => {
                        setAddOpen(false);
                        setEditing(k.id);
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
                          Regenerate the secret for <strong>{label(k)}</strong>? The current secret stops working
                          immediately, with no grace period.
                        </>
                      }
                      onConfirm={() => regenerate(k)}
                    />
                    {/* Permanent delete is only offered once a key is disabled, so a
                        live caller can't be broken (and its audit trail erased) in one
                        click. Disable is the reversible revoke. */}
                    {k.is_active ? null : (
                      <InlineConfirm
                        trigger="Delete"
                        confirmLabel="Delete permanently"
                        isPending={deleteKey.isPending}
                        message={
                          <>
                            Permanently delete <strong>{label(k)}</strong>? This removes the key and unlinks its usage
                            history. Cannot be undone.
                          </>
                        }
                        onConfirm={() => deleteKey.mutate(k.id)}
                      />
                    )}
                  </div>
                </Td>
              </Tr>
            ))
          )}
        </tbody>
      </Table>

      {revealed ? (
        <RevealSecretModal
          title={revealed.title}
          result={revealed.result}
          onClose={() => {
            setRevealed(null);
            // Drop the one-time secret from mutation state so reopening Create/Regenerate
            // never flashes the previous key.
            rotateKey.reset();
          }}
        />
      ) : null}
    </div>
  );
}
