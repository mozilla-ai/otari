import { Button, Card, Chip } from "@heroui/react"
import { useState } from "react"

import type {
  CreateOrgProviderKeyRequest,
  OrgProviderKey,
  UpdateOrgProviderKeyRequest,
} from "@/client"
import {
  ClientArgsField,
  formatClientArgs,
  ProviderComboBox,
  parseClientArgs,
} from "@/features/providers/providerFields"
import {
  useArchiveOrgProviderKey,
  useCreateOrgProviderKey,
  useDeleteOrgProviderKey,
  useOrganizationContext,
  useOrgProviderKeys,
  useRestoreOrgProviderKey,
  useSetOrgProviderKeyDefault,
  useSettings,
  useUpdateOrgProviderKey,
} from "@/shared/api/hooks"
import { DataTable, type DataTableColumn } from "@/shared/components/DataTable"
import { Field } from "@/shared/components/Field"
import { SecretField } from "@/shared/components/SecretField"
import {
  Checkbox,
  ConfirmButton,
  ErrorBanner,
  InfoBanner,
  PageHeader,
} from "@/shared/components/ui"
import { formatRelative } from "@/shared/helpers/format"

import { canManage } from "./roles"

// The organization's own upstream credentials: one BYO key per provider that
// every workspace under the tenant inherits.
//
// Not the workspace rail's `/providers`, which manages `provider_credentials`,
// keyed on an instance name and therefore owned by the process rather than by
// anyone in particular. The two pages look alike and are not the same thing, so
// a deployment shows one or the other: `organization_providers` is reported by a
// hosted deployment and `providers` by a standalone one
// (`STANDALONE_SURFACES` / `HOSTED_SURFACES` in
// `src/gateway/api/routes/bootstrap.py`).
//
// What is deliberately absent is the per-workspace half of the same API
// (`/v1/workspaces/{id}/provider-keys`: pin, disable, restrict to models). Those
// are one workspace's departure from what this page sets, so they belong beside
// that workspace rather than here, and the organization view would have to ask
// "which workspace" before it could show any of it.

/** A key's editable fields, seeded from a row when one is being edited. */
interface KeyDraft {
  provider: string
  name: string
  apiKey: string
  apiBase: string
  clientArgs: string
}

const EMPTY_DRAFT: KeyDraft = {
  provider: "",
  name: "",
  apiKey: "",
  apiBase: "",
  clientArgs: "",
}

function draftFrom(key: OrgProviderKey): KeyDraft {
  return {
    provider: key.provider,
    name: key.name,
    // Never prefilled: the gateway stores the ciphertext and returns `last4`,
    // so there is nothing to prefill with. Blank on save means "leave it".
    apiKey: "",
    apiBase: key.api_base ?? "",
    clientArgs: formatClientArgs(key.client_args),
  }
}

function KeyForm({
  editing,
  onClose,
}: {
  /** The key being edited, or null when the form is creating one. */
  editing: OrgProviderKey | null
  onClose: () => void
}) {
  const create = useCreateOrgProviderKey()
  const update = useUpdateOrgProviderKey()
  const [draft, setDraft] = useState<KeyDraft>(() =>
    editing ? draftFrom(editing) : EMPTY_DRAFT,
  )

  const parsedClientArgs = parseClientArgs(draft.clientArgs)
  const clientArgsError = parsedClientArgs.ok ? null : parsedClientArgs.error
  const pending = create.isPending || update.isPending
  const canSubmit =
    parsedClientArgs.ok &&
    draft.name.trim() !== "" &&
    (editing !== null || draft.provider !== "")

  const submit = () => {
    if (!parsedClientArgs.ok) return
    const apiBase = draft.apiBase.trim()
    if (editing) {
      const body: UpdateOrgProviderKeyRequest = {
        name: draft.name.trim(),
        api_base: apiBase === "" ? null : apiBase,
        client_args: parsedClientArgs.value,
      }
      // Omitted rather than sent as null when it was left blank: an explicit
      // null clears the stored credential, and "I did not retype the secret" is
      // not a request to delete it.
      if (draft.apiKey !== "") body.api_key = draft.apiKey
      update.mutate({ keyId: editing.id, body }, { onSuccess: onClose })
      return
    }
    const body: CreateOrgProviderKeyRequest = {
      provider: draft.provider,
      name: draft.name.trim(),
      api_key: draft.apiKey === "" ? null : draft.apiKey,
      api_base: apiBase === "" ? null : apiBase,
      client_args: parsedClientArgs.value,
    }
    create.mutate(body, { onSuccess: onClose })
  }

  return (
    <Card>
      <Card.Content className="flex flex-col gap-4 p-5">
        <div className="text-title">
          {editing ? `Edit ${editing.name}` : "Add provider key"}
        </div>
        <ErrorBanner error={create.error ?? update.error} />

        {editing ? (
          // The provider is part of the key's identity (it is half of the
          // uniqueness constraint and the whole of what dispatch matches on),
          // and the API's update body cannot change it.
          <div className="flex flex-col gap-1">
            <span className="text-sm font-medium text-foreground">
              Provider
            </span>
            <span className="text-sm text-muted">
              {editing.provider}. Create a second key to use another provider.
            </span>
          </div>
        ) : (
          <ProviderComboBox
            label="Provider"
            value={draft.provider}
            onChange={(provider) => setDraft({ ...draft, provider })}
            description="Which upstream this credential is for. The gateway matches it against the provider half of a model name."
          />
        )}

        <Field
          label="Name"
          value={draft.name}
          onChange={(name) => setDraft({ ...draft, name })}
          isRequired
          placeholder="Production"
          description="What this key is called in the organization. Unique per provider, so a second OpenAI key needs a different name."
        />

        <SecretField
          label="API key"
          value={draft.apiKey}
          onChange={(apiKey) => setDraft({ ...draft, apiKey })}
          description={
            editing
              ? "Encrypted at rest and never shown again. Leave blank to keep the current key."
              : "Encrypted at rest and never shown again; only the last 4 characters come back."
          }
        />

        <Field
          label="API base URL"
          value={draft.apiBase}
          onChange={(apiBase) => setDraft({ ...draft, apiBase })}
          placeholder="https://api.example.com/v1"
          description="Optional. Point this key at a compatible endpoint of your own instead of the provider's default."
        />

        <ClientArgsField
          value={draft.clientArgs}
          onChange={(clientArgs) => setDraft({ ...draft, clientArgs })}
          error={clientArgsError}
        />

        <div className="flex gap-2">
          <Button
            variant="primary"
            isDisabled={!canSubmit}
            isPending={pending}
            onPress={submit}
          >
            {editing ? "Save" : "Add provider key"}
          </Button>
          <Button variant="ghost" isDisabled={pending} onPress={onClose}>
            Close
          </Button>
        </div>
      </Card.Content>
    </Card>
  )
}

export function OrganizationProviderKeysPage() {
  const context = useOrganizationContext()
  const keys = useOrgProviderKeys()
  const settings = useSettings()
  const archive = useArchiveOrgProviderKey()
  const restore = useRestoreOrgProviderKey()
  const remove = useDeleteOrgProviderKey()
  const setDefault = useSetOrgProviderKeyDefault()

  const [adding, setAdding] = useState(false)
  const [editingId, setEditingId] = useState<string | null>(null)
  const [showArchived, setShowArchived] = useState(false)

  const canEdit = canManage(context.data)
  const editing = keys.data?.find((key) => key.id === editingId) ?? null
  const archivedCount = (keys.data ?? []).filter((key) =>
    Boolean(key.archived_at),
  ).length
  const rows = (keys.data ?? []).filter(
    (key) => showArchived || !key.archived_at,
  )

  // Same gate the `/providers` page applies, for the same reason: without
  // `OTARI_SECRET_KEY` the gateway cannot encrypt a credential, so the write
  // would fail at submit time. Fail closed on an error, open while loading.
  const secretKeyConfigured = settings.data
    ? settings.data.secret_key_configured !== false
    : !settings.isError

  const columns: DataTableColumn<OrgProviderKey>[] = [
    {
      id: "name",
      header: "Name",
      isRowHeader: true,
      cell: (row) => (
        <div className="flex items-center gap-2">
          <span className="font-medium text-foreground">{row.name}</span>
          {row.is_org_default ? (
            <Chip size="sm" color="accent">
              default
            </Chip>
          ) : null}
          {row.archived_at ? (
            <Chip size="sm" color="default">
              archived
            </Chip>
          ) : null}
        </div>
      ),
    },
    {
      id: "provider",
      header: "Provider",
      cell: (row) => <span className="text-muted">{row.provider}</span>,
    },
    {
      id: "api_key",
      header: "API key",
      cell: (row) => (
        <code className="text-muted">
          {row.last4 ? `••••${row.last4}` : "none set"}
        </code>
      ),
    },
    {
      id: "api_base",
      header: "API base",
      cell: (row) => (
        <span className="text-muted">{row.api_base ?? "provider default"}</span>
      ),
    },
    {
      id: "created",
      header: "Created",
      cell: (row) => (
        <span className="text-muted">{formatRelative(row.created_at)}</span>
      ),
    },
  ]

  // Appended rather than declared with the rest and rendered empty: a column of
  // blank cells reads as actions that failed to load.
  if (canEdit) {
    columns.push({
      id: "actions",
      header: "Actions",
      align: "end",
      cell: (row) => (
        <div className="flex items-center justify-end gap-1.5">
          {row.archived_at ? (
            <>
              <Button
                size="sm"
                variant="outline"
                isDisabled={restore.isPending}
                onPress={() => restore.mutate(row.id)}
              >
                Restore
              </Button>
              {/* Permanent, and the only place it is offered: the API
                    accepts a delete for an archived key alone. */}
              <ConfirmButton
                confirmLabel="Delete"
                isPending={remove.isPending}
                onConfirm={() => remove.mutate(row.id)}
              >
                Delete
              </ConfirmButton>
            </>
          ) : (
            <>
              <Button
                size="sm"
                variant="outline"
                isDisabled={row.is_org_default || setDefault.isPending}
                onPress={() => setDefault.mutate(row.id)}
              >
                Make default
              </Button>
              <Button
                size="sm"
                variant="ghost"
                onPress={() => {
                  setAdding(false)
                  setEditingId(row.id)
                }}
              >
                Edit
              </Button>
              {/* Archive rather than delete: it is reversible, it is what
                    clears the default, and it is the step the API requires
                    before a key can be removed for good. */}
              <ConfirmButton
                confirmLabel="Archive"
                isPending={archive.isPending}
                onConfirm={() =>
                  archive.mutate(row.id, {
                    onSuccess: () => {
                      if (editingId === row.id) setEditingId(null)
                    },
                  })
                }
              >
                Archive
              </ConfirmButton>
            </>
          )}
        </div>
      ),
    })
  }

  return (
    <div className="flex flex-col gap-6">
      <PageHeader
        title="Providers"
        description="The organization's own upstream credentials. Every workspace in the organization can use them, and the default for a provider is the one a request gets when it names no instance. Keys are encrypted at rest and never shown again."
        action={
          canEdit && !adding ? (
            <Button
              variant="primary"
              isDisabled={!secretKeyConfigured}
              onPress={() => {
                setEditingId(null)
                setAdding(true)
              }}
            >
              Add provider key
            </Button>
          ) : null
        }
      />

      <ErrorBanner
        error={
          context.error ??
          keys.error ??
          archive.error ??
          restore.error ??
          remove.error ??
          setDefault.error
        }
      />

      {/* Held back until the context has actually answered: `canEdit` is false
          while the role is still resolving, and a banner that flashes "you may
          not do this" on every load would be telling most people the opposite
          of the truth. */}
      {context.data && !canEdit ? (
        <InfoBanner>
          Only organization owners and admins can add or change provider keys.
        </InfoBanner>
      ) : null}

      {/* Gated on `canEdit` as well: a member who cannot add a key has nothing
          to do about a missing server setting, and the sentence would only tell
          them that a control they never see is disabled. */}
      {canEdit && !secretKeyConfigured ? (
        <InfoBanner tone="warning">
          <code>OTARI_SECRET_KEY</code> is not set, so provider keys can't be
          encrypted at rest and adding one from the dashboard is disabled. Set
          it on the server and restart.
        </InfoBanner>
      ) : null}

      {adding && secretKeyConfigured ? (
        <KeyForm editing={null} onClose={() => setAdding(false)} />
      ) : null}
      {editing ? (
        // Remounted per row: the draft is seeded from the key once, so editing a
        // second key would otherwise open with the first one's values.
        <KeyForm
          key={editing.id}
          editing={editing}
          onClose={() => setEditingId(null)}
        />
      ) : null}

      {archivedCount > 0 ? (
        <Checkbox isSelected={showArchived} onChange={setShowArchived}>
          Show archived ({archivedCount})
        </Checkbox>
      ) : null}

      <DataTable
        ariaLabel="Organization provider keys"
        columns={columns}
        rows={rows}
        getRowKey={(row) => row.id}
        isLoading={keys.isLoading}
        emptyContent="No provider keys yet. Add one to let every workspace in this organization call that provider."
      />
    </div>
  )
}
