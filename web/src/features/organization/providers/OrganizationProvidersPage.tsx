import { Button } from "@heroui/react"
import { useState } from "react"
import {
  FiArchive,
  FiEdit2,
  FiList,
  FiRotateCcw,
  FiStar,
  FiTrash2,
} from "react-icons/fi"

import type {
  CreateOrgProviderKeyRequest,
  OrgProviderKey,
  UpdateOrgProviderKeyRequest,
} from "@/client"
import { ConfirmRowAction } from "@/design-system/actions/ConfirmRowAction"
import { RowAction, RowActionRow } from "@/design-system/actions/RowAction"
import { DataTable, type DataTableColumn } from "@/design-system/data/DataTable"
import { PAGE_SIZE_OPTIONS } from "@/design-system/data/TablePagination"
import { ConfirmDialog } from "@/design-system/feedback/ConfirmDialog"
import { ErrorBanner } from "@/design-system/feedback/ErrorBanner"
import { FormDialog } from "@/design-system/feedback/FormDialog"
import { InfoBanner } from "@/design-system/feedback/InfoBanner"
import { Checkbox } from "@/design-system/forms/Checkbox"
import { Field } from "@/design-system/forms/Field"
import { SecretField } from "@/design-system/forms/SecretField"
import { useDirtySnapshot } from "@/design-system/forms/useDirtySnapshot"
import { Dot } from "@/design-system/indicators/Dot"
import { PageIntro } from "@/design-system/layout/PageIntro"
import { TableScrollFrame } from "@/design-system/layout/TableScrollFrame"
import { PricingOverrideDialog } from "@/features/organization/PricingOverrideDialog"
import { canManage } from "@/features/organization/roles"
import {
  BYO_UNSUPPORTED_PROVIDERS,
  type CredentialFieldValues,
  credentialFieldsFor,
  credentialSpecFor,
  mergeCredentialFields,
  splitClientArgs,
  validateCredentialFields,
} from "@/features/providers/providerCredentialFields"
import {
  ClientArgsField,
  formatClientArgs,
  ProviderComboBox,
  ProviderCredentialFields,
  parseClientArgs,
} from "@/features/providers/providerFields"
import {
  useArchiveOrgProviderKey,
  useCreateOrgProviderKey,
  useDeleteOrgProviderKey,
  useOrganizationContext,
  useOrgProviderKeys,
  useProviderKeyEncryption,
  useRestoreOrgProviderKey,
  useSetOrgProviderKeyDefault,
  useUpdateOrgProviderKey,
} from "@/shared/api/organizations"
import { useOrganizationPricing } from "@/shared/api/pricing"
import { formatRelative } from "@/shared/helpers/format"
import { providerDisplayName } from "@/shared/helpers/providers"
import { useUrlState } from "@/shared/helpers/urlState"

import { ProviderModelsPanel } from "./ProviderModelsPanel"

// Every period one model can carry, which is what the editor reads. Generous
// rather than tuned: the read is narrowed to a single model, so this bounds the
// periods stored for it rather than the organization's whole table.
const OVERRIDE_PAGE_SIZE = 200

// The two URL-held pieces of page state: which key's models are open, and which
// model's rate is being edited. `override` is the name the Models detail page
// already links with, so it is a contract rather than a choice.
const URL_DEFAULTS = {
  provider: "",
  override: "",
  models_page: "0",
  models_size: String(PAGE_SIZE_OPTIONS[0]),
}

// The organization's own upstream credentials: one BYO key per provider that
// every workspace under the tenant inherits.
//
// Not the workspace rail's `/providers`, which manages `provider_credentials`,
// keyed on an instance name and therefore owned by the process rather than by
// anyone in particular. The two pages look alike and are not the same thing.
// A standalone deployment reports both surfaces and shows both pages, under
// different labels; a hosted one reports `organization_providers` alone, since
// one `provider_credentials` row would serve every tenant
// (`STANDALONE_SURFACES` / `HOSTED_SURFACES` in
// `src/gateway/api/routes/bootstrap.py`).
//
// The per-workspace half of the same API (`/workspaces/{id}/provider-keys`:
// pin, disable, restrict to models) is deliberately not here. Those are one
// workspace's departure from what this page sets, so they belong beside that
// workspace: `WorkspaceProviderKeys`, on the Workspaces page, which has already
// asked "which workspace" before it shows any of it.

/** A key's editable fields, seeded from a row when one is being edited. */
interface KeyDraft {
  provider: string
  name: string
  apiKey: string
  apiBase: string
  /** The JSON escape hatch: whatever the provider's typed fields do not own. */
  clientArgs: string
  /** The typed `client_args` entries, keyed by SDK keyword argument. */
  credentials: CredentialFieldValues
  /** Typed secrets already stored, so blank means "keep it" rather than "clear it". */
  redacted: string[]
}

const EMPTY_DRAFT: KeyDraft = {
  provider: "",
  name: "",
  apiKey: "",
  apiBase: "",
  clientArgs: "",
  credentials: {},
  redacted: [],
}

function draftFrom(key: OrgProviderKey): KeyDraft {
  // The registry's fields come out of the stored JSON and into their own
  // controls; everything else stays in the textarea it was entered in.
  const { typed, rest, redacted } = splitClientArgs(
    credentialFieldsFor(key.provider),
    key.client_args,
  )
  return {
    provider: key.provider,
    name: key.name,
    // Never prefilled: the gateway stores the ciphertext and returns `last4`,
    // so there is nothing to prefill with. Blank on save means "leave it".
    apiKey: "",
    apiBase: key.api_base ?? "",
    clientArgs: formatClientArgs(rest),
    credentials: typed,
    redacted,
  }
}

function KeyForm({
  isOpen,
  editing,
  onClose,
}: {
  isOpen: boolean
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
  const credentialFields = credentialFieldsFor(draft.provider)
  const credentialErrors = validateCredentialFields(
    credentialFields,
    draft.credentials,
    draft.redacted,
  )
  const spec = credentialSpecFor(draft.provider)
  const isPending = create.isPending || update.isPending
  // The whole draft against what the form was seeded with, so a guard cannot
  // miss a field the form grows later.
  const { isDirty } = useDirtySnapshot(draft)
  const canSubmit =
    parsedClientArgs.ok &&
    Object.keys(credentialErrors).length === 0 &&
    draft.name.trim() !== "" &&
    (editing !== null || draft.provider !== "")

  const submit = () => {
    if (!parsedClientArgs.ok) return
    const apiBase = draft.apiBase.trim()
    // The typed fields and the textarea are two views of one `client_args`
    // object, so they are recombined before it goes out.
    const clientArgs = mergeCredentialFields(
      draft.credentials,
      parsedClientArgs.value,
      draft.redacted,
    )
    if (editing) {
      const body: UpdateOrgProviderKeyRequest = {
        name: draft.name.trim(),
        api_base: apiBase === "" ? null : apiBase,
        client_args: clientArgs,
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
      client_args: clientArgs,
    }
    create.mutate(body, { onSuccess: onClose })
  }

  return (
    <FormDialog
      isOpen={isOpen}
      onOpenChange={(open) => {
        if (!open) onClose()
      }}
      title={editing ? "Edit provider key" : "New provider key"}
      // The key it is about, where the title used to carry it inline. A dialog
      // title is a noun phrase.
      description={editing ? editing.name : undefined}
      submitLabel={editing ? "Save" : "Add provider key"}
      onSubmit={submit}
      isPending={isPending}
      isSubmitDisabled={!canSubmit}
      isDirty={isDirty}
      error={create.error ?? update.error}
    >
      {editing ? (
        // The provider is part of the key's identity (it is half of the
        // uniqueness constraint and the whole of what dispatch matches on),
        // and the API's update body cannot change it.
        <div className="flex flex-col gap-1">
          <span className="text-body">Provider</span>
          <span className="text-sm text-muted">
            {editing.provider}. Create a second key to use another provider.
          </span>
        </div>
      ) : (
        <ProviderComboBox
          label="Provider"
          value={draft.provider}
          // The typed fields belong to the provider, so a change to it drops
          // values that no longer have a field to sit in.
          onChange={(provider) =>
            setDraft({ ...draft, provider, credentials: {}, redacted: [] })
          }
          excludeIds={BYO_UNSUPPORTED_PROVIDERS}
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
        // Named for what the provider actually calls its credential, where that
        // is not an opaque API key (Bedrock's is a bearer token).
        label={spec?.apiKeyLabel ?? "API key"}
        value={draft.apiKey}
        onChange={(apiKey) => setDraft({ ...draft, apiKey })}
        description={[
          editing
            ? "Encrypted at rest and never shown again. Leave blank to keep the current key."
            : "Encrypted at rest and never shown again; only the last 4 characters come back.",
          spec?.apiKeyHelpText,
        ]
          .filter(Boolean)
          .join(" ")}
      />

      <Field
        label="API base URL"
        value={draft.apiBase}
        onChange={(apiBase) => setDraft({ ...draft, apiBase })}
        placeholder="https://api.example.com/v1"
        description="Optional. Point this key at a compatible endpoint of your own instead of the provider's default."
      />

      {/* What this provider needs beyond a key, asked for by name rather than
          left to the JSON below. */}
      <ProviderCredentialFields
        provider={draft.provider}
        values={draft.credentials}
        onChange={(credentials) => setDraft({ ...draft, credentials })}
        errors={credentialErrors}
        redacted={draft.redacted}
      />

      <ClientArgsField
        value={draft.clientArgs}
        onChange={(clientArgs) => setDraft({ ...draft, clientArgs })}
        error={clientArgsError}
      />
    </FormDialog>
  )
}

export function OrganizationProvidersPage() {
  const context = useOrganizationContext()
  // One predicate for the whole page now that the list read is
  // organization-management-gated on the server too (otari-ai#1944): a member
  // cannot see these rows, not only leave them alone. Withheld rather than
  // fired and refused, the way `OrganizationGuardrailsPage` gates its own reads
  // and WorkspacesPage withholds the operator-only budget ones.
  //
  // Not widened to `isDeploymentOperator`: the server gates these rows on the
  // organization role alone, and operating the deployment grants no role.
  const canEdit = canManage(context.data)
  const keys = useOrgProviderKeys(canEdit)
  // Same gate the `/providers` page applies, for the same reason: without
  // `OTARI_SECRET_KEY` the gateway cannot encrypt a credential, so the write
  // would fail at submit time.
  const secretKeyConfigured = useProviderKeyEncryption()
  const archive = useArchiveOrgProviderKey()
  const restore = useRestoreOrgProviderKey()
  const remove = useDeleteOrgProviderKey()
  const setDefault = useSetOrgProviderKeyDefault()

  const [adding, setAdding] = useState(false)
  const [addOpenCount, setAddOpenCount] = useState(0)
  const [editingId, setEditingId] = useState<string>()
  const [showArchived, setShowArchived] = useState(false)
  const [pendingDelete, setPendingDelete] = useState<OrgProviderKey>()
  // Which key's models are open, and which model's rate is being edited. Both
  // in the URL: an expanded panel is worth sharing, and `?override=` is a
  // contract the Models detail page links into.
  const url = useUrlState(URL_DEFAULTS)
  const expandedKeyId = url.get("provider")
  // Snapped to an offered size, the way ActivityPage snaps its own: a
  // hand-edited or stale `models_size` must not reach the API as a limit it
  // never offers, or leave the rows-per-page select showing a value it does not.
  const modelsPage = Math.max(0, url.getNumber("models_page"))
  const modelsSize =
    PAGE_SIZE_OPTIONS.find((size) => size === url.getNumber("models_size")) ??
    PAGE_SIZE_OPTIONS[0]
  const ratingModelKey = url.get("override")
  // Narrowed to the model being edited, and read only while one is. Every
  // period of that model is what the editor needs, and the first page of the
  // whole table is not that: an organization with more overrides than fit in it
  // would open a create form over a rate that already exists, and the save
  // would earn the 409 the overlap check exists to prevent.
  const overrides = useOrganizationPricing(
    0,
    OVERRIDE_PAGE_SIZE,
    canEdit && ratingModelKey !== "",
    ratingModelKey,
  )

  // Bumped on each open, and the create form is keyed on it, so the draft (the
  // plaintext secret included) is fresh every time and untouched through the
  // exit: the dialog keeps its content while it animates out, so clearing on
  // the way out would blank the body in front of the operator. The mutation is
  // inside `KeyForm`, below the key, so a previous refusal's banner goes with
  // it. See feedback.md, "A draft is fresh on every open and untouched through
  // the exit".
  const openAdd = () => {
    setEditingId(undefined)
    setAddOpenCount((n) => n + 1)
    setAdding(true)
  }

  const editing = keys.data?.find((key) => key.id === editingId)
  const archivedCount = (keys.data ?? []).filter((key) =>
    Boolean(key.archived_at),
  ).length
  const rows = (keys.data ?? []).filter(
    (key) => showArchived || !key.archived_at,
  )
  // Archived keys are excluded: one that is out of use is not a problem to
  // report, and counting it would keep the banner up after the fix.
  const unreadableCount = (keys.data ?? []).filter(
    (key) => !key.usable && !key.archived_at,
  ).length

  const columns: DataTableColumn<OrgProviderKey>[] = [
    {
      id: "name",
      header: "Name",
      isRowHeader: true,
      cell: (row) => (
        // The marker states what is true, not what is missing: a key that is
        // the organization's default is marked, and one that is merely
        // available carries nothing. Archived takes the subtle dot, because it
        // is a row still present rather than a problem.
        <div className="flex items-center gap-3">
          <span className="font-medium text-foreground">{row.name}</span>
          {row.is_org_default ? (
            <span className="flex items-center gap-2 text-mono-caption text-muted">
              <Dot className="bg-accent" />
              DEFAULT
            </span>
          ) : null}
          {row.archived_at ? (
            <span className="flex items-center gap-2 text-mono-caption text-subtle">
              <Dot className="bg-text-subtle" />
              ARCHIVED
            </span>
          ) : null}
          {/* The exception to the rule above, because this one *is* a problem:
              the credential is stored but this deployment cannot decrypt it, so
              the key serves nothing and its models are absent from the catalog.
              Nothing else on the page says so, and the row is otherwise
              indistinguishable from a working one. */}
          {row.usable ? null : (
            <span className="flex items-center gap-2 text-mono-caption text-danger">
              <Dot className="bg-danger" />
              UNREADABLE
            </span>
          )}
        </div>
      ),
    },
    {
      id: "provider",
      header: "Provider",
      // The vendor's own spelling; the id stays what the form and the API use.
      cell: (row) => (
        <span className="text-muted">{providerDisplayName(row.provider)}</span>
      ),
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
        <RowActionRow>
          {row.archived_at ? (
            <>
              <RowAction
                icon={FiRotateCcw}
                label="Restore"
                isDisabled={restore.isPending}
                onPress={() => restore.mutate(row.id)}
              />
              {/* Permanent, and the only place it is offered: the API
                    accepts a delete for an archived key alone. */}
              <RowAction
                icon={FiTrash2}
                label="Delete"
                onPress={() => setPendingDelete(row)}
              />
            </>
          ) : (
            <>
              <RowAction
                icon={FiList}
                label="Models"
                ariaLabel={`Models on ${row.name}`}
                onPress={() =>
                  url.patch({
                    provider: expandedKeyId === row.id ? "" : row.id,
                    // A different provider's panel starts at its own first page.
                    models_page: "0",
                  })
                }
              />
              <RowAction
                icon={FiStar}
                label="Make default"
                isDisabled={row.is_org_default || setDefault.isPending}
                onPress={() => setDefault.mutate(row.id)}
              />
              <RowAction
                icon={FiEdit2}
                label="Edit"
                onPress={() => {
                  setAdding(false)
                  setEditingId(row.id)
                }}
              />
              {/* Archive rather than delete: it is reversible, it is what
                    clears the default, and it is the step the API requires
                    before a key can be removed for good. */}
              <ConfirmRowAction
                icon={FiArchive}
                label="Archive"
                confirmLabel="Archive"
                isPending={archive.isPending}
                onConfirm={() =>
                  archive.mutate(row.id, {
                    onSuccess: () => {
                      if (editingId === row.id) setEditingId(undefined)
                    },
                  })
                }
              />
            </>
          )}
        </RowActionRow>
      ),
    })
  }

  return (
    <div className="flex flex-col">
      <PageIntro
        title="Providers"
        action={
          canEdit ? (
            <Button
              // Visible while the dialog is open; disabled rather than hidden
              // without a server secret key, which is the rule for a control
              // that carries its own reason nearby.
              variant="primary"
              isDisabled={!secretKeyConfigured}
              onPress={openAdd}
            >
              Add provider key
            </Button>
          ) : null
        }
      >
        The organization&rsquo;s own upstream credentials. Every workspace in
        the organization can use them, and the default for a provider is the one
        a request gets when it names no instance. Keys are encrypted at rest and
        never shown again.
      </PageIntro>

      <ErrorBanner
        error={
          context.error ??
          keys.error ??
          overrides.error ??
          archive.error ??
          restore.error ??
          setDefault.error
        }
      />

      {/* Held back until the context has actually answered: `canEdit` is false
          while the role is still resolving, and a banner that flashes "you may
          not do this" on every load would be telling most people the opposite
          of the truth. */}
      {context.data && !canEdit ? (
        <InfoBanner>
          Only organization owners and admins can see this organization's
          provider keys.
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

      {/* The other half of the same setting: the key is set, and it is not the
          one these credentials were encrypted under, so nothing here can be
          decrypted. Worth its own sentence because every other signal on the
          page reads normal, while the models these providers serve are missing
          from the catalog entirely. */}
      {canEdit && secretKeyConfigured && unreadableCount > 0 ? (
        <InfoBanner tone="warning">
          {unreadableCount === 1
            ? "One provider key can't be decrypted on this deployment, so it serves nothing and its models are absent from the catalog."
            : `${unreadableCount} provider keys can't be decrypted on this deployment, so they serve nothing and their models are absent from the catalog.`}{" "}
          This happens when <code>OTARI_SECRET_KEY</code> changed since they
          were stored. Re-enter the API key on each to store it under the
          current one.
        </InfoBanner>
      ) : null}

      <KeyForm
        key={addOpenCount}
        isOpen={adding && secretKeyConfigured}
        editing={null}
        onClose={() => setAdding(false)}
      />
      {editing ? (
        // Remounted per row: the draft is seeded from the key once, so editing a
        // second key would otherwise open with the first one's values.
        <KeyForm
          key={editing.id}
          isOpen
          editing={editing}
          onClose={() => setEditingId(undefined)}
        />
      ) : null}

      {/* The page is a stack of bands that set their own spacing, so this one
          carries its own air rather than taking it from a column gap. */}
      {archivedCount > 0 ? (
        <div className="pb-3">
          <Checkbox isSelected={showArchived} onChange={setShowArchived}>
            Show archived ({archivedCount})
          </Checkbox>
        </div>
      ) : null}

      {/* Withheld from a member along with the read that fills it: the banner
          above is their whole answer, and an empty table would tell them the
          organization has no keys rather than that they cannot see them. Drawn
          as loading while the role is still resolving, so an admin's first
          paint is the table they are about to get rather than an empty state,
          and withheld again if the context read is what failed, since neither
          an empty table nor a spinner is a true answer there. */}
      {canEdit || context.isPending ? (
        <TableScrollFrame className="otari-provider-keys-table">
          <DataTable
            ariaLabel="Organization provider keys"
            columns={columns}
            rows={rows}
            getRowKey={(row) => row.id}
            isLoading={context.isPending || keys.isLoading}
            emptyContent="No provider keys yet. Add one to let every workspace in this organization call that provider."
            detailKey={expandedKeyId === "" ? null : expandedKeyId}
            renderDetail={(row) => (
              <ProviderModelsPanel
                providerKey={row}
                canEdit={canEdit}
                onEditRate={(model) =>
                  url.patch({ override: `${row.provider}:${model.model}` })
                }
                page={modelsPage}
                pageSize={modelsSize}
                onPageChange={(next) =>
                  url.patch({ models_page: String(next) })
                }
                onPageSizeChange={(size) =>
                  url.patch({ models_size: String(size), models_page: "0" })
                }
              />
            )}
          />
        </TableScrollFrame>
      ) : null}

      <ConfirmDialog
        isOpen={pendingDelete !== undefined}
        // Cleared on the way out: a refusal otherwise sits on the mutation and
        // greets the next row's confirm as if that row had failed.
        onOpenChange={(open) => {
          if (open) return
          setPendingDelete(undefined)
          remove.reset()
        }}
        heading="Delete provider key"
        body={
          pendingDelete
            ? `${pendingDelete.name} and its stored ${pendingDelete.provider} credential are removed for good. Archiving is the reversible step; this one cannot be undone.`
            : null
        }
        confirmLabel="Delete permanently"
        isPending={remove.isPending}
        error={remove.error}
        onConfirm={() => {
          if (!pendingDelete) return
          remove.mutate(pendingDelete.id, {
            onSuccess: () => setPendingDelete(undefined),
          })
        }}
      />

      {/* The organization's own rate for one model, opened from a model row or
          from the Models detail page's "Set your rate" link. Mounted here
          rather than inside the panel because the panel lives in a table cell,
          and a dialog rendered from one closes with the row that opened it.

          Not mounted until *this model's* rates are in hand. The editor reads
          its start values once, when it mounts, and `?override=` is in the URL a
          render before the read answering it settles: mounted early it seeds
          itself from nothing and keeps that, showing an admin an empty form over
          a rate that exists and replacing it on save.

          `isSuccess` alone is not that condition. The read is narrowed per model
          and keeps the previous model's rows as placeholder data while the next
          one is in flight, so opening a second model is a success carrying the
          first model's answer, which holds no row for the second. A read that
          fails leaves it unmounted and says so in the banner above. */}
      {canEdit && overrides.isSuccess && !overrides.isPlaceholderData ? (
        <PricingOverrideDialog
          key={ratingModelKey}
          isOpen={ratingModelKey !== ""}
          onOpenChange={(open) => {
            if (!open) url.patch({ override: "" })
          }}
          editing={(overrides.data?.data ?? []).find(
            (row) => row.model_key === ratingModelKey,
          )}
          initialModelKey={ratingModelKey}
          existing={overrides.data?.data ?? []}
          onSaved={() => url.patch({ override: "" })}
        />
      ) : null}
    </div>
  )
}
