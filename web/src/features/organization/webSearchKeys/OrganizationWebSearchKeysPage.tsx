import { useState } from "react"
import type { IconType } from "react-icons"
import {
  FiArchive,
  FiEdit2,
  FiRotateCcw,
  FiStar,
  FiTrash2,
} from "react-icons/fi"

import type { OrgWebSearchKey, SearchProviderInfo } from "@/client"
import { Button } from "@/design-system/actions/Button"
import { ConfirmRowAction } from "@/design-system/actions/ConfirmRowAction"
import { RowAction, RowActionRow } from "@/design-system/actions/RowAction"
import { DataTable, type DataTableColumn } from "@/design-system/data/DataTable"
import { ConfirmDialog } from "@/design-system/feedback/ConfirmDialog"
import { ErrorBanner } from "@/design-system/feedback/ErrorBanner"
import { InfoBanner } from "@/design-system/feedback/InfoBanner"
import { Checkbox } from "@/design-system/forms/Checkbox"
import { Dot } from "@/design-system/indicators/Dot"
import { PageIntro } from "@/design-system/layout/PageIntro"
import { SettingsGroup } from "@/design-system/layout/SettingsGroup"
import { canManage } from "@/features/organization/roles"
import { WEB_TOOL_KINDS, type WebToolKind } from "@/features/webSearch/webTools"
import {
  useOrganizationContext,
  useProviderKeyEncryption,
} from "@/shared/api/organizations"
import { useSearchProviders } from "@/shared/api/tools"
import {
  KEY_PAGE_LIMIT,
  useArchiveOrgWebSearchKey,
  useDeleteOrgWebSearchKey,
  useOrgWebSearchKeys,
  useRestoreOrgWebSearchKey,
  useSetOrgWebSearchKeyDefault,
  useUnsetOrgWebSearchKeyDefault,
} from "@/shared/api/webSearchKeys"
import { docsSourceHref } from "@/shared/helpers/docs"
import { formatNumber, formatRelative } from "@/shared/helpers/format"

import { WebSearchKeyDialog } from "./WebSearchKeyDialog"
import { KIND_COPY, keyForEveryWorkspace, keysOfKind } from "./webSearchKeys"

// The organization's own search and fetch keys. A workspace uses the key it
// pinned on Tools > Web search, else the key starred here, else the
// deployment's own tool, so adding a key changes nothing until it is starred or
// pinned. It sets no prices: the deployment's operator prices an
// organization's search and fetch calls.

// The star filled is the workspace default, the key every workspace without a
// pin uses, so the glyph shows the state its press toggles.
const FilledStar: IconType = (props) => (
  <FiStar {...props} fill="currentColor" />
)

function KeysTable({
  kind,
  keys,
  showArchived,
  isLoading,
  canEdit,
  canAdd,
  catalogLoaded,
  onAdd,
  onEdit,
  onDelete,
}: {
  kind: WebToolKind
  keys: readonly OrgWebSearchKey[]
  showArchived: boolean
  isLoading: boolean
  canEdit: boolean
  canAdd: boolean
  /** The dialog seeds its fields from the provider catalog, so it opens only once that has loaded. */
  catalogLoaded: boolean
  onAdd: () => void
  onEdit: (key: OrgWebSearchKey) => void
  onDelete: (key: OrgWebSearchKey) => void
}) {
  const copy = KIND_COPY[kind]
  const archive = useArchiveOrgWebSearchKey()
  const restore = useRestoreOrgWebSearchKey()
  const setDefault = useSetOrgWebSearchKeyDefault()
  const unsetDefault = useUnsetOrgWebSearchKeyDefault()
  const rows = keysOfKind(keys, kind, showArchived)
  const live = keysOfKind(keys, kind, false)
  const marked = keyForEveryWorkspace(keys, kind)

  const columns: DataTableColumn<OrgWebSearchKey>[] = [
    {
      id: "name",
      header: "Name",
      isRowHeader: true,
      cell: (row) => (
        <div className="flex flex-wrap items-center gap-3">
          <span className="font-medium text-foreground">{row.name}</span>
          {row.is_org_default ? (
            <span className="flex items-center gap-2 text-mono-caption text-muted">
              <Dot className="bg-accent" />
              WORKSPACE DEFAULT
            </span>
          ) : null}
          {row.archived_at ? (
            <span className="flex items-center gap-2 text-mono-caption text-subtle">
              <Dot className="bg-text-subtle" />
              ARCHIVED
            </span>
          ) : null}
          {/* A problem: the key is stored and this deployment cannot decrypt
              it, so every call passes over it. */}
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
      // A custom endpoint under the provider; none means the provider's own.
      cell: (row) => (
        <div className="flex flex-col gap-0.5">
          <span className="text-muted">{row.provider}</span>
          {row.api_base ? (
            <span className="break-all text-caption text-subtle">
              {row.api_base}
            </span>
          ) : null}
        </div>
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
      id: "created",
      header: "Created",
      cell: (row) => (
        <span className="text-muted">{formatRelative(row.created_at)}</span>
      ),
    },
  ]

  // Appended only for an editor: a column of blank cells reads as actions that
  // failed to load.
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
                ariaLabel={`Restore ${row.name}`}
                isDisabled={restore.isPending}
                onPress={() => restore.mutate(row.id)}
              />
              {/* The API deletes an archived key only. */}
              <RowAction
                icon={FiTrash2}
                label="Delete"
                ariaLabel={`Delete ${row.name}`}
                onPress={() => onDelete(row)}
              />
            </>
          ) : (
            <>
              <RowAction
                icon={row.is_org_default ? FilledStar : FiStar}
                label={
                  row.is_org_default
                    ? "Unset as workspace default"
                    : "Set as workspace default"
                }
                ariaLabel={
                  row.is_org_default
                    ? `Unset ${row.name} as workspace default`
                    : `Set ${row.name} as workspace default`
                }
                isDisabled={setDefault.isPending || unsetDefault.isPending}
                onPress={() =>
                  (row.is_org_default ? unsetDefault : setDefault).mutate(
                    row.id,
                  )
                }
              />
              <RowAction
                icon={FiEdit2}
                label="Edit"
                ariaLabel={`Edit ${row.name}`}
                isDisabled={!catalogLoaded}
                onPress={() => onEdit(row)}
              />
              {/* Reversible, and the step the API requires before a delete. It
                  also stops the key being used for every workspace. */}
              <ConfirmRowAction
                icon={FiArchive}
                label="Archive"
                confirmLabel="Archive"
                isPending={archive.isPending}
                onConfirm={() => archive.mutate(row.id)}
              />
            </>
          )}
        </RowActionRow>
      ),
    })
  }

  return (
    <SettingsGroup
      title={copy.title}
      count={isLoading ? undefined : live.length}
      description={copy.serves}
      action={
        canEdit ? (
          <Button
            variant="primary"
            className="min-h-11 md:min-h-9"
            // Disabled without a server secret key, which the banner above
            // explains, and until the provider catalog loads.
            isDisabled={!canAdd || !catalogLoaded}
            onPress={onAdd}
          >
            {`Add ${copy.noun}`}
          </Button>
        ) : null
      }
    >
      <ErrorBanner
        error={
          archive.error ??
          restore.error ??
          setDefault.error ??
          unsetDefault.error
        }
      />
      <DataTable
        ariaLabel={copy.title}
        columns={columns}
        rows={rows}
        getRowKey={(row) => row.id}
        isLoading={isLoading}
        emptyContent={copy.empty}
        mobileLayout="cards"
      />
      {live.length > 0 ? (
        <p className="pt-3 text-caption">
          {!marked
            ? copy.withoutDefault
            : marked.usable
              ? `Every workspace that pinned no key uses ${marked.name}.`
              : `${marked.name} can't be decrypted on this deployment, so workspaces that pinned no key use this deployment's own tool.`}
        </p>
      ) : null}
    </SettingsGroup>
  )
}

export function OrganizationWebSearchKeysPage() {
  const context = useOrganizationContext()
  // The server gates these rows on the organization role, so a member's read is
  // not sent.
  const canEdit = canManage(context.data)
  const keys = useOrgWebSearchKeys(canEdit)
  // Without OTARI_SECRET_KEY the gateway cannot encrypt a key, so an add would
  // fail at submit.
  const secretKeyConfigured = useProviderKeyEncryption()
  const searchCatalog = useSearchProviders("search", canEdit)
  const fetchCatalog = useSearchProviders("fetch", canEdit)
  const catalogs: Record<WebToolKind, SearchProviderInfo[] | undefined> = {
    search: searchCatalog.data,
    fetch: fetchCatalog.data,
  }
  const remove = useDeleteOrgWebSearchKey()

  // Bumped on each open, and the add dialog is keyed on it, so its draft (the
  // secret included) is fresh every time and untouched through the exit.
  const [adding, setAdding] = useState(false)
  const [addKind, setAddKind] = useState<WebToolKind>("search")
  const [addOpenCount, setAddOpenCount] = useState(0)
  const [editingId, setEditingId] = useState<string>()
  const [showArchived, setShowArchived] = useState(false)
  const [pendingDelete, setPendingDelete] = useState<OrgWebSearchKey>()

  const all = keys.data?.data ?? []
  const editing = all.find((key) => key.id === editingId)
  const archivedCount = all.filter((key) => Boolean(key.archived_at)).length
  // An archived key is out of use, so it is not a problem to report.
  const unreadableCount = all.filter(
    (key) => !key.usable && !key.archived_at,
  ).length

  return (
    <div className="flex flex-col gap-10 pb-10">
      <PageIntro
        title="Web search providers"
        docsHref={docsSourceHref("tools.md", "organization-search-keys")}
      >
        Your organization&rsquo;s own keys for web search and fetch. A workspace
        uses the key its admins pinned, else the workspace default, else this
        deployment&rsquo;s own tool. Keys are encrypted at rest and never shown
        again.
      </PageIntro>

      <ErrorBanner
        error={
          context.error ??
          keys.error ??
          searchCatalog.error ??
          fetchCatalog.error
        }
      />

      {keys.data && keys.data.count > keys.data.data.length ? (
        <InfoBanner>
          {`This page lists the first ${formatNumber(KEY_PAGE_LIMIT)} of this organization's ${formatNumber(keys.data.count)} keys.`}
        </InfoBanner>
      ) : null}

      {/* Held back until the context answers: `canEdit` is false while the role
          resolves, and the banner would flash for an admin on every load. */}
      {context.data && !canEdit ? (
        <InfoBanner>
          Only organization owners and admins can see this organization&rsquo;s
          web search keys.
        </InfoBanner>
      ) : null}

      {canEdit && !secretKeyConfigured ? (
        <InfoBanner tone="warning">
          <code>OTARI_SECRET_KEY</code> is not set, so keys can&rsquo;t be
          encrypted at rest and adding one from the dashboard is disabled. Set
          it on the server and restart.
        </InfoBanner>
      ) : null}

      {canEdit && secretKeyConfigured && unreadableCount > 0 ? (
        <InfoBanner tone="warning">
          {unreadableCount === 1
            ? "One key can't be decrypted on this deployment, so every call passes over it."
            : `${unreadableCount} keys can't be decrypted on this deployment, so every call passes over them.`}{" "}
          This happens when <code>OTARI_SECRET_KEY</code> changed since they
          were stored. Edit each one and enter its API key again to encrypt it
          under the current one.
        </InfoBanner>
      ) : null}

      {archivedCount > 0 && canEdit ? (
        <Checkbox isSelected={showArchived} onChange={setShowArchived}>
          Show archived ({archivedCount})
        </Checkbox>
      ) : null}

      {/* Withheld from a member along with the read: the banner above is their
          answer, and an empty table would say the organization has no keys. */}
      {canEdit || context.isPending
        ? WEB_TOOL_KINDS.map((kind) => (
            <KeysTable
              key={kind}
              kind={kind}
              keys={all}
              showArchived={showArchived}
              isLoading={context.isPending || keys.isLoading}
              canEdit={canEdit}
              canAdd={secretKeyConfigured}
              catalogLoaded={catalogs[kind] !== undefined}
              onAdd={() => {
                setAddKind(kind)
                setAddOpenCount((count) => count + 1)
                setAdding(true)
              }}
              onEdit={(key) => setEditingId(key.id)}
              onDelete={setPendingDelete}
            />
          ))
        : null}

      <WebSearchKeyDialog
        key={`${addKind}-${addOpenCount}`}
        kind={addKind}
        providers={catalogs[addKind] ?? []}
        isOpen={adding}
        onClose={() => setAdding(false)}
      />
      {editing && catalogs[editing.kind] ? (
        // Remounted per key, since the draft is seeded once.
        <WebSearchKeyDialog
          key={editing.id}
          kind={editing.kind}
          editing={editing}
          providers={catalogs[editing.kind] ?? []}
          onClose={() => setEditingId(undefined)}
        />
      ) : null}

      <ConfirmDialog
        isOpen={pendingDelete !== undefined}
        // Cleared on the way out, so a refusal does not greet the next row.
        onOpenChange={(open) => {
          if (open) return
          setPendingDelete(undefined)
          remove.reset()
        }}
        heading="Delete key"
        body={
          pendingDelete
            ? `${pendingDelete.name} and its stored ${pendingDelete.provider} key are removed for good, with every workspace's pin on it. Archiving is the reversible step; this one cannot be undone.`
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
    </div>
  )
}
