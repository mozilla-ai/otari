import { useState } from "react"
import { FiEdit2, FiList, FiRefreshCw, FiTrash2 } from "react-icons/fi"

import type {
  HostedCatalogRefresh,
  HostedProvider,
  ProviderHealth,
} from "@/client"
import { Button } from "@/design-system/actions/Button"
import { RowAction, RowActionRow } from "@/design-system/actions/RowAction"
import { DataTable, type DataTableColumn } from "@/design-system/data/DataTable"
import { ConfirmDialog } from "@/design-system/feedback/ConfirmDialog"
import { EmptyState } from "@/design-system/feedback/EmptyState"
import { ErrorBanner } from "@/design-system/feedback/ErrorBanner"
import { InfoBanner } from "@/design-system/feedback/InfoBanner"
import { PageLoading } from "@/design-system/feedback/PageLoading"
import { Toggle } from "@/design-system/forms/Toggle"
import { Badge } from "@/design-system/indicators/Badge"
import { PageIntro } from "@/design-system/layout/PageIntro"
import { TableScrollFrame } from "@/design-system/layout/TableScrollFrame"
import { HealthPill } from "@/features/providers/HealthPill"
import { ApiError } from "@/shared/api/client"
import { useDeploymentAdminAccess } from "@/shared/api/deployment"
import {
  useDeleteHostedProvider,
  useHostedCatalogRefresh,
  useHostedCatalogRefreshPreview,
  useHostedProviders,
  useUpdateHostedProvider,
} from "@/shared/api/hostedProviders"
import { useProviderHealth, useProviders } from "@/shared/api/providers"
import { UnavailableHere } from "@/shared/components/access/UnavailableHere"
import { useSurfaces } from "@/shared/hooks/useDeployment"

import { AddHostedProviderForm } from "./AddHostedProviderForm"
import { ConfiguredInstancesSection } from "./ConfiguredInstancesSection"
import { DeploymentProviderMigration } from "./DeploymentProviderMigration"
import { EditHostedProviderForm } from "./EditHostedProviderForm"
import { HostedProviderModelsPanel } from "./HostedProviderModelsPanel"
import {
  catalogPreviewOutcome,
  catalogRefreshFailures,
  catalogRefreshOutcome,
} from "./refreshOutcome"

// The providers this deployment serves inference on with its own keys, for a
// request that brings no credential of its own. Deployment-scoped, which is
// what sets it apart from Providers (an organization's own keys).
//
// Operator-only, and the API refuses anyone else with a 404, so the page asks
// the deployment admin gate first, as Accounts does, and words a refusal as the
// page not being theirs rather than as an error. Whether this deployment hosts
// the surface at all is the shell's question, answered from the bootstrap; the
// page reads the same answer, so it says so itself wherever it is rendered.
//
// A provider kept to one organization shows which; the restriction is set and
// lifted through the API, since this build has no deployment-wide list of
// organizations to pick one from.
//
// A provider defined in the configuration's `hosted_providers:` block is listed
// with a `config.yml` badge and changed only there: its key, endpoint and switch
// are read-only here, while its models are administered as any other's.

/**
 * What the last sweep did, stated rather than implied by the table redrawing.
 *
 * The kept groups are worth the space: a sweep that leaves priced models under
 * a provider this page does not offer looks like it failed, when something else
 * on the deployment serves them.
 */
function CatalogSweepReport({
  result,
  onRemoveProvider,
}: {
  result: HostedCatalogRefresh
  onRemoveProvider: (provider: string) => void
}) {
  const failures = catalogRefreshFailures(result)
  return (
    <InfoBanner tone={failures.length > 0 ? "warning" : "info"}>
      <p role="status">{catalogRefreshOutcome(result)}</p>
      {result.removed_count > 0 ? (
        <p className="mt-1 text-caption">
          {result.removed_price_rows} stored{" "}
          {result.removed_price_rows === 1 ? "rate" : "rates"} and{" "}
          {result.removed_override_rows} organization{" "}
          {result.removed_override_rows === 1 ? "override" : "overrides"} went
          with them: {result.removed.join(", ")}
          {result.removed_count > result.removed.length
            ? `, and ${result.removed_count - result.removed.length} more`
            : ""}
          .
        </p>
      ) : null}
      {result.kept.map((group) => (
        <p key={group.reason} className="mt-1 text-caption">
          Kept {group.count} {group.count === 1 ? "model" : "models"}:{" "}
          {group.reason}.
        </p>
      ))}
      {failures.map((failure) => (
        <div key={failure.provider} className="mt-1 text-caption">
          <p>{failure.message}</p>
          {failure.offerRemoval ? (
            // Offered rather than done: a delete destroys the stored key.
            <p className="mt-1 flex flex-wrap items-center gap-2">
              <span>This provider serves nothing until it answers again.</span>
              <Button
                size="sm"
                variant="danger"
                onPress={() => onRemoveProvider(failure.provider)}
              >
                <FiTrash2 aria-hidden className="size-4" />
                Remove provider
              </Button>
            </p>
          ) : null}
        </div>
      ))}
    </InfoBanner>
  )
}

/** The models a sweep is about to remove, named before anything is removed. */
function CatalogSweepPreview({
  preview,
}: {
  preview: ReturnType<typeof useHostedCatalogRefreshPreview>
}) {
  if (preview.error) {
    // The confirm refuses to fire in this state, so the banner is the answer.
    return <ErrorBanner error={preview.error} />
  }
  if (preview.isPending || preview.isFetching) {
    return <p>Working out what would be removed…</p>
  }
  const result = preview.data
  const failures = catalogRefreshFailures(result)
  return (
    <div className="flex flex-col gap-3">
      <p>{catalogPreviewOutcome(result)}</p>
      {result.removed.length > 0 ? (
        <ul className="max-h-48 overflow-y-auto font-mono text-caption">
          {result.removed.map((modelKey) => (
            <li key={modelKey}>{modelKey}</li>
          ))}
        </ul>
      ) : null}
      {result.removed_count > result.removed.length ? (
        <p className="text-caption">
          … and {result.removed_count - result.removed.length} more.
        </p>
      ) : null}
      {result.kept.map((group) => (
        <p key={group.reason} className="text-caption">
          Keeping {group.count} {group.count === 1 ? "model" : "models"}:{" "}
          {group.reason}.
        </p>
      ))}
      {failures.map((failure) => (
        <p key={failure.provider} className="text-caption">
          {failure.message}
        </p>
      ))}
      <p className="text-caption">
        Removing a model takes its rate history and the overrides of
        organizations with no key of their own for the provider. Requests
        already billed keep the cost they were billed at.
      </p>
    </div>
  )
}

export function HostedProvidersPage() {
  const hostsSurface = useSurfaces()
  // Before any hook that asks the API, so an unpublished surface asks nothing.
  if (!hostsSurface("hosted_providers")) {
    return <UnavailableHere title="Hosted providers" />
  }
  return <HostedProvidersContent />
}

function HostedProvidersContent() {
  const access = useDeploymentAdminAccess()
  // Withheld until the gate answers, so a non-operator load does not put a 404
  // in the console to learn what the gate is about to say.
  const isGranted = access.data === true
  const providers = useHostedProviders(isGranted)
  const health = useProviderHealth(isGranted)
  const configured = useProviders(isGranted)
  const update = useUpdateHostedProvider()
  const remove = useDeleteHostedProvider()
  const sweep = useHostedCatalogRefresh()
  const [isSweepOpen, setIsSweepOpen] = useState(false)
  const [sweepOpenCount, setSweepOpenCount] = useState(0)
  // Asked only while the dialog is open, because answering dials every
  // configured provider, and asked afresh on every opening.
  const preview = useHostedCatalogRefreshPreview(
    isSweepOpen ? sweepOpenCount : 0,
  )
  // The sweep takes no list, so the confirm waits for one this opening read
  // successfully and shows: never a previous opening's, a failed or an
  // in-flight one.
  const isPreviewShown = preview.isSuccess && !preview.isFetching

  const [isAdding, setIsAdding] = useState(false)
  const [addOpenCount, setAddOpenCount] = useState(0)
  const [editingProvider, setEditingProvider] = useState("")
  const [modelsFor, setModelsFor] = useState("")
  // A name rather than a row: the sweep report's own remove offer can name a
  // provider this page has not loaded.
  const [pendingDelete, setPendingDelete] = useState("")

  const openAdd = () => {
    setEditingProvider("")
    setAddOpenCount((n) => n + 1)
    setIsAdding(true)
  }

  if (access.isLoading) {
    return <PageLoading label="Loading hosted providers…" />
  }

  // A gate that failed to answer is not a gate that said no.
  if (access.isError) {
    return (
      <div className="flex flex-col">
        <PageIntro title="Hosted providers" />
        <ErrorBanner error={access.error} />
      </div>
    )
  }

  // The list's 404 is the router's operator refusal, the same predicate as the
  // gate's, so a disagreement between the two still offers no write.
  const isListRefused =
    providers.error instanceof ApiError && providers.error.status === 404
  if (!isGranted || isListRefused) {
    return (
      <div className="flex flex-col">
        <PageIntro title="Hosted providers" />
        <EmptyState
          title="Hosted providers is not available to you"
          description="Managing the keys this deployment serves inference on is for its operators. Ask one of them if you need access here."
        />
      </div>
    )
  }

  const rows = providers.data?.data ?? []
  const deleting = rows.find((row) => row.name === pendingDelete)
  const editing = rows.find((row) => row.name === editingProvider)
  const expanded = rows.find((row) => row.name === modelsFor)
  // Health is keyed by name, and a configured instance of the same name is the
  // one dialed under it, so its status would read as this row's. Such a row,
  // and every row while the configured names are unknown, shows none.
  const configuredNames = configured.isSuccess
    ? new Set(configured.data.providers.map((row) => row.instance))
    : undefined
  const healthByName = new Map<string, ProviderHealth>(
    (health.data?.providers ?? []).map((row) => [row.instance, row]),
  )
  const healthOf = (name: string) =>
    configuredNames === undefined || configuredNames.has(name)
      ? undefined
      : healthByName.get(name)
  // A failed list is not an empty one: rendering "none configured" over a 500
  // would tell an operator their live providers are gone.
  const isEmpty = !providers.error && !providers.isPending && rows.length === 0

  const columns: DataTableColumn<HostedProvider>[] = [
    {
      id: "provider",
      header: "Provider",
      isRowHeader: true,
      cell: (row) => (
        <span className="flex flex-wrap items-center gap-2">
          {row.name}
          {row.name === row.provider ? null : (
            <span className="text-caption">{row.provider}</span>
          )}
          {row.source === "config" ? (
            <Badge tone="muted">config.yml</Badge>
          ) : null}
          {row.shadows_config ? (
            <Badge tone="muted">Overrides config.yml</Badge>
          ) : null}
        </span>
      ),
    },
    {
      id: "key",
      header: "API key",
      cell: (row) =>
        !row.decryptable ? (
          <Badge tone="warn">Key unreadable</Badge>
        ) : row.api_key_last4 ? (
          <code className="text-muted">••••{row.api_key_last4}</code>
        ) : row.has_api_key ? (
          <code className="text-muted">••••</code>
        ) : (
          <span className="text-muted">None needed</span>
        ),
    },
    {
      id: "health",
      header: "Status",
      cell: (row) =>
        row.enabled ? (
          <HealthPill health={healthOf(row.name)} />
        ) : (
          <span className="text-caption">Switched off</span>
        ),
    },
    {
      id: "api_base",
      header: "Endpoint",
      cell: (row) => (
        <span className="text-muted">{row.api_base ?? "provider default"}</span>
      ),
    },
    {
      id: "scope",
      header: "Serves",
      cell: (row) =>
        row.restricted_to_organization_id ? (
          <span
            className="flex flex-col gap-0.5"
            title={row.restricted_to_organization_id}
          >
            <span className="text-body">One organization</span>
            <code className="truncate text-caption">
              {row.restricted_to_organization_id}
            </code>
          </span>
        ) : (
          <span className="text-muted">Every organization</span>
        ),
    },
    {
      id: "enabled",
      header: "Serving",
      cell: (row) => (
        <Toggle
          isSelected={row.enabled}
          // A configured provider is switched in the configuration.
          isDisabled={
            row.source === "config" ||
            (update.isPending && update.variables?.provider === row.name)
          }
          label={`Serve traffic on ${row.name}`}
          onChange={(enabled) =>
            update.mutate({ provider: row.name, body: { enabled } })
          }
        />
      ),
    },
    {
      id: "actions",
      header: "Actions",
      align: "end",
      cell: (row) => (
        <RowActionRow>
          <RowAction
            icon={FiList}
            label="Models"
            ariaLabel={`Models on ${row.name}`}
            onPress={() =>
              setModelsFor((current) => (current === row.name ? "" : row.name))
            }
          />
          {row.source === "config" ? null : (
            <>
              <RowAction
                icon={FiEdit2}
                label="Edit"
                ariaLabel={`Edit ${row.name}`}
                onPress={() => {
                  setIsAdding(false)
                  setEditingProvider(row.name)
                }}
              />
              <RowAction
                icon={FiTrash2}
                label="Remove"
                ariaLabel={`Remove ${row.name}`}
                onPress={() => setPendingDelete(row.name)}
              />
            </>
          )}
        </RowActionRow>
      ),
    },
  ]

  return (
    <div className="flex flex-col">
      <PageIntro
        title="Hosted providers"
        action={
          <div className="flex flex-wrap items-center gap-2">
            <Button
              className="min-h-11 md:min-h-9"
              isPending={sweep.isPending}
              onPress={() => {
                setSweepOpenCount((n) => n + 1)
                setIsSweepOpen(true)
              }}
            >
              <FiRefreshCw aria-hidden className="size-4" />
              Refresh and purge
            </Button>
            <Button
              variant="primary"
              className="min-h-11 md:min-h-9"
              onPress={openAdd}
            >
              Add provider
            </Button>
          </div>
        }
      >
        The providers this deployment serves inference on, and the keys it
        serves them with. Used only for requests that bring no credential of
        their own, after an organization's own keys and the configured
        providers.
      </PageIntro>

      <ErrorBanner
        error={providers.error ?? update.error ?? remove.error ?? sweep.error}
      />
      <DeploymentProviderMigration />
      {sweep.data ? (
        <CatalogSweepReport
          result={sweep.data}
          onRemoveProvider={setPendingDelete}
        />
      ) : null}

      {isEmpty ? (
        <EmptyState
          title="No hosted providers configured"
          description="Until one is configured, a request that brings no credential of its own is not served here."
          actionLabel="Add provider"
          onAction={openAdd}
        />
      ) : (
        <TableScrollFrame className="otari-hosted-providers-table">
          <DataTable
            ariaLabel="Hosted providers"
            columns={columns}
            rows={rows}
            // The name, not the id: a configured provider has no row id.
            getRowKey={(row) => row.name}
            isLoading={providers.isLoading}
          />
        </TableScrollFrame>
      )}

      {/* Below the table rather than a detail row inside it, as on the
          organization Providers page: the provider table scrolls sideways on a
          narrow screen, and a row inside it would scroll and clip with it. */}
      {expanded ? (
        <section
          key={expanded.name}
          aria-label={`Models on ${expanded.name}`}
          className="mt-4 rounded-lg border border-border bg-surface"
        >
          <HostedProviderModelsPanel provider={expanded} />
        </section>
      ) : null}

      <ConfiguredInstancesSection />

      <AddHostedProviderForm
        key={addOpenCount}
        isOpen={isAdding}
        onClose={() => setIsAdding(false)}
        configuredProviders={rows
          .filter((row) => row.source === "config")
          .map((row) => row.name)}
      />
      {editing ? (
        <EditHostedProviderForm
          key={editing.name}
          provider={editing}
          isOpen
          onClose={() => setEditingProvider("")}
        />
      ) : null}

      <ConfirmDialog
        isOpen={isSweepOpen}
        onOpenChange={(open) => {
          if (!open) setIsSweepOpen(false)
        }}
        heading="Refresh and purge the catalog?"
        body={<CatalogSweepPreview preview={preview} />}
        confirmLabel="Refresh and purge"
        confirmVariant="danger"
        // The sweep's own pending state only: the preview dials every provider,
        // and folding it in would leave Cancel dead for that whole fan-out.
        isPending={sweep.isPending}
        isConfirmDisabled={!isPreviewShown}
        error={sweep.error}
        onConfirm={() => {
          // Nothing is confirmed against a list nobody has seen.
          if (!isPreviewShown) return
          sweep.mutate(undefined, { onSuccess: () => setIsSweepOpen(false) })
        }}
      />
      <ConfirmDialog
        isOpen={pendingDelete !== ""}
        onOpenChange={(open) => {
          if (!open) setPendingDelete("")
        }}
        heading="Remove this hosted provider?"
        body={
          deleting?.shadows_config
            ? `The stored key for ${pendingDelete} is deleted, and the provider defined in config.yml serves again, on the same models.`
            : `${pendingDelete} will stop being served, and its stored key is deleted. There is no way to recover the key; adding the provider again means entering it fresh.`
        }
        confirmLabel="Remove provider"
        isPending={remove.isPending}
        error={remove.error}
        onConfirm={() => {
          remove.mutate(pendingDelete, {
            onSuccess: () => {
              if (modelsFor === pendingDelete) setModelsFor("")
              setPendingDelete("")
              // Its counts and remove offers described a deployment that has
              // since changed.
              sweep.reset()
            },
          })
        }}
      />
    </div>
  )
}
