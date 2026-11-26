import { Link, useNavigate } from "@tanstack/react-router"
import { type ReactNode, useState } from "react"
import { FiRefreshCw, FiTrash2 } from "react-icons/fi"

import type {
  HostedCatalogRefresh,
  HostedProvider,
  HostedProviderModelSummary,
  ProviderHealth,
  ProviderInfo,
} from "@/client"
import { Button } from "@/design-system/actions/Button"
import { DataTable, type DataTableColumn } from "@/design-system/data/DataTable"
import { ConfirmDialog } from "@/design-system/feedback/ConfirmDialog"
import { EmptyState } from "@/design-system/feedback/EmptyState"
import { ErrorBanner } from "@/design-system/feedback/ErrorBanner"
import { InfoBanner } from "@/design-system/feedback/InfoBanner"
import { PageLoading } from "@/design-system/feedback/PageLoading"
import { Checkbox } from "@/design-system/forms/Checkbox"
import { Toggle } from "@/design-system/forms/Toggle"
import { Badge } from "@/design-system/indicators/Badge"
import { PageIntro } from "@/design-system/layout/PageIntro"
import { TableScrollFrame } from "@/design-system/layout/TableScrollFrame"
import { HealthPill } from "@/features/providers/HealthPill"
import { HealthSummary } from "@/features/providers/HealthSummary"
import { OnboardingPanel } from "@/features/providers/OnboardingPanel"
import {
  PLATFORM_KEYS_URL,
  PLATFORM_PROVIDER_NAME,
  PlatformConnect,
} from "@/features/providers/PlatformConnect"
import { SecretKeyMissingBanner } from "@/features/providers/SecretKeyMissingBanner"
import { OWN_KEYS, useWelcomeAnswer } from "@/features/welcome/welcomeAnswer"
import { ApiError } from "@/shared/api/client"
import { useDeploymentAdminAccess } from "@/shared/api/deployment"
import {
  useDeleteHostedProvider,
  useHostedCatalogUpdate,
  useHostedCatalogUpdatePreview,
  useHostedProviders,
  usePlatformConnectStatus,
  useUpdateHostedProvider,
} from "@/shared/api/hostedProviders"
import { useOrganizationContext } from "@/shared/api/organizations"
import { useProviderHealth, useProviders } from "@/shared/api/providers"
import { useSettings, useUpdateSettings } from "@/shared/api/settings"
import { useDeployment } from "@/shared/hooks/useDeployment"

import { AddHostedProviderForm } from "./AddHostedProviderForm"
import { DeploymentProviderMigration } from "./DeploymentProviderMigration"
import {
  dataLocationLabel,
  providerDataLocations,
  servingRegions,
} from "./dataLocations"
import {
  catalogPreviewOutcome,
  catalogRefreshFailures,
  catalogRefreshOutcome,
  modelsPreviewOutcome,
  pricePreviewOutcome,
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
// The list says which providers there are, whether each is healthy and how
// much of its catalog it serves; a provider's name opens its own page
// (`HostedProviderPage`), where its key, endpoint, scope and models are.
//
// A provider defined in the configuration's `hosted_providers:` block is listed
// with a `config.yml` badge and changed only there: its key, endpoint and switch
// are read-only, while its models are administered as any other's. A
// `providers:` instance is listed beside them, wholly read-only and opening no
// page: it serves every model a request names on it, ahead of every hosted
// provider.

/** A row of the table: a hosted provider, or a `providers:` instance from config.yml. */
type ProviderRow =
  | { kind: "hosted"; name: string; hosted: HostedProvider }
  | { kind: "instance"; name: string; instance: ProviderInfo }

/**
 * What the last catalog update did, stated rather than implied by the table
 * redrawing.
 *
 * The kept groups are worth the space: a cleanup that leaves priced models under
 * a provider this page does not offer looks like it failed, when something else
 * on the deployment serves them.
 */
function CatalogUpdateReport({
  result,
  pricesApplied,
  onRemoveProvider,
}: {
  result: HostedCatalogRefresh
  pricesApplied: boolean
  onRemoveProvider: (provider: string) => void
}) {
  const failures = catalogRefreshFailures(result)
  return (
    <InfoBanner tone={failures.length > 0 ? "warning" : "info"}>
      <p role="status">
        {pricesApplied ? "Applied the latest models.dev prices. " : ""}
        {catalogRefreshOutcome(result)}
      </p>
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

/**
 * What a catalog update is about to do: the price changes, the models each
 * provider newly lists, and, when the cleanup is ticked, the models it removes,
 * named before anything is removed.
 */
function CatalogUpdatePreview({
  preview,
  cleanup,
  onCleanupChange,
}: {
  preview: ReturnType<typeof useHostedCatalogUpdatePreview>
  cleanup: boolean
  onCleanupChange: (cleanup: boolean) => void
}) {
  const body = () => {
    if (preview.error) {
      // The confirm refuses to fire in this state, so the banner is the answer.
      return <ErrorBanner error={preview.error} />
    }
    if (preview.isPending || preview.isFetching) {
      return <p>Asking models.dev and every provider what changed…</p>
    }
    const { catalog, prices } = preview.data
    const failures = catalogRefreshFailures(catalog)
    return (
      <>
        <p>{pricePreviewOutcome(preview.data)}</p>
        {prices?.needs_review ? (
          <p className="text-caption text-warning">
            Unusual: {prices.review_reason}. A scheduled update would wait for a
            person.
          </p>
        ) : null}
        <p>{modelsPreviewOutcome(catalog)}</p>
        {failures.map((failure) => (
          <p key={failure.provider} className="text-caption">
            {failure.message}
          </p>
        ))}
        {cleanup ? (
          <>
            <p>{catalogPreviewOutcome(catalog)}</p>
            {catalog.removed.length > 0 ? (
              <ul className="max-h-48 overflow-y-auto font-mono text-caption">
                {catalog.removed.map((modelKey) => (
                  <li key={modelKey}>{modelKey}</li>
                ))}
              </ul>
            ) : null}
            {catalog.removed_count > catalog.removed.length ? (
              <p className="text-caption">
                … and {catalog.removed_count - catalog.removed.length} more.
              </p>
            ) : null}
            {catalog.kept.map((group) => (
              <p key={group.reason} className="text-caption">
                Keeping {group.count} {group.count === 1 ? "model" : "models"}:{" "}
                {group.reason}.
              </p>
            ))}
          </>
        ) : null}
      </>
    )
  }
  return (
    <div className="flex flex-col gap-3">
      {body()}
      <Checkbox isSelected={cleanup} onChange={onCleanupChange}>
        Also remove the prices of models no provider serves
      </Checkbox>
      {cleanup ? (
        <p className="text-caption">
          Removing a model takes its rate history and the overrides of
          organizations with no key of their own for the provider. Requests
          already billed keep the cost they were billed at.
        </p>
      ) : null}
    </div>
  )
}

/** "12 of 18 serving", with what holds back the rest when anything does. */
function ModelsSummaryCell({
  summary,
}: {
  summary: HostedProviderModelSummary
}) {
  if (summary.total === 0) return <span className="text-muted">None yet</span>
  return (
    <span className="flex flex-wrap items-center gap-2">
      <span className="tabular-nums">
        {summary.serving} of {summary.total} serving
      </span>
      {summary.vendors_off > 0 ? (
        <Badge tone="warn">
          {`${summary.vendors_off} ${summary.vendors_off === 1 ? "vendor off" : "vendors off"}`}
        </Badge>
      ) : null}
      {summary.unpriced > 0 ? (
        <Badge tone="warn">{`${summary.unpriced} unpriced`}</Badge>
      ) : null}
    </span>
  )
}

/**
 * How many regions a provider serves from, where it has several, and every
 * place they keep data, each region's own. A switched-off region serves
 * nothing, so it counts toward neither. Nothing declared says so, since a
 * workspace limited to certain locations refuses such a provider.
 */
function DataLocationBadges({ provider }: { provider: HostedProvider }) {
  const regionCount = servingRegions(provider).length
  const locations = providerDataLocations(provider)
  const names = [
    ...locations.filter((location) => location !== null).map(dataLocationLabel),
    ...(locations.includes(null) ? ["not declared"] : []),
  ]
  return (
    <>
      {regionCount >= 2 ? (
        <Badge tone="muted">{`${regionCount} regions`}</Badge>
      ) : null}
      {names.length > 0 ? (
        <Badge tone="muted">{`Data: ${names.join(", ")}`}</Badge>
      ) : null}
    </>
  )
}

export function HostedProvidersPage() {
  const access = useDeploymentAdminAccess()
  // Withheld until the gate answers, so a non-operator load does not put a 404
  // in the console to learn what the gate is about to say.
  const isGranted = access.data === true
  const providers = useHostedProviders(isGranted)
  const health = useProviderHealth(isGranted)
  const configured = useProviders(isGranted)
  const context = useOrganizationContext(isGranted)
  // Only a context that answered says the key is unset; a failed read is not
  // evidence of it, and the server refuses the write either way.
  const isKeyEncryptionMissing =
    context.data?.provider_key_encryption_available === false
  const settings = useSettings(isGranted)
  const updateSettings = useUpdateSettings()
  const navigate = useNavigate()
  const { platform_connect: isConnectOffered } = useDeployment()
  // Read here as well as in the card, so a code still waiting for approval is
  // shown on a page that was reloaded, without the operator asking for it.
  const connectStatus = usePlatformConnectStatus(isGranted && isConnectOffered)
  const [isConnectOpen, setIsConnectOpen] = useState(false)
  // Skipping the welcome takes otari.ai out of the empty page's first-run
  // panel, so the header keeps offering it there.
  const answeredOwnKeys = useWelcomeAnswer() === OWN_KEYS
  const update = useUpdateHostedProvider()
  const remove = useDeleteHostedProvider()
  const catalogUpdate = useHostedCatalogUpdate()
  const [isUpdateOpen, setIsUpdateOpen] = useState(false)
  const [updateOpenCount, setUpdateOpenCount] = useState(0)
  const [cleanup, setCleanup] = useState(false)
  // Asked only while the dialog is open, because answering dials every
  // configured provider, and asked afresh on every opening.
  const preview = useHostedCatalogUpdatePreview(
    isUpdateOpen ? updateOpenCount : 0,
    cleanup,
  )
  // The confirm waits for a preview this opening read successfully and shows:
  // never a previous opening's, a failed or an in-flight one, because the
  // update applies the prices that preview named.
  const isPreviewShown = preview.isSuccess && !preview.isFetching

  const [isAdding, setIsAdding] = useState(false)
  const [addOpenCount, setAddOpenCount] = useState(0)
  // A name rather than a row: the update report's own remove offer can name a
  // provider this page has not loaded.
  const [pendingDelete, setPendingDelete] = useState("")

  const openAdd = () => {
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
  const tableRows: ProviderRow[] = [
    ...(configured.data?.providers ?? []).map(
      (instance): ProviderRow => ({
        kind: "instance",
        name: instance.instance,
        instance,
      }),
    ),
    ...rows.map(
      (hosted): ProviderRow => ({ kind: "hosted", name: hosted.name, hosted }),
    ),
  ].sort((a, b) => a.name.localeCompare(b.name))
  // A failed list is not an empty one: rendering "none configured" over a 500
  // would tell an operator their live providers are gone.
  const isEmpty =
    !providers.error && !providers.isPending && tableRows.length === 0
  const isConnectInHeader =
    isConnectOffered && (tableRows.length > 0 || answeredOwnKeys)

  const hostedColumns: DataTableColumn<HostedProvider>[] = [
    {
      id: "provider",
      header: "Provider",
      isRowHeader: true,
      cell: (row) => (
        <span className="flex flex-wrap items-center gap-2">
          <Link
            to="/admin/hosted-providers/$name"
            params={{ name: row.name }}
            className="font-medium text-foreground hover:text-link hover:underline"
          >
            {row.name}
          </Link>
          {row.name === row.provider ? null : (
            <span className="text-caption">{row.provider}</span>
          )}
          {row.source === "config" ? (
            <Badge tone="muted">config.yml</Badge>
          ) : null}
          {row.shadows_config ? (
            <Badge tone="muted">Overrides config.yml</Badge>
          ) : null}
          {row.zero_data_retention ? (
            <Badge tone="muted">Zero data retention</Badge>
          ) : null}
          <DataLocationBadges provider={row} />
          {row.decryptable ? null : <Badge tone="warn">Key unreadable</Badge>}
        </span>
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
      id: "models",
      header: "Models",
      cell: (row) =>
        row.models ? (
          <ModelsSummaryCell summary={row.models} />
        ) : (
          <span className="text-muted">—</span>
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
  ]
  const instanceCells: Record<string, (row: ProviderInfo) => ReactNode> = {
    provider: (row) => (
      <span className="flex flex-wrap items-center gap-2">
        <span className="font-medium text-foreground">{row.instance}</span>
        {row.instance === row.provider_type ? null : (
          <span className="text-caption">{row.provider_type}</span>
        )}
        <Badge tone="muted">config.yml</Badge>
      </span>
    ),
    health: (row) => <HealthPill health={healthByName.get(row.instance)} />,
    models: () => (
      <span className="text-muted">Every model a request names</span>
    ),
    enabled: () => <span className="text-caption">Always</span>,
  }
  const columns: DataTableColumn<ProviderRow>[] = hostedColumns.map(
    (column) => ({
      ...column,
      cell: (row: ProviderRow) =>
        row.kind === "hosted"
          ? column.cell(row.hosted)
          : instanceCells[column.id]?.(row.instance),
    }),
  )

  return (
    <div className="flex flex-col">
      <PageIntro
        title="Hosted providers"
        action={
          <div className="flex flex-wrap items-center gap-2">
            <Button
              className="min-h-11 md:min-h-9"
              isPending={catalogUpdate.isPending}
              onPress={() => {
                setCleanup(false)
                setUpdateOpenCount((n) => n + 1)
                setIsUpdateOpen(true)
              }}
            >
              <FiRefreshCw aria-hidden className="size-4" />
              Update catalog
            </Button>
            {isConnectInHeader ? (
              <Button
                className="min-h-11 md:min-h-9"
                isDisabled={isKeyEncryptionMissing}
                onPress={() => setIsConnectOpen((open) => !open)}
              >
                Connect otari.ai
              </Button>
            ) : null}
            <Button
              variant="primary"
              className="min-h-11 md:min-h-9"
              // Disabled rather than hidden, with its reason in the banner.
              isDisabled={isKeyEncryptionMissing}
              onPress={openAdd}
            >
              Add provider
            </Button>
          </div>
        }
      />

      <ErrorBanner
        error={
          providers.error ??
          update.error ??
          remove.error ??
          catalogUpdate.error ??
          updateSettings.error
        }
      />
      {isKeyEncryptionMissing ? <SecretKeyMissingBanner /> : null}
      <DeploymentProviderMigration />
      {catalogUpdate.data ? (
        <CatalogUpdateReport
          result={catalogUpdate.data.catalog}
          pricesApplied={catalogUpdate.data.prices_applied}
          onRemoveProvider={setPendingDelete}
        />
      ) : null}

      {isConnectInHeader &&
      !isKeyEncryptionMissing &&
      (isConnectOpen || connectStatus.data?.status === "pending") ? (
        <section
          aria-label="Connect otari.ai"
          className="mb-4 border border-border p-4"
        >
          <PlatformConnect
            // The empty page's own route to the setup guide, as the panel's.
            onConnected={isEmpty ? () => void navigate({ to: "/" }) : undefined}
          />
        </section>
      ) : null}
      {isEmpty ? (
        <OnboardingPanel
          onAddProvider={openAdd}
          // The settings read only feeds this advisory tip, so its failure
          // hides the tip rather than raising an error.
          needsPricing={
            settings.data?.require_pricing === true &&
            settings.data.default_pricing === false
          }
          onEnablePricing={() =>
            updateSettings.mutate({ default_pricing: true })
          }
          isEnabling={updateSettings.isPending}
          secretKeyConfigured={!isKeyEncryptionMissing}
          // On to the Overview, whose setup guide hands out a key and the
          // first request to send now that a provider can serve it.
          onConnected={() => void navigate({ to: "/" })}
        />
      ) : (
        <>
          {tableRows.length > 0 && health.data && health.data.total > 0 ? (
            <HealthSummary
              healthy={health.data.healthy}
              degraded={health.data.degraded}
              total={health.data.total}
              checkedAt={health.data.checked_at ?? ""}
            />
          ) : null}
          <TableScrollFrame className="otari-hosted-providers-table">
            <DataTable
              ariaLabel="Hosted providers"
              mobileLayout="cards"
              columns={columns}
              rows={tableRows}
              // The kind and the name, not the id: a configured provider has no
              // row id, and a hosted provider may share a configured instance's name.
              getRowKey={(row) => `${row.kind}:${row.name}`}
              isLoading={providers.isLoading}
            />
          </TableScrollFrame>
        </>
      )}

      <AddHostedProviderForm
        key={addOpenCount}
        // Retracted too if the context reports the key unset after it opened.
        isOpen={isAdding && !isKeyEncryptionMissing}
        onClose={() => setIsAdding(false)}
        // On to the new provider's page, where its models are the next step.
        onCreated={(created) =>
          void navigate({
            to: "/admin/hosted-providers/$name",
            params: { name: created.name },
          })
        }
        configuredProviders={rows
          .filter((row) => row.source === "config")
          .map((row) => row.name)}
      />
      <ConfirmDialog
        isOpen={isUpdateOpen}
        onOpenChange={(open) => {
          if (!open) setIsUpdateOpen(false)
        }}
        heading="Update the catalog?"
        body={
          <CatalogUpdatePreview
            preview={preview}
            cleanup={cleanup}
            onCleanupChange={setCleanup}
          />
        }
        confirmLabel={cleanup ? "Update and remove" : "Update catalog"}
        // Danger only once it deletes: without the cleanup it only adds and reprices.
        confirmVariant={cleanup ? "danger" : "primary"}
        // The update's own pending state only: the preview dials every provider,
        // and folding it in would leave Cancel dead for that whole fan-out.
        isPending={catalogUpdate.isPending}
        isConfirmDisabled={!isPreviewShown}
        error={catalogUpdate.error}
        onConfirm={() => {
          // Nothing is confirmed against a preview nobody has seen.
          if (!isPreviewShown) return
          catalogUpdate.mutate(
            { digest: preview.data.prices?.digest ?? null, cleanup },
            { onSuccess: () => setIsUpdateOpen(false) },
          )
        }}
      />
      <ConfirmDialog
        isOpen={pendingDelete !== ""}
        onOpenChange={(open) => {
          if (!open) setPendingDelete("")
        }}
        heading="Remove this hosted provider?"
        body={
          <>
            {deleting?.shadows_config
              ? `The stored key for ${pendingDelete} is deleted, and the provider defined in config.yml serves again, on the same models.`
              : `${pendingDelete} will stop being served, and its stored key is deleted. There is no way to recover the key; adding the provider again means entering it fresh.`}
            {isConnectOffered && pendingDelete === PLATFORM_PROVIDER_NAME ? (
              <>
                {" "}
                The key stays valid on otari.ai until you revoke it on its{" "}
                <a
                  href={PLATFORM_KEYS_URL}
                  target="_blank"
                  rel="noreferrer"
                  className="font-medium text-link hover:text-link-hover"
                >
                  API keys page
                </a>
                .
              </>
            ) : null}
          </>
        }
        confirmLabel="Remove provider"
        isPending={remove.isPending}
        error={remove.error}
        onConfirm={() => {
          remove.mutate(pendingDelete, {
            onSuccess: () => {
              setPendingDelete("")
              // Its counts and remove offers described a deployment that has
              // since changed.
              catalogUpdate.reset()
            },
          })
        }}
      />
    </div>
  )
}
