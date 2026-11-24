import { Link, useNavigate } from "@tanstack/react-router"
import { type ReactNode, useState } from "react"
import { FiActivity, FiEdit2, FiPlus, FiTrash2 } from "react-icons/fi"

import type { HostedProvider, ProviderHealth } from "@/client"
import { Button } from "@/design-system/actions/Button"
import { ConfirmDialog } from "@/design-system/feedback/ConfirmDialog"
import { EmptyState } from "@/design-system/feedback/EmptyState"
import { ErrorBanner } from "@/design-system/feedback/ErrorBanner"
import { PageLoading } from "@/design-system/feedback/PageLoading"
import { Toggle } from "@/design-system/forms/Toggle"
import { Badge } from "@/design-system/indicators/Badge"
import { PageIntro } from "@/design-system/layout/PageIntro"
import { ConnectionTestRowResult } from "@/features/providers/ConnectionTest"
import { HealthPill } from "@/features/providers/HealthPill"
import {
  PLATFORM_KEYS_URL,
  PLATFORM_PROVIDER_NAME,
} from "@/features/providers/PlatformConnect"
import { ApiError } from "@/shared/api/client"
import { useDeploymentAdminAccess } from "@/shared/api/deployment"
import {
  useDeleteHostedProvider,
  useHostedProviders,
  useTestHostedProvider,
  useUpdateHostedProvider,
} from "@/shared/api/hostedProviders"
import { useProviderHealth, useProviders } from "@/shared/api/providers"
import { useDeployment } from "@/shared/hooks/useDeployment"

import { dataLocationLabel } from "./dataLocations"
import { EditHostedProviderForm } from "./EditHostedProviderForm"
import { HostedProviderModelsPanel } from "./HostedProviderModelsPanel"
import { RegionFormDialog } from "./RegionFormDialog"
import { RegionsCard } from "./RegionsCard"
import { routingLocalityLabel } from "./RoutingLocalityField"
import { formatCountries } from "./supportedCountries"

// One hosted provider on a page of its own: how it is reached and who it
// serves, its switch, test, edit and remove, then the models it offers under
// their vendors. Operator-only, gated as the list is, and read from the list
// the API serves, since a provider has no read of its own.
//
// A provider defined in config.yml shows its settings read-only, since the
// file is where they change, and its models are managed here like any other's.

function BackLink() {
  return (
    <nav aria-label="Breadcrumb" className="text-caption">
      <Link
        to="/admin/hosted-providers"
        className="text-link hover:text-link-hover"
      >
        ← Hosted providers
      </Link>
    </nav>
  )
}

/** One fact about the provider, label above value. */
function Setting({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div className="flex min-w-0 flex-col gap-0.5 border-border px-3 py-2 sm:border-r sm:last:border-r-0">
      <dt className="text-caption">{label}</dt>
      <dd className="min-w-0 text-body break-words">{children}</dd>
    </div>
  )
}

function ApiKeyValue({ provider }: { provider: HostedProvider }) {
  if (!provider.decryptable) return <Badge tone="warn">Key unreadable</Badge>
  if (provider.api_key_last4) {
    return <code className="text-muted">••••{provider.api_key_last4}</code>
  }
  if (provider.has_api_key) return <code className="text-muted">••••</code>
  return <span className="text-muted">None needed</span>
}

/**
 * The provider's settings. Without regions, everything about its one
 * connection. With regions, where data is kept, the endpoint and who may call
 * belong to each region, so the grid keeps only what applies provider-wide and
 * a smaller block names what regions share unless they set their own.
 */
function ProviderSettings({ provider }: { provider: HostedProvider }) {
  const hasRegions = (provider.regions?.length ?? 0) > 0
  const serves = (
    <Setting label="Serves">
      {provider.restricted_to_organization_id ? (
        <span className="flex flex-col gap-0.5">
          <span>One organization</span>
          <code className="truncate text-caption">
            {provider.restricted_to_organization_id}
          </code>
        </span>
      ) : (
        <span className="text-muted">Every organization</span>
      )}
    </Setting>
  )
  const providerWide = (
    <>
      <Setting label="Zero data retention">
        {provider.zero_data_retention ? "Declared" : "Not declared"}
      </Setting>
      <Setting label="Routing locality">
        {provider.routing_locality ? (
          routingLocalityLabel(provider.routing_locality)
        ) : (
          <span className="text-muted">Not declared</span>
        )}
      </Setting>
      <Setting label="Models not on its list">
        {provider.serves_unlisted_models ? "Served" : "Refused"}
      </Setting>
    </>
  )
  if (hasRegions) {
    const clientArgs = Object.keys(provider.client_args ?? {})
    return (
      <div className="flex flex-col gap-3">
        <dl
          aria-label={`${provider.name} settings`}
          className="grid grid-cols-2 border border-border sm:grid-cols-3"
        >
          {serves}
          {providerWide}
        </dl>
        <div className="flex flex-col gap-1">
          <h2 className="text-overline">Shared by regions</h2>
          <dl
            aria-label={`Shared by ${provider.name}'s regions`}
            className="grid grid-cols-2 border border-border sm:grid-cols-3"
          >
            <Setting label="API key">
              <ApiKeyValue provider={provider} />
            </Setting>
            <Setting label="Fallback endpoint">
              <span className="text-muted">
                {provider.api_base ?? "Provider default"}
              </span>
            </Setting>
            <Setting label="Client args">
              {clientArgs.length > 0 ? (
                <code className="text-caption">{clientArgs.join(", ")}</code>
              ) : (
                <span className="text-muted">None</span>
              )}
            </Setting>
          </dl>
        </div>
      </div>
    )
  }
  return (
    <dl
      aria-label={`${provider.name} settings`}
      className="grid grid-cols-2 border border-border sm:grid-cols-3 xl:grid-cols-7"
    >
      <Setting label="API key">
        <ApiKeyValue provider={provider} />
      </Setting>
      <Setting label="Endpoint">
        <span className="text-muted">
          {provider.api_base ?? "Provider default"}
        </span>
      </Setting>
      {serves}
      <Setting label="Data hosted in">
        {provider.data_location ? (
          dataLocationLabel(provider.data_location)
        ) : (
          <span className="text-muted">Not declared</span>
        )}
      </Setting>
      <Setting label="Requests from">
        {provider.supported_countries == null ? (
          <span className="text-muted">Every country</span>
        ) : provider.supported_countries.length === 0 ? (
          "None"
        ) : (
          formatCountries(provider.supported_countries)
        )}
      </Setting>
      {providerWide}
    </dl>
  )
}

export function HostedProviderPage({ name }: { name: string }) {
  const access = useDeploymentAdminAccess()
  const isGranted = access.data === true
  const providers = useHostedProviders(isGranted, { isFresh: true })
  const health = useProviderHealth(isGranted)
  const configured = useProviders(isGranted)
  const update = useUpdateHostedProvider()
  const remove = useDeleteHostedProvider()
  const test = useTestHostedProvider()
  const navigate = useNavigate()
  const { platform_connect: isConnectOffered } = useDeployment()
  const [isEditing, setIsEditing] = useState(false)
  const [editOpening, setEditOpening] = useState(0)
  const [isRemoving, setIsRemoving] = useState(false)
  const [isAddingRegion, setIsAddingRegion] = useState(false)
  const [regionOpening, setRegionOpening] = useState(0)

  if (access.isLoading || (isGranted && providers.isPending)) {
    return <PageLoading label="Loading hosted provider…" />
  }

  if (access.isError) {
    return (
      <div className="flex flex-col">
        <BackLink />
        <PageIntro title={name} />
        <ErrorBanner error={access.error} />
      </div>
    )
  }

  // The list's 404 is the router's operator refusal, worded as the list words it.
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

  if (providers.error) {
    return (
      <div className="flex flex-col">
        <BackLink />
        <PageIntro title={name} />
        <ErrorBanner error={providers.error} />
      </div>
    )
  }

  const provider = providers.data?.data.find((row) => row.name === name)
  if (!provider) {
    // A cached list read before the provider was added is not an answer.
    if (providers.isFetching) {
      return <PageLoading label="Loading hosted provider…" />
    }
    return (
      <div className="flex flex-col gap-2">
        <BackLink />
        <EmptyState
          title={`No hosted provider named ${name}`}
          description="It may have been removed. The list shows every provider this deployment serves on its own keys."
        />
      </div>
    )
  }

  const isConfig = provider.source === "config"
  // Health is keyed by name, and a configured instance of the same name is the
  // one dialed under it, so its status would read as this provider's.
  const isShadowedByInstance =
    !configured.isSuccess ||
    configured.data.providers.some((row) => row.instance === name)
  const healthRow: ProviderHealth | undefined = isShadowedByInstance
    ? undefined
    : health.data?.providers.find((row) => row.instance === name)

  return (
    <div className="flex flex-col gap-6">
      <div className="flex flex-col gap-2">
        <BackLink />
        <PageIntro
          title={provider.name}
          action={
            <div className="flex flex-wrap items-center gap-3">
              <span className="flex items-center gap-2 text-caption">
                Serving
                <Toggle
                  isSelected={provider.enabled}
                  // A configured provider is switched in the configuration.
                  isDisabled={isConfig || update.isPending}
                  label={`Serve traffic on ${provider.name}`}
                  onChange={(enabled) =>
                    update.mutate({
                      provider: provider.name,
                      body: { enabled },
                    })
                  }
                />
              </span>
              {isConfig ? null : (
                <>
                  <Button
                    className="min-h-11 md:min-h-9"
                    aria-label={`Test ${provider.name}`}
                    // A key no secret reads cannot be tested; Edit replaces it.
                    isDisabled={test.isPending || !provider.decryptable}
                    onPress={() =>
                      test.mutate({
                        provider: provider.provider,
                        name: provider.name,
                      })
                    }
                  >
                    <FiActivity aria-hidden className="size-4" />
                    Test
                  </Button>
                  <Button
                    className="min-h-11 md:min-h-9"
                    aria-label={`Edit ${provider.name}`}
                    onPress={() => {
                      setEditOpening((n) => n + 1)
                      setIsEditing(true)
                    }}
                  >
                    <FiEdit2 aria-hidden className="size-4" />
                    Edit
                  </Button>
                  <Button
                    className="min-h-11 md:min-h-9"
                    aria-label={`Add a region to ${provider.name}`}
                    onPress={() => {
                      setRegionOpening((n) => n + 1)
                      setIsAddingRegion(true)
                    }}
                  >
                    <FiPlus aria-hidden className="size-4" />
                    Add region
                  </Button>
                  <Button
                    variant="danger"
                    className="min-h-11 md:min-h-9"
                    aria-label={`Remove ${provider.name}`}
                    onPress={() => setIsRemoving(true)}
                  >
                    <FiTrash2 aria-hidden className="size-4" />
                    Remove
                  </Button>
                </>
              )}
            </div>
          }
          detail={
            <span className="inline-flex flex-wrap items-center gap-2">
              {provider.name === provider.provider ? null : (
                <span>{provider.provider}</span>
              )}
              {/* No pill before a check has run: its placeholder dash reads
                as a stray mark under a title, where a table cell needs one. */}
              {!provider.enabled ? (
                <span>Switched off</span>
              ) : healthRow ? (
                <HealthPill health={healthRow} />
              ) : null}
              {isConfig ? <Badge tone="muted">config.yml</Badge> : null}
              {provider.shadows_config ? (
                <Badge tone="muted">Overrides config.yml</Badge>
              ) : null}
            </span>
          }
        />
        <ConnectionTestRowResult
          isPending={test.isPending}
          data={test.data}
          error={test.error}
        />
        <ErrorBanner error={update.error ?? remove.error} />
        {isConfig ? (
          <p className="text-caption">
            This provider is defined in config.yml: change its key, endpoint or
            switch there. Its models are managed here.
          </p>
        ) : null}
        <ProviderSettings provider={provider} />
      </div>

      {/* Shown from one region on, since a lone region left by a removal is
          still where the provider serves from and has to stay editable. */}
      {(provider.regions?.length ?? 0) > 0 ? (
        <RegionsCard provider={provider} isReadOnly={isConfig} />
      ) : null}

      <HostedProviderModelsPanel key={provider.name} provider={provider} />

      {isConfig ? null : (
        <RegionFormDialog
          key={`${provider.name}:region:${regionOpening}`}
          provider={provider}
          isOpen={isAddingRegion}
          onClose={() => setIsAddingRegion(false)}
        />
      )}
      {isConfig ? null : (
        <EditHostedProviderForm
          key={`${provider.name}:${editOpening}`}
          provider={provider}
          isOpen={isEditing}
          onClose={() => setIsEditing(false)}
        />
      )}
      <ConfirmDialog
        isOpen={isRemoving}
        onOpenChange={(open) => {
          if (!open) setIsRemoving(false)
        }}
        heading="Remove this hosted provider?"
        body={
          <>
            {provider.shadows_config
              ? `The stored key for ${provider.name} is deleted, and the provider defined in config.yml serves again, on the same models.`
              : `${provider.name} will stop being served, and its stored key is deleted. There is no way to recover the key; adding the provider again means entering it fresh.`}
            {isConnectOffered && provider.name === PLATFORM_PROVIDER_NAME ? (
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
        onConfirm={() =>
          remove.mutate(provider.name, {
            onSuccess: () => {
              setIsRemoving(false)
              void navigate({ to: "/admin/hosted-providers" })
            },
          })
        }
      />
    </div>
  )
}
