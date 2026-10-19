import type { ProviderHealth, ProviderInfo } from "@/client"
import { DataTable, type DataTableColumn } from "@/design-system/data/DataTable"
import { ErrorBanner } from "@/design-system/feedback/ErrorBanner"
import { TableScrollFrame } from "@/design-system/layout/TableScrollFrame"
import { HealthPill } from "@/features/providers/HealthPill"
import { useProviderHealth, useProviders } from "@/shared/api/providers"

// The instances stored through the deployment providers page are left out:
// the move banner carries them onto hosted providers instead.

export function ConfiguredInstancesSection() {
  const providers = useProviders()
  const health = useProviderHealth()
  const configured = (providers.data?.providers ?? []).filter(
    (provider) => provider.source === "config",
  )
  if (!providers.error && configured.length === 0) return null

  const healthByInstance = new Map<string, ProviderHealth>(
    (health.data?.providers ?? []).map((row) => [row.instance, row]),
  )
  const columns: DataTableColumn<ProviderInfo>[] = [
    {
      id: "instance",
      header: "Instance",
      isRowHeader: true,
      cell: (row) => row.instance,
    },
    {
      id: "type",
      header: "Provider",
      cell: (row) => <span className="text-muted">{row.name}</span>,
    },
    {
      id: "health",
      header: "Status",
      cell: (row) => <HealthPill health={healthByInstance.get(row.instance)} />,
    },
  ]

  return (
    <section
      aria-labelledby="configured-providers-heading"
      className="mt-8 flex flex-col gap-3"
    >
      <div className="flex flex-col gap-1">
        <h2 id="configured-providers-heading" className="text-title">
          Configured providers
        </h2>
        <p className="text-caption">
          Declared under <code>providers:</code> in config.yml or
          OTARI_CONFIG_YAML, and changed there. A request naming one is served
          on it ahead of any organization key or hosted provider.
        </p>
      </div>
      <ErrorBanner error={providers.error} />
      <TableScrollFrame className="otari-configured-providers-table">
        <DataTable
          ariaLabel="Configured providers"
          columns={columns}
          rows={configured}
          getRowKey={(row) => row.instance}
          isLoading={providers.isLoading}
        />
      </TableScrollFrame>
    </section>
  )
}
