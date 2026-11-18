import { useState } from "react"
import { FiArrowDown, FiArrowUp, FiEdit2, FiTrash2 } from "react-icons/fi"

import type { HostedProvider, HostedProviderRegion } from "@/client"
import { CopyButton } from "@/design-system/actions/CopyButton"
import { RowMenu } from "@/design-system/actions/RowMenu"
import { DataTable, type DataTableColumn } from "@/design-system/data/DataTable"
import { ConfirmDialog } from "@/design-system/feedback/ConfirmDialog"
import { ErrorBanner } from "@/design-system/feedback/ErrorBanner"
import { Toggle } from "@/design-system/forms/Toggle"
import { Badge } from "@/design-system/indicators/Badge"
import {
  useRemoveHostedProviderRegion,
  useUpdateHostedProviderRegion,
} from "@/shared/api/hostedProviders"

import { dataLocationLabel, orderedRegions } from "./dataLocations"
import { RegionFormDialog } from "./RegionFormDialog"
import { formatCountries } from "./supportedCountries"

/** Who may call a connection: its list, or every country where it has none. */
function countriesText(countries: readonly string[] | null | undefined) {
  if (countries == null) return "Every country"
  return countries.length === 0 ? "None" : formatCountries([...countries])
}

/**
 * Where a switched-off region's requests go. An unpinned request goes to the
 * first remaining serving region that admits it (its country, and the
 * workspace's allowed locations), and is refused when none does, so the next
 * region in order is only the likely one. A request pinned to this region is
 * refused while it is off.
 */
export function switchOffConsequence(
  regions: readonly HostedProviderRegion[],
  region: HostedProviderRegion,
  providerName: string,
): string {
  const others = regions.filter(
    (other) => other.enabled && other.slug !== region.slug,
  )
  const next = regions
    .slice(regions.indexOf(region) + 1)
    .find((other) => other.enabled)
  const pinned = `A request that pins ${region.selector} is refused while it is off.`
  if (others.length === 0) {
    return `No region serves ${providerName} then, so every request to it is refused.`
  }
  const likely = next
    ? ` ${next.label}, the next serving region in order, is the likely one.`
    : ""
  return `Its requests go to the first other serving region that admits them, by the caller's country and the workspace's allowed data locations, and are refused where none does.${likely} ${pinned}`
}

/**
 * The regions a hosted provider serves from, numbered in the order requests
 * try them. A request goes to the first serving region its workspace allows,
 * with no failover to the next; `provider@slug:model` pins one. What a region
 * does not set of its own (key, endpoint, countries) it shares with the
 * provider. A provider defined in config.yml lists its regions read-only.
 */
export function RegionsCard({
  provider,
  isReadOnly,
}: {
  provider: HostedProvider
  isReadOnly: boolean
}) {
  const update = useUpdateHostedProviderRegion(provider.name)
  const remove = useRemoveHostedProviderRegion(provider.name)
  const [editingSlug, setEditingSlug] = useState("")
  const [editOpening, setEditOpening] = useState(0)
  const [removingSlug, setRemovingSlug] = useState("")
  const [switchingOffSlug, setSwitchingOffSlug] = useState("")

  const regions = orderedRegions(provider)
  const editing = regions.find((region) => region.slug === editingSlug)
  const removing = regions.find((region) => region.slug === removingSlug)
  const switchingOff = regions.find(
    (region) => region.slug === switchingOffSlug,
  )
  const example = regions[1]?.selector ?? regions[0]?.selector ?? ""

  const move = (slug: string, position: number) =>
    update.mutate({ slug, body: { position } })

  const columns: DataTableColumn<HostedProviderRegion>[] = [
    {
      id: "order",
      header: "Order",
      cell: (row) => (
        <span className="text-mono-caption tabular-nums text-muted">
          {regions.indexOf(row) + 1}
        </span>
      ),
    },
    {
      id: "region",
      header: "Region",
      isRowHeader: true,
      cell: (row) => (
        <span className="flex flex-col gap-0.5">
          <span className="flex flex-wrap items-center gap-2">
            <span className="font-medium text-foreground">{row.label}</span>
            {row.decryptable ? null : <Badge tone="warn">Key unreadable</Badge>}
          </span>
          <span className="flex items-center gap-1">
            <code className="min-w-0 text-caption break-all">
              {row.selector}
            </code>
            <span className="shrink-0">
              <CopyButton
                value={row.selector}
                label={`Copy ${row.label}'s selector`}
              />
            </span>
          </span>
        </span>
      ),
    },
    {
      id: "location",
      header: "Data hosted in",
      cell: (row) =>
        row.data_location ? (
          dataLocationLabel(row.data_location)
        ) : (
          <span className="text-muted">Not declared</span>
        ),
    },
    {
      id: "key",
      header: "Key",
      cell: (row) =>
        row.has_api_key ? (
          <code className="text-muted">Own ••{row.api_key_last4 ?? "••"}</code>
        ) : (
          <span className="text-muted">Shared key</span>
        ),
    },
    {
      id: "endpoint",
      header: "Endpoint",
      cell: (row) =>
        row.api_base ? (
          <span className="break-all">{row.api_base}</span>
        ) : (
          <span className="text-muted">Shared endpoint</span>
        ),
    },
    {
      id: "countries",
      header: "Requests from",
      cell: (row) =>
        row.supported_countries == null ? (
          // The provider's list, shared by every region without its own.
          <span className="text-muted">
            {countriesText(provider.supported_countries)}
          </span>
        ) : (
          countriesText(row.supported_countries)
        ),
    },
    {
      id: "enabled",
      header: "Serving",
      cell: (row) => (
        <Toggle
          isSelected={row.enabled}
          isDisabled={
            isReadOnly ||
            (update.isPending && update.variables?.slug === row.slug)
          }
          label={`Serve traffic from ${row.label}`}
          // Switching off moves traffic elsewhere, so it is confirmed first.
          onChange={(enabled) => {
            if (enabled) {
              update.mutate({ slug: row.slug, body: { enabled } })
              return
            }
            update.reset()
            setSwitchingOffSlug(row.slug)
          }}
        />
      ),
    },
    ...(isReadOnly
      ? []
      : [
          {
            id: "actions",
            header: "Actions",
            align: "end",
            cell: (row) => {
              const index = regions.indexOf(row)
              return (
                <RowMenu
                  label={`Actions for ${row.label}`}
                  actions={[
                    {
                      id: "edit",
                      label: "Edit…",
                      icon: FiEdit2,
                      run: () => {
                        setEditOpening((n) => n + 1)
                        setEditingSlug(row.slug)
                      },
                    },
                    {
                      id: "up",
                      label: "Move up",
                      icon: FiArrowUp,
                      isDisabled: index === 0 || update.isPending,
                      run: () => move(row.slug, index - 1),
                    },
                    {
                      id: "down",
                      label: "Move down",
                      icon: FiArrowDown,
                      isDisabled:
                        index === regions.length - 1 || update.isPending,
                      run: () => move(row.slug, index + 1),
                    },
                    {
                      id: "remove",
                      label: "Remove…",
                      icon: FiTrash2,
                      isDestructive: true,
                      run: () => setRemovingSlug(row.slug),
                    },
                  ]}
                />
              )
            },
          } satisfies DataTableColumn<HostedProviderRegion>,
        ]),
  ]

  return (
    <section
      aria-label={`Regions of ${provider.name}`}
      className="flex flex-col gap-3"
    >
      <div className="flex min-w-0 flex-col">
        <h2 className="text-title">Regions</h2>
        <p className="text-caption">
          {`Tried in this order: requests go to the first serving region the workspace allows; ${example || `${provider.name}@<id>`}:model pins one.`}
        </p>
      </div>
      {/* A refused switch-off is shown in its own dialog. */}
      <ErrorBanner error={switchingOff ? null : update.error} />
      <DataTable
        ariaLabel={`Regions of ${provider.name}`}
        mobileLayout="cards"
        columns={columns}
        rows={regions}
        getRowKey={(row) => row.slug}
      />
      {editing && !isReadOnly ? (
        <RegionFormDialog
          key={`${editing.slug}:${editOpening}`}
          provider={provider}
          region={editing}
          isOpen
          onClose={() => setEditingSlug("")}
        />
      ) : null}
      <ConfirmDialog
        isOpen={switchingOff !== undefined}
        onOpenChange={(open) => {
          if (!open) setSwitchingOffSlug("")
        }}
        heading={`Stop serving from ${switchingOff?.label ?? "this region"}?`}
        body={
          switchingOff
            ? switchOffConsequence(regions, switchingOff, provider.name)
            : ""
        }
        confirmLabel="Switch off"
        isPending={update.isPending}
        error={update.error}
        onConfirm={() =>
          update.mutate(
            { slug: switchingOffSlug, body: { enabled: false } },
            { onSuccess: () => setSwitchingOffSlug("") },
          )
        }
      />
      <ConfirmDialog
        isOpen={removing !== undefined}
        onOpenChange={(open) => {
          if (!open) setRemovingSlug("")
        }}
        heading={`Remove ${removing?.label ?? "this region"}?`}
        body={
          regions.length <= 1
            ? `${provider.name} goes back to serving from its own connection. A key stored for this region is deleted.`
            : `Requests stop going to ${removing?.label ?? "this region"}, and a request that pins ${removing?.selector ?? "it"} is refused. A key stored for it is deleted.`
        }
        confirmLabel="Remove region"
        isPending={remove.isPending}
        error={remove.error}
        onConfirm={() =>
          remove.mutate(removingSlug, {
            onSuccess: () => setRemovingSlug(""),
          })
        }
      />
    </section>
  )
}
