import { useState } from "react"
import { FiPlus, FiRefreshCw } from "react-icons/fi"

import type { HostedModel, HostedProvider, HostedVendor } from "@/client"
import { Button } from "@/design-system/actions/Button"
import { EmptyMessage } from "@/design-system/feedback/EmptyMessage"
import { EmptyState } from "@/design-system/feedback/EmptyState"
import { ErrorBanner } from "@/design-system/feedback/ErrorBanner"
import { PageLoading } from "@/design-system/feedback/PageLoading"
import { Toggle } from "@/design-system/forms/Toggle"
import { OfferedKindTabs } from "@/features/models/OfferedKindTabs"
import {
  filteredEmptyContent,
  OfferedModelFilters,
  type ServingFilter,
  servedParam,
  useModelSearchTerm,
} from "@/features/models/OfferedModelFilters"
import type { KindFilter } from "@/features/models/offeredKinds"
import {
  useHostedModels,
  useRefreshHostedModels,
  useResetHostedModelPrices,
  useSetHostedVendor,
  useSwitchHostedVendorModels,
  useUpdateHostedModel,
} from "@/shared/api/hostedProviders"
import { orderedRegions } from "./dataLocations"
import { HostedModelForm } from "./HostedModelForm"
import { HostedModelsTable } from "./HostedModelsTable"
import { HostedVendorRow, type VendorBulkAction } from "./HostedVendorRow"
import { regionRatesOf } from "./modelPriceDraft"
import { PriceSyncDialog } from "./PriceSyncDialog"
import { refreshOutcome } from "./refreshOutcome"

// One hosted provider's offered models, under the vendors that make them.
//
// A provider that routes to others serves many vendors' models, so each vendor
// gets a row with its own switch, a count of what it serves and a menu of
// bulk actions; its models open beneath it with the same row as ever: price,
// rate source, serving switch and Edit price. A vendor switched off withholds
// every model of it, ones a refresh adds later included, and keeps each model's
// own switch for when it comes back on. A provider with one vendor shows its
// models flat, since a vendor row would only repeat the provider.

/** What the bulk menu's actions send, and how their outcome reads. */
const BULK: Record<
  Exclude<VendorBulkAction, "reset">,
  { enabled: boolean; onlyUnpriced: boolean }
> = {
  "serve-all": { enabled: true, onlyUnpriced: false },
  "withhold-all": { enabled: false, onlyUnpriced: false },
  "withhold-unpriced": { enabled: false, onlyUnpriced: true },
}

function plural(count: number, one: string, many: string) {
  return `${count} ${count === 1 ? one : many}`
}

/** The vendors a list shows, in the order the API counts them, each with its shown models. */
function groupByVendor(rows: HostedModel[], vendors: HostedVendor[]) {
  const byVendor = new Map<string, HostedModel[]>()
  for (const row of rows) {
    const vendor = row.vendor ?? "Other"
    byVendor.set(vendor, [...(byVendor.get(vendor) ?? []), row])
  }
  return vendors
    .filter(
      (vendor) =>
        byVendor.has(vendor.vendor) || (vendor.total === 0 && !vendor.enabled),
    )
    .map((vendor) => ({ vendor, rows: byVendor.get(vendor.vendor) ?? [] }))
}

/** The provider's totals, before the vendors: what it serves and what needs a look. */
function ModelSummary({ vendors }: { vendors: HostedVendor[] }) {
  const sum = (pick: (vendor: HostedVendor) => number) =>
    vendors.reduce((total, vendor) => total + pick(vendor), 0)
  const items = [
    plural(
      sum((vendor) => vendor.total),
      "model",
      "models",
    ),
    plural(vendors.length, "vendor", "vendors"),
    `${sum((vendor) => vendor.serving)} serving`,
    `${sum((vendor) => vendor.unpriced)} unpriced`,
    `${sum((vendor) => vendor.custom_differs)} custom ≠ default`,
    plural(
      vendors.filter((vendor) => !vendor.enabled).length,
      "vendor off",
      "vendors off",
    ),
  ]
  return (
    <ul
      aria-label="Models summary"
      className="flex flex-wrap border border-border"
    >
      {items.map((item) => (
        <li
          key={item}
          className="border-r border-border px-3 py-1.5 text-caption tabular-nums last:border-r-0"
        >
          {item}
        </li>
      ))}
    </ul>
  )
}

export function HostedProviderModelsPanel({
  provider,
}: {
  provider: HostedProvider
}) {
  const [kind, setKind] = useState<KindFilter>("")
  const [serving, setServing] = useState<ServingFilter>("")
  const [search, setSearch] = useState("")
  const term = useModelSearchTerm(search)
  const models = useHostedModels(
    provider.name,
    kind,
    term,
    servedParam(serving),
  )
  const update = useUpdateHostedModel(provider.name)
  const refresh = useRefreshHostedModels(provider.name)
  const setVendor = useSetHostedVendor(provider.name)
  const switchVendor = useSwitchHostedVendorModels(provider.name)
  const resetPrices = useResetHostedModelPrices(provider.name)

  const [opened, setOpened] = useState<ReadonlySet<string>>(new Set())
  const [isAdding, setIsAdding] = useState(false)
  const [addOpenCount, setAddOpenCount] = useState(0)
  const [editingId, setEditingId] = useState("")
  // The region whose price of the edited model is open, or empty for the provider's.
  const [editingRegionSlug, setEditingRegionSlug] = useState("")
  const [isComparingRegions, setIsComparingRegions] = useState(false)
  // A column per region wherever the provider has one: a lone region left by a
  // removal may still carry a price of its own.
  const regions = orderedRegions(provider)
  const editingRegion = regions.find(
    (region) => region.slug === editingRegionSlug,
  )
  // A region editor whose region was removed under it is closed, never turned
  // into the provider's editor, whose save would reprice every region.
  const isEditorTargetGone =
    editingRegionSlug !== "" && editingRegion === undefined
  if (isEditorTargetGone) {
    setEditingRegionSlug("")
    setEditingId("")
  }
  // Counts the sync dialog's openings, so each opening asks for a fresh preview.
  const [syncOpening, setSyncOpening] = useState(0)
  const [isSyncOpen, setIsSyncOpen] = useState(false)
  const [syncVendor, setSyncVendor] = useState("")
  const [outcome, setOutcome] = useState("")

  const isEstimate = provider.provider === "otari"
  const rows = models.data?.data ?? []
  // A region gets a column once some model has a price of its own there, or
  // every region does while they are being compared.
  const shownRegions = isComparingRegions
    ? regions
    : regions.filter((region) =>
        rows.some((row) => regionRatesOf(row, region.slug) !== undefined),
      )
  const vendors = models.data?.vendors ?? []
  const editing = rows.find((row) => row.id === editingId)
  const refreshed = refresh.data ? refreshOutcome(refresh.data) : ""
  const status = outcome || refreshed

  const openAdd = () => {
    setEditingId("")
    setAddOpenCount((n) => n + 1)
    setIsAdding(true)
  }
  const openSync = (vendor: string) => {
    setSyncVendor(vendor)
    setSyncOpening((n) => n + 1)
    setIsSyncOpen(true)
  }
  const clearOutcome = () => {
    refresh.reset()
    setOutcome("")
  }

  const runBulk = (vendor: string, action: VendorBulkAction) => {
    if (action === "reset") {
      openSync(vendor)
      return
    }
    clearOutcome()
    const { enabled, onlyUnpriced } = BULK[action]
    switchVendor.mutate(
      { vendor, enabled, only_unpriced: onlyUnpriced },
      {
        onSuccess: (result) => {
          const changed = enabled
            ? `Served ${plural(result.changed, "more model", "more models")} by ${vendor}.`
            : `Withheld ${plural(result.changed, "model", "models")} by ${vendor}.`
          const skipped = [
            result.skipped_unpriced > 0
              ? `${result.skipped_unpriced} without a price`
              : "",
            (result.skipped_unavailable ?? 0) > 0
              ? `${result.skipped_unavailable} the provider refused to serve`
              : "",
          ].filter(Boolean)
          setOutcome(
            skipped.length > 0
              ? `${changed} Skipped ${skipped.join(" and ")}.`
              : changed,
          )
        },
      },
    )
  }

  const tableFor = (
    shown: HostedModel[],
    label: string,
    isVendorOff = false,
  ) => (
    <HostedModelsTable
      ariaLabel={label}
      rows={shown}
      isEstimate={isEstimate}
      isVendorOff={isVendorOff}
      isLoading={models.isLoading}
      emptyContent={filteredEmptyContent(term, serving)}
      pendingModelId={update.isPending ? (update.variables?.modelId ?? "") : ""}
      onToggle={(row, enabled) => {
        clearOutcome()
        update.mutate({ modelId: row.id, body: { enabled } })
      }}
      regions={shownRegions}
      hasRegions={regions.length > 0}
      onEdit={(row) => {
        setIsAdding(false)
        setEditingRegionSlug("")
        setEditingId(row.id)
      }}
      onEditRegionPrice={(row, region) => {
        setIsAdding(false)
        setEditingRegionSlug(region.slug)
        setEditingId(row.id)
      }}
      onUseDefault={(row) => {
        clearOutcome()
        resetPrices.mutate([row.model], {
          onSuccess: (result) =>
            setOutcome(
              result.applied
                ? `${row.model} now uses the default price.`
                : `${row.model} kept its price: it has no default price now.`,
            ),
        })
      }}
    />
  )

  if (models.isPending && !models.data) {
    return <PageLoading label="Loading models…" />
  }

  const groups = groupByVendor(rows, vendors)
  // A failed list is not an empty one, and a filter with no models left is an
  // empty list rather than "no models offered". Nor is a provider with a
  // vendor still switched off: its row is the way to switch it back on.
  const isEmptyCatalog =
    !models.error &&
    !models.isPlaceholderData &&
    rows.length === 0 &&
    groups.length === 0 &&
    kind === "" &&
    serving === "" &&
    term === ""
  // A provider's only vendor keeps its row, with its switch and menu, and
  // starts open, since there is nothing to choose between.
  const isSole = groups.length === 1 && groups[0].rows.length > 0
  // The list is read whole up to its limit; a provider offering more is cut.
  const isCut = models.data !== undefined && models.data.count > rows.length

  return (
    <section
      aria-label={`Models on ${provider.name}`}
      className="flex flex-col gap-3"
    >
      <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
        <div className="flex min-w-0 flex-col">
          <h2 className="text-title">Models</h2>
          <p className="text-caption">
            Prices in USD per million tokens for text models. An image, audio or
            video model's price is in its own unit, named when you hover it. A
            model offered without a price starts switched off.
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-2 sm:shrink-0">
          <Button
            size="sm"
            className="min-h-11 md:min-h-9"
            isPending={refresh.isPending}
            onPress={() => {
              setOutcome("")
              refresh.mutate()
            }}
          >
            <FiRefreshCw aria-hidden className="size-4" />
            Refresh models
          </Button>
          <Button
            size="sm"
            className="min-h-11 md:min-h-9"
            isDisabled={isEmptyCatalog}
            onPress={() => openSync("")}
          >
            Sync prices…
          </Button>
          <Button
            size="sm"
            variant="primary"
            className="min-h-11 md:min-h-9"
            onPress={openAdd}
          >
            <FiPlus aria-hidden className="size-4" />
            Add model
          </Button>
        </div>
      </div>

      <ErrorBanner
        error={
          models.error ??
          update.error ??
          refresh.error ??
          setVendor.error ??
          switchVendor.error ??
          resetPrices.error
        }
      />
      {regions.length > 0 && !isEmptyCatalog ? (
        <span className="flex items-center gap-2 self-start text-caption">
          Compare regions
          <Toggle
            isSelected={isComparingRegions}
            label="Compare regions: a price column for every region"
            onChange={setIsComparingRegions}
          />
        </span>
      ) : null}
      {status ? (
        <p className="text-caption" role="status">
          {status}
        </p>
      ) : null}

      {isEmptyCatalog ? (
        <EmptyState
          title="No models offered yet"
          description="Refresh to offer what the provider lists, or add a model to say what this provider serves and at what price."
          actionLabel="Refresh models"
          onAction={() => refresh.mutate()}
          isActionDisabled={refresh.isPending}
        />
      ) : (
        <>
          {vendors.length > 0 ? <ModelSummary vendors={vendors} /> : null}
          <OfferedModelFilters
            providerName={provider.name}
            serving={serving}
            onServingChange={setServing}
            search={search}
            onSearchChange={setSearch}
          />
          {models.data ? (
            <OfferedKindTabs
              counts={models.data.type_counts}
              shown={kind}
              onShow={setKind}
            />
          ) : null}
          {isCut ? (
            <p className="text-caption" role="status">
              {`Showing the first ${rows.length} of ${models.data?.count} models. Search or filter to reach the rest.`}
            </p>
          ) : null}
          {groups.length === 0 ? (
            <EmptyMessage>
              {filteredEmptyContent(term, serving) ?? "No model matches."}
            </EmptyMessage>
          ) : (
            <ul
              aria-label={`Vendors on ${provider.name}`}
              className="flex flex-col divide-y divide-border-subtle border border-border"
            >
              {groups.map(({ vendor, rows: shown }) => {
                // A search or a filter opens every vendor it leaves, since
                // the reader asked to see those models.
                const isOpen =
                  term !== "" ||
                  serving !== "" ||
                  kind !== "" ||
                  // `opened` records a press, which closes a sole vendor.
                  opened.has(vendor.vendor) !== isSole
                return (
                  <li key={vendor.vendor} className="flex flex-col">
                    <HostedVendorRow
                      vendor={vendor}
                      isOpen={isOpen}
                      isSwitching={
                        setVendor.isPending &&
                        setVendor.variables?.vendor === vendor.vendor
                      }
                      isBulkPending={switchVendor.isPending}
                      onToggleOpen={() =>
                        setOpened((current) => {
                          const next = new Set(current)
                          if (!next.delete(vendor.vendor))
                            next.add(vendor.vendor)
                          return next
                        })
                      }
                      onSwitch={(enabled) => {
                        clearOutcome()
                        setVendor.mutate({ vendor: vendor.vendor, enabled })
                      }}
                      onBulk={(action) => runBulk(vendor.vendor, action)}
                    />
                    {isOpen ? (
                      <div className="border-t border-border-subtle bg-surface-subtle">
                        {vendor.enabled ? null : (
                          <p className="px-4 py-2 text-caption">
                            {vendor.vendor} is switched off on this provider, so
                            none of its models are served, including ones a
                            refresh adds. Each model keeps its own setting for
                            when the vendor is switched back on. Prices can
                            still be edited.
                          </p>
                        )}
                        {tableFor(
                          shown,
                          `${vendor.vendor} models on ${provider.name}`,
                          !vendor.enabled,
                        )}
                      </div>
                    ) : null}
                  </li>
                )
              })}
            </ul>
          )}
        </>
      )}

      <HostedModelForm
        key={addOpenCount}
        provider={provider}
        editing={undefined}
        isOpen={isAdding}
        onClose={() => setIsAdding(false)}
        onSaved={clearOutcome}
      />
      {editing && !isEditorTargetGone ? (
        // Remounted per row, so a second row's editor cannot open with the
        // first one's draft.
        <HostedModelForm
          key={`${editing.id}:${editingRegion?.slug ?? ""}`}
          provider={provider}
          editing={editing}
          region={editingRegion}
          isOpen
          onClose={() => setEditingId("")}
          onSaved={clearOutcome}
        />
      ) : null}
      <PriceSyncDialog
        provider={provider}
        opening={isSyncOpen ? syncOpening : 0}
        vendor={syncVendor}
        isOpen={isSyncOpen}
        onClose={() => setIsSyncOpen(false)}
        onApplied={(count, kept) => {
          setIsSyncOpen(false)
          refresh.reset()
          setOutcome(
            `Reset ${plural(count, "model", "models")} to the default price.${
              kept.length > 0
                ? ` Kept the price of ${kept.join(", ")}: no default price now.`
                : ""
            }`,
          )
        }}
      />
    </section>
  )
}
