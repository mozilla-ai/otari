import { Button, Modal } from "@heroui/react"
import { Link } from "@tanstack/react-router"
import type { ReactNode } from "react"
import { useCallback, useState } from "react"
import type { SortDescriptor } from "react-aria-components"

import type { CatalogModelDetail, CatalogOffering } from "@/client"
import {
  CAPABILITY_LABELS,
  credentialLabel,
  defaultOffering,
  MODALITY_LABELS,
  priceSourceLabel,
} from "@/features/models/catalog"
import { publicCatalogHref } from "@/features/models/publicCatalog"
import { canManage, isDeploymentOperator } from "@/features/organization/roles"
import { useCatalog, useCatalogModel } from "@/shared/api/models"
import { useOrganizationContext } from "@/shared/api/organizations"
import { CopyableValue, CopyField } from "@/shared/components/actions/CopyField"
import {
  DataTable,
  type DataTableColumn,
} from "@/shared/components/data/DataTable"
import { ErrorBanner } from "@/shared/components/feedback/ErrorBanner"
import { InfoBanner } from "@/shared/components/feedback/InfoBanner"
import { PageLoading } from "@/shared/components/feedback/PageLoading"
import { Badge } from "@/shared/components/indicators/Badge"
import { Dot } from "@/shared/components/indicators/Dot"
import { TableScrollFrame } from "@/shared/components/layout/TableScrollFrame"
import { FilterSelect } from "@/shared/components/navigation/FilterSelect"
import { Tab, TabRow } from "@/shared/components/navigation/TabRow"
import {
  formatContext,
  formatRate,
  formatReleaseDate,
} from "@/shared/helpers/format"
import {
  buildCurlSnippet,
  buildPythonSnippet,
  resolveSnippetBaseUrl,
} from "@/shared/helpers/requestSnippets"
import { useUrlValue } from "@/shared/helpers/urlState"
import { useDeployment } from "@/shared/hooks/useDeployment"

// One model, on a page of its own: the header with its facts, then the
// sections a reader scrolls through, with a rail on the left that jumps to
// each. Providers is the one that matters most and comes first: every
// offering of the model this viewer may call, cheapest first, with the price
// they would be charged and where it came from. Pricing rolls up what the
// organization actually paid. "Use this model" opens the request to copy.
//
// Read-only for everyone (otari-ai#2095, #2096): a rate is edited on Model
// pricing, which the operator's link here points at, so the page that compares
// prices never becomes the page that changes them.

function Stat({ label, value }: { label: string; value: ReactNode }) {
  return (
    <div className="flex flex-col gap-1 border border-border bg-surface px-4 py-3">
      <span className="text-overline">{label}</span>
      <span className="text-body tabular-nums">{value}</span>
    </div>
  )
}

/**
 * Which price list an offering's rate came from, as a dot and a word.
 *
 * The organization's own rate is the one deliberate choice on the row and takes
 * the accent dot; a custom deployment rate and a genai-prices default are facts
 * about the deployment and read muted; unpriced is the absence and reads
 * subtle. The accent is data ink, not a way to say "look here".
 */
function SourceMark({ source }: { source: CatalogOffering["price_source"] }) {
  const word = priceSourceLabel(source)
  const dot =
    source === "organization"
      ? "bg-accent"
      : source === "deployment"
        ? "bg-foreground"
        : "bg-text-subtle"
  const ink =
    source === "organization" || source === "deployment"
      ? "text-foreground"
      : source === "defaults"
        ? "text-muted"
        : "text-subtle"
  return (
    <span className={`flex items-center gap-2 text-mono-caption ${ink}`}>
      <Dot className={dot} />
      {word.toUpperCase()}
    </span>
  )
}

// A popular open model is resold by dozens of providers models.dev lists, and
// the sentence exists to say "you could add one", not to be the list.
const ELSEWHERE_SHOWN = 6

function elsewhere(
  providers: CatalogModelDetail["also_available_from"],
): string {
  const names = providers.map((provider) => provider.name)
  if (names.length <= ELSEWHERE_SHOWN) return names.join(", ")
  const rest = names.length - ELSEWHERE_SHOWN
  return `${names.slice(0, ELSEWHERE_SHOWN).join(", ")} and ${rest} more`
}

function rate(value: number | null | undefined): string {
  // An em dash rather than $0.00: an offering with no cache-read rate is not
  // one that reads cache for free.
  return value == null ? "—" : formatRate(value)
}

// How far the metered rate may sit from the provider's published list price
// before the row says so. models.dev and genai-prices round differently, so a
// hair's width of disagreement is noise; a real gap is a stale stored price or
// a deliberate markup, and either is worth a glance.
const LIST_PRICE_TOLERANCE = 0.02

/** The provider's list price, where it disagrees with what is metered. */
export function listPriceNote(
  metered: number | null | undefined,
  listed: number | null | undefined,
): string | null {
  if (metered == null || listed == null || listed <= 0) return null
  if (Math.abs(metered - listed) / listed <= LIST_PRICE_TOLERANCE) return null
  return `list ${formatRate(listed)}`
}

function RateCell({
  metered,
  listed,
}: {
  metered: number | null | undefined
  listed: number | null | undefined
}) {
  const note = listPriceNote(metered, listed)
  return (
    <span className="flex flex-col items-end">
      <span className="text-mono-caption">{rate(metered)}</span>
      {note ? <span className="text-caption text-subtle">{note}</span> : null}
    </span>
  )
}

function percent(value: number | null): string {
  return value == null ? "—" : `${Math.round(value * 100)}%`
}

/** Whether any offering carries the organization's own last-30-day figures. */
function hasUsage(offerings: readonly CatalogOffering[]): boolean {
  return offerings.some((offering) => offering.usage_30d != null)
}

/** An offering with its base rates lifted out, for the table's sort. */
interface OfferingRow {
  offering: CatalogOffering
  input: number | null
  output: number | null
}

type OfferingSortColumn =
  | "provider"
  | "input"
  | "outputPrice"
  | "cacheRead"
  | "context"

function compareOfferings(
  column: OfferingSortColumn,
  direction: "asc" | "desc",
): (a: OfferingRow, b: OfferingRow) => number {
  const sign = direction === "asc" ? 1 : -1
  const byProvider = (a: OfferingRow, b: OfferingRow) =>
    a.offering.provider.localeCompare(b.offering.provider)
  const pick = (row: OfferingRow): number | string | null => {
    switch (column) {
      case "provider":
        return row.offering.provider
      case "input":
        return row.input
      case "outputPrice":
        return row.output
      case "cacheRead":
        return row.offering.pricing?.cache_read_price_per_million ?? null
      default:
        return row.offering.context_window ?? null
    }
  }
  return (a, b) => {
    const av = pick(a)
    const bv = pick(b)
    if (av == null && bv == null) return byProvider(a, b)
    if (av == null) return 1
    if (bv == null) return -1
    const order = av < bv ? -1 : av > bv ? 1 : 0
    return order * sign || byProvider(a, b)
  }
}

function offeringColumns({
  canPrice,
  canOverride,
  withUsage,
}: {
  canPrice: boolean
  canOverride: boolean
  withUsage: boolean
}): DataTableColumn<OfferingRow>[] {
  // Lane order is what a laptop sees without scrolling: the provider and its
  // selector, then the prices the page exists to compare, then where each came
  // from. Limits sit past the fold, since a model's limits are mostly the
  // model's and the header above already says them.
  const columns: DataTableColumn<OfferingRow>[] = [
    {
      id: "provider",
      header: "Provider",
      isRowHeader: true,
      allowsSorting: true,
      cell: ({ offering: row }) => (
        // One line: the selector, which is as long as the provider makes it,
        // opens under the row instead of setting every row's height.
        <span className="text-body whitespace-nowrap">
          {row.provider}
          <span className="text-caption">
            {" · "}
            {row.provider_type !== row.provider
              ? `${row.provider_type} · `
              : ""}
            {credentialLabel(row.credential)}
            {row.quantization ? ` · ${row.quantization}` : ""}
          </span>
        </span>
      ),
    },
    {
      id: "input",
      header: "Input / 1M",
      align: "end",
      allowsSorting: true,
      cell: (row) => (
        <RateCell
          metered={row.input}
          listed={row.offering.metadata_input_price_per_million}
        />
      ),
    },
    {
      id: "outputPrice",
      header: "Output / 1M",
      align: "end",
      allowsSorting: true,
      cell: (row) => (
        <RateCell
          metered={row.output}
          listed={row.offering.metadata_output_price_per_million}
        />
      ),
    },
    {
      id: "cacheRead",
      header: "Cache read / 1M",
      align: "end",
      allowsSorting: true,
      cell: ({ offering: row }) => (
        <span className="text-mono-caption">
          {rate(row.pricing?.cache_read_price_per_million)}
        </span>
      ),
    },
    {
      id: "source",
      header: "Price from",
      cell: ({ offering: row }) => <SourceMark source={row.price_source} />,
    },
    {
      id: "context",
      header: "Context / max out",
      align: "end",
      allowsSorting: true,
      cell: ({ offering: row }) => (
        <span className="text-mono-caption whitespace-nowrap">
          {formatContext(row.context_window)} /{" "}
          {formatContext(row.max_output_tokens)}
        </span>
      ),
    },
  ]
  if (withUsage) {
    // What the organization was charged for this offering, after cache reads
    // and tiers: the number that says whether the sticker price is the one
    // that matters.
    columns.push({
      id: "usage",
      header: "Yours, 30d",
      align: "end",
      cell: ({ offering: row }) =>
        row.usage_30d ? (
          <span className="flex flex-col items-end">
            <span className="text-mono-caption">
              {rate(row.usage_30d.effective_price_per_million)}
            </span>
            <span className="whitespace-nowrap text-caption text-subtle">
              {row.usage_30d.requests} req · cache{" "}
              {percent(row.usage_30d.cache_hit_rate)}
            </span>
          </span>
        ) : (
          <span className="text-mono-caption text-subtle">—</span>
        ),
    })
  }
  if (canPrice) {
    columns.push({
      id: "actions",
      header: "Actions",
      cell: ({ offering: row }) => (
        <Link
          to="/organization/pricing"
          search={{ model: row.selector }}
          className="text-link hover:text-link-hover"
        >
          Edit rate
        </Link>
      ),
    })
  } else if (canOverride) {
    // An organization admin cannot touch the deployment's price, but may set
    // what their own organization is billed above it.
    columns.push({
      id: "actions",
      header: "Actions",
      cell: ({ offering: row }) => (
        <Link
          to="/organization/pricing"
          search={{ override: row.selector }}
          className="text-link hover:text-link-hover"
        >
          Set your rate
        </Link>
      ),
    })
  }
  return columns
}

/**
 * What opens under an offering's row: the string to send and the request that
 * sends it. The table stays a comparison; the row a reader picks becomes the
 * integration.
 */
function OfferingDetail({
  offering,
  publicView,
  onClose,
}: {
  offering: CatalogOffering
  publicView: boolean
  onClose: () => void
}) {
  const deployment = useDeployment()
  const baseUrl = resolveSnippetBaseUrl(deployment)
  const [language, setLanguage] = useState("curl")
  const modelId = offering.selector.startsWith(`${offering.provider}:`)
    ? offering.selector.slice(offering.provider.length + 1)
    : offering.selector
  // The short spelling where the gateway has one: it is what the catalog
  // shows, and the full selector is still accepted.
  const sendAs = offering.short_selector ?? offering.selector
  const input = {
    baseUrl: baseUrl ?? "",
    apiKey: "$OTARI_API_KEY",
    model: sendAs,
  }
  return (
    <div className="flex flex-col gap-4 px-4 py-4">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <dl className="grid grid-cols-[auto_minmax(0,1fr)] gap-x-4 gap-y-1 text-sm">
          <dt className="text-caption">Selector</dt>
          <dd className="flex flex-wrap items-center gap-x-3 gap-y-1">
            <CopyableValue value={sendAs} label="selector">
              <code className="text-mono-caption break-all">{sendAs}</code>
            </CopyableValue>
            {offering.short_selector ? (
              <span className="text-caption">
                also{" "}
                <code className="text-mono-caption break-all">
                  {offering.selector}
                </code>
              </span>
            ) : null}
          </dd>
          <dt className="text-caption">Provider's id</dt>
          <dd>
            <code className="text-mono-caption break-all">{modelId}</code>
          </dd>
        </dl>
        <Button size="sm" variant="ghost" onPress={onClose}>
          Close
        </Button>
      </div>
      {baseUrl === undefined ? (
        <p className="text-sm text-muted">
          This deployment has not published its gateway address, so there is no
          request to copy yet.
        </p>
      ) : (
        <div className="flex max-w-3xl flex-col gap-2">
          <TabRow>
            <Tab
              isActive={language === "curl"}
              onPress={() => setLanguage("curl")}
            >
              cURL
            </Tab>
            <Tab
              isActive={language === "python"}
              onPress={() => setLanguage("python")}
            >
              Python (OpenAI SDK)
            </Tab>
          </TabRow>
          <CopyField
            label={language === "curl" ? "cURL" : "Python"}
            value={
              language === "curl"
                ? buildCurlSnippet(input)
                : buildPythonSnippet(input)
            }
            multiline
          />
          {publicView ? (
            <p className="text-caption">
              Sign in and create a key on API keys to send this.
            </p>
          ) : null}
        </div>
      )}
    </div>
  )
}

const SECTIONS = [
  { id: "providers", label: "Providers" },
  { id: "pricing", label: "Pricing" },
  { id: "use", label: "Quick start" },
]

function jumpTo(id: string) {
  document
    .getElementById(id)
    ?.scrollIntoView({ behavior: "smooth", block: "start" })
}

/** The request to copy, as a dialog: an API key first, then the call. */
function QuickStart({
  selector,
  isOpen,
  onOpenChange,
  publicView,
}: {
  selector: string
  isOpen: boolean
  onOpenChange: (open: boolean) => void
  publicView: boolean
}) {
  const deployment = useDeployment()
  const baseUrl = resolveSnippetBaseUrl(deployment)
  const [language, setLanguage] = useState("curl")
  const input = {
    baseUrl: baseUrl ?? "",
    apiKey: "$OTARI_API_KEY",
    model: selector,
  }
  return (
    <Modal isOpen={isOpen} onOpenChange={onOpenChange}>
      <Modal.Trigger className="hidden">Open quick start</Modal.Trigger>
      <Modal.Backdrop className="bg-backdrop/50">
        <Modal.Container placement="center" size="lg">
          <Modal.Dialog aria-label="Use this model" className="p-0">
            <div className="flex flex-col gap-5 p-6">
              <div className="flex flex-col gap-1">
                <h2 className="text-heading">Quick start</h2>
                <p className="text-sm text-muted">
                  Drop-in code to call this model through the gateway's
                  OpenAI-compatible API.
                </p>
              </div>
              <div className="flex flex-col gap-2">
                <h3 className="text-title">1. Get an API key</h3>
                <p className="text-sm text-muted">
                  {publicView ? (
                    <>
                      <a href="#/" className="text-link hover:text-link-hover">
                        Sign in
                      </a>{" "}
                      and create a key on API keys, then set it as an
                      environment variable.
                    </>
                  ) : (
                    <>
                      Create a key on{" "}
                      <Link
                        to="/keys"
                        className="text-link hover:text-link-hover"
                      >
                        API keys
                      </Link>{" "}
                      and set it as an environment variable.
                    </>
                  )}
                </p>
                <CopyField
                  label="Environment"
                  value="export OTARI_API_KEY=sk-…"
                />
              </div>
              <div className="flex flex-col gap-2">
                <h3 className="text-title">2. Make your first request</h3>
                <p className="text-sm text-muted">
                  Send <code className="text-mono-caption">{selector}</code> as
                  the model.
                </p>
                {baseUrl === undefined ? (
                  <p className="text-sm text-muted">
                    This deployment has not published its gateway address, so
                    there is no request to copy yet.
                  </p>
                ) : (
                  <>
                    <TabRow>
                      <Tab
                        isActive={language === "curl"}
                        onPress={() => setLanguage("curl")}
                      >
                        cURL
                      </Tab>
                      <Tab
                        isActive={language === "python"}
                        onPress={() => setLanguage("python")}
                      >
                        Python (OpenAI SDK)
                      </Tab>
                    </TabRow>
                    <CopyField
                      label={language === "curl" ? "cURL" : "Python"}
                      value={
                        language === "curl"
                          ? buildCurlSnippet(input)
                          : buildPythonSnippet(input)
                      }
                      multiline
                    />
                  </>
                )}
              </div>
              <div className="flex justify-end">
                <Button variant="ghost" onPress={() => onOpenChange(false)}>
                  Close
                </Button>
              </div>
            </div>
          </Modal.Dialog>
        </Modal.Container>
      </Modal.Backdrop>
    </Modal>
  )
}

export function ModelDetailView({
  modelId,
  publicView = false,
  initialQuickStart = false,
}: {
  modelId: string
  /** Ahead of a session: nothing to link through, and the rates are the deployment's list. */
  publicView?: boolean
  /** Open on the request to copy, as `?view=api` asks. */
  initialQuickStart?: boolean
}) {
  const organization = useOrganizationContext(!publicView)
  const canPrice = !publicView && isDeploymentOperator(organization.data)
  const canOverride = !publicView && !canPrice && canManage(organization.data)
  const catalog = useCatalog()
  const selected = useCatalogModel(modelId)
  const [quantization, setQuantization] = useState("all")
  const [quickStart, setQuickStart] = useState(initialQuickStart)
  const [opened, setOpened] = useState<string | null>(null)
  const [sort, setSort] = useState<{
    column: OfferingSortColumn
    direction: "asc" | "desc"
  }>({ column: "input", direction: "asc" })
  // Stable, as the DataTable asks: it depends on nothing that changes after
  // the first render, so the row cache holds for the life of the page.
  const renderDetail = useCallback(
    (row: OfferingRow) => (
      <OfferingDetail
        offering={row.offering}
        publicView={publicView}
        onClose={() => setOpened(null)}
      />
    ),
    [publicView],
  )

  if (selected.error) return <ErrorBanner error={selected.error} />
  const model = selected.data
  if (selected.isPending || !model) {
    return <PageLoading label="Loading model…" />
  }

  const quantizations = [
    ...new Set(
      model.offerings
        .map((o) => o.quantization)
        .filter((q): q is string => !!q),
    ),
  ]
  const rows: OfferingRow[] = model.offerings
    .filter(
      (offering) =>
        quantization === "all" || offering.quantization === quantization,
    )
    .map((offering) => ({
      offering,
      input: offering.pricing?.input_price_per_million ?? null,
      output: offering.pricing?.output_price_per_million ?? null,
    }))
    .sort(compareOfferings(sort.column, sort.direction))
  const withUsage = !publicView && hasUsage(model.offerings)
  const first = defaultOffering(model.offerings)
  const unpriced = model.offerings.filter((o) => o.pricing === null).length
  const defaultPricing = catalog.data?.default_pricing
  const capabilities = CAPABILITY_LABELS.filter(
    ({ key }) => model.capabilities[key],
  )
  const listPriceDiffers = model.offerings.some(
    (o) =>
      listPriceNote(
        o.pricing?.input_price_per_million,
        o.metadata_input_price_per_million,
      ) !== null ||
      listPriceNote(
        o.pricing?.output_price_per_million,
        o.metadata_output_price_per_million,
      ) !== null,
  )
  const used = model.offerings.filter((o) => o.usage_30d)
  const spend = used.reduce((sum, o) => sum + (o.usage_30d?.spend_usd ?? 0), 0)
  const requests = used.reduce(
    (sum, o) => sum + (o.usage_30d?.requests ?? 0),
    0,
  )
  const tokens = used.reduce(
    (sum, o) => sum + (o.usage_30d?.total_tokens ?? 0),
    0,
  )
  const modalities = (list: string[]) =>
    list.length === 0
      ? "—"
      : list.map((m) => MODALITY_LABELS[m] ?? m).join(", ")
  const title = model.vendor ? `${model.vendor}: ${model.name}` : model.name
  const sortDescriptor: SortDescriptor = {
    column: sort.column,
    direction: sort.direction === "asc" ? "ascending" : "descending",
  }

  return (
    <div className="flex flex-col gap-6">
      <nav aria-label="Breadcrumb" className="text-caption">
        {publicView ? (
          <a
            href={publicCatalogHref()}
            className="text-link hover:text-link-hover"
          >
            ← All models
          </a>
        ) : (
          <Link to="/models" className="text-link hover:text-link-hover">
            ← All models
          </Link>
        )}
      </nav>

      <header className="flex flex-col gap-3">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div className="flex min-w-0 flex-col gap-2">
            <h1 className="text-display break-words">{title}</h1>
            <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
              <CopyableValue value={model.id} label="model id">
                <code className="text-mono-caption">{model.id}</code>
              </CopyableValue>
              {model.selector ? (
                <span className="text-caption">
                  send it as <code className="text-mono-caption">model</code>
                  {model.resolves_to ? (
                    <>
                      {" "}
                      and{" "}
                      <code className="text-mono-caption">
                        {model.resolves_to}
                      </code>{" "}
                      answers
                    </>
                  ) : null}
                </span>
              ) : null}
              {model.open_weights ? (
                <Badge tone="muted">Open weights</Badge>
              ) : null}
              {model.deprecated ? <Badge tone="warn">Deprecated</Badge> : null}
            </div>
          </div>
          <div className="flex shrink-0 flex-wrap items-center gap-2">
            {canPrice ? (
              <Link
                to="/organization/pricing"
                className="inline-flex min-h-9 items-center text-sm text-link hover:text-link-hover"
              >
                Model pricing
              </Link>
            ) : null}
            {first ? (
              <Button variant="primary" onPress={() => setQuickStart(true)}>
                Use this model
              </Button>
            ) : null}
          </div>
        </div>
        {model.description ? (
          <p className="max-w-prose text-sm text-foreground">
            {model.description}
          </p>
        ) : null}
        <div className="grid grid-cols-2 gap-3 md:grid-cols-3 xl:grid-cols-6">
          <Stat
            label="Modalities"
            value={`${modalities(model.input_modalities)} → ${modalities(model.output_modalities)}`}
          />
          <Stat
            label="In / out price"
            value={
              model.min_input_price_per_million == null
                ? "unpriced"
                : `${formatRate(model.min_input_price_per_million)} / ${rate(model.min_output_price_per_million)} per 1M`
            }
          />
          <Stat
            label="Context"
            value={
              model.context_window == null
                ? "—"
                : `up to ${formatContext(model.context_window)}`
            }
          />
          <Stat
            label="Max output"
            value={
              model.max_output_tokens == null
                ? "—"
                : `up to ${formatContext(model.max_output_tokens)}`
            }
          />
          <Stat
            label="Released"
            value={formatReleaseDate(model.release_date)}
          />
          <Stat label="Knowledge" value={model.knowledge_cutoff ?? "—"} />
        </div>
        <div className="flex flex-wrap gap-x-4 gap-y-1.5 text-mono-overline text-muted">
          {capabilities.map(({ key, label }) => (
            <span key={key}>{label}</span>
          ))}
          {capabilities.length === 0 ? (
            <span className="normal-case tracking-normal text-subtle">
              No capability metadata for this model.
            </span>
          ) : null}
        </div>
      </header>

      <div className="border-t border-border" />

      <div className="flex flex-col gap-6 lg:grid lg:grid-cols-[11rem_minmax(0,1fr)] lg:gap-10">
        <nav aria-label="Sections" className="lg:sticky lg:top-4 lg:self-start">
          <div className="lg:hidden">
            <TabRow>
              {SECTIONS.map((section) => (
                <Tab
                  key={section.id}
                  isActive={false}
                  onPress={() => jumpTo(section.id)}
                >
                  {section.label}
                </Tab>
              ))}
            </TabRow>
          </div>
          <ul className="hidden flex-col gap-1 lg:flex">
            {SECTIONS.map((section) => (
              <li key={section.id}>
                <button
                  type="button"
                  onClick={() => jumpTo(section.id)}
                  className="flex min-h-9 w-full items-center px-2 text-left text-sm text-muted hover:bg-surface-subtle hover:text-foreground"
                >
                  {section.label}
                </button>
              </li>
            ))}
          </ul>
        </nav>

        <div className="flex min-w-0 flex-col gap-10">
          <section
            id="providers"
            aria-labelledby="providers-title"
            className="flex scroll-mt-4 flex-col gap-3"
          >
            <div className="flex flex-col gap-1">
              <h2 id="providers-title" className="text-heading">
                Providers
              </h2>
              <p className="max-w-prose text-sm text-muted">
                Several providers serve the same model. Each row is one
                offering: what{" "}
                {publicView ? "this deployment lists it at" : "you pay"} and
                where that price comes from; open a row for the selector to send
                and the request that sends it. {model.offering_count} on{" "}
                {model.provider_count}{" "}
                {model.provider_count === 1 ? "provider" : "providers"},
                cheapest first.
              </p>
            </div>
            <div className="flex flex-col gap-3 border border-border bg-surface p-3">
              {quantizations.length > 0 ? (
                <div className="otari-toolbar flex flex-wrap items-center gap-2">
                  <FilterSelect
                    ariaLabel="Filter quantization"
                    value={quantization}
                    onChange={setQuantization}
                    options={[
                      { value: "all", label: "Any quantization" },
                      ...quantizations.map((q) => ({ value: q, label: q })),
                    ]}
                  />
                </div>
              ) : null}
              <TableScrollFrame className="otari-offerings-table">
                <DataTable
                  ariaLabel={`Offerings of ${model.name}`}
                  columns={offeringColumns({
                    canPrice,
                    canOverride,
                    withUsage,
                  })}
                  rows={rows}
                  getRowKey={(row) => row.offering.selector}
                  onRowAction={(key) =>
                    setOpened((current) => (current === key ? null : key))
                  }
                  detailKey={opened}
                  renderDetail={renderDetail}
                  sortDescriptor={sortDescriptor}
                  onSortChange={(descriptor) =>
                    setSort({
                      column: String(descriptor.column) as OfferingSortColumn,
                      direction:
                        descriptor.direction === "ascending" ? "asc" : "desc",
                    })
                  }
                  emptyContent="No provider you can use serves this model."
                />
              </TableScrollFrame>
            </div>
            {unpriced > 0 && defaultPricing === false ? (
              <p className="text-caption">
                Default pricing is off, so an offering with no stored rate is
                unpriced here even where genai-prices publishes one.
                {canPrice ? " Both switches live on Settings." : ""}
              </p>
            ) : null}
            {listPriceDiffers ? (
              <InfoBanner>
                A rate marked <em>list</em> differs from the price the provider
                publishes on models.dev. What is metered here is the rate shown;
                the list price is what the provider would charge you directly.
              </InfoBanner>
            ) : null}
            {model.also_available_from.length > 0 ? (
              <p className="text-caption">
                Also served by {elsewhere(model.also_available_from)}, which{" "}
                {model.also_available_from.length === 1 ? "is" : "are"} not
                configured here.
                {canPrice ? (
                  <>
                    {" "}
                    <Link
                      to="/providers"
                      className="text-link hover:text-link-hover"
                    >
                      Add a provider
                    </Link>
                    .
                  </>
                ) : null}
              </p>
            ) : null}
          </section>

          <section
            id="pricing"
            aria-labelledby="pricing-title"
            className="flex scroll-mt-4 flex-col gap-3"
          >
            <div className="flex flex-col gap-1">
              <h2 id="pricing-title" className="text-heading">
                Pricing
              </h2>
              <p className="max-w-prose text-sm text-muted">
                {publicView
                  ? "The cheapest listed rates for this model on this deployment."
                  : "What your organization actually paid for this model over the last 30 days, beside the rates providers list. Cache reads and tiers mean the price paid is often below the listed one."}
              </p>
            </div>
            <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
              <Stat
                label="Cheapest input"
                value={
                  model.min_input_price_per_million == null
                    ? "—"
                    : `${formatRate(model.min_input_price_per_million)} /M`
                }
              />
              <Stat
                label="Cheapest output"
                value={
                  model.min_output_price_per_million == null
                    ? "—"
                    : `${formatRate(model.min_output_price_per_million)} /M`
                }
              />
              {!publicView ? (
                <>
                  <Stat
                    label="Your effective price, 30d"
                    value={
                      tokens > 0
                        ? `${formatRate((spend / tokens) * 1_000_000)} /M`
                        : "no usage"
                    }
                  />
                  <Stat
                    label="Your spend, 30d"
                    value={
                      requests > 0
                        ? `${formatRate(spend)} · ${requests} req`
                        : "no usage"
                    }
                  />
                </>
              ) : null}
            </div>
          </section>

          <section
            id="use"
            aria-labelledby="use-title"
            className="flex scroll-mt-4 flex-col gap-3"
          >
            <div className="flex flex-wrap items-baseline justify-between gap-3">
              <h2 id="use-title" className="text-heading">
                Quick start
              </h2>
              {first ? (
                <span className="text-caption">
                  Send{" "}
                  <code className="text-mono-caption">
                    {model.selector ?? first.short_selector ?? first.selector}
                  </code>
                  {model.selector ? " for the cheapest offering" : ""}
                  {model.offerings.length > 1 && !publicView ? (
                    <>
                      , or{" "}
                      <Link
                        to="/routing"
                        className="text-link hover:text-link-hover"
                      >
                        route across the offerings
                      </Link>
                    </>
                  ) : null}
                </span>
              ) : null}
            </div>
            {first ? (
              <p className="max-w-prose text-sm text-muted">
                Open an offering above for its selector and a request that sends
                it. "Use this model" at the top of the page does the same for
                the cheapest one, with the API key step first.
              </p>
            ) : (
              <p className="text-sm text-muted">
                No provider you can use serves this model.
              </p>
            )}
          </section>
        </div>
      </div>

      {first ? (
        <QuickStart
          selector={model.selector ?? first.short_selector ?? first.selector}
          isOpen={quickStart}
          onOpenChange={setQuickStart}
          publicView={publicView}
        />
      ) : null}
    </div>
  )
}

/** The model page on the router. */
export function ModelDetailPage({ modelId }: { modelId: string }) {
  const view = useUrlValue("view")
  return (
    <ModelDetailView modelId={modelId} initialQuickStart={view === "api"} />
  )
}
