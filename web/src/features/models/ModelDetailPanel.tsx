import { Link } from "@tanstack/react-router"
import type { ReactNode } from "react"

import type { CatalogModelDetail, CatalogOffering } from "@/client"
import {
  CAPABILITY_LABELS,
  credentialLabel,
  defaultOffering,
  MODALITY_LABELS,
  priceSourceLabel,
} from "@/features/models/catalog"
import { CopyableValue, CopyField } from "@/shared/components/actions/CopyField"
import {
  DataTable,
  type DataTableColumn,
} from "@/shared/components/data/DataTable"
import { ErrorBanner } from "@/shared/components/feedback/ErrorBanner"
import { PageLoading } from "@/shared/components/feedback/PageLoading"
import { Dot } from "@/shared/components/indicators/Dot"
import { TableScrollFrame } from "@/shared/components/layout/TableScrollFrame"
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
import { useDeployment } from "@/shared/hooks/useDeployment"

// The right-hand column of the catalog: one model, and every offering of it
// this viewer may call. Read-only for everyone (otari-ai#2095, #2096): a rate
// is edited on Model pricing, which the operator's link here points at, so the
// page that compares prices never becomes the page that changes them.

function Spec({ label, value }: { label: string; value: ReactNode }) {
  return (
    <div className="flex flex-col gap-0.5">
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

function offeringColumns(
  canPrice: boolean,
): DataTableColumn<CatalogOffering>[] {
  const columns: DataTableColumn<CatalogOffering>[] = [
    {
      id: "provider",
      header: "Provider",
      isRowHeader: true,
      cell: (row) => (
        <div className="flex flex-col">
          <span className="text-body">{row.provider}</span>
          <span className="text-caption">
            {row.provider_type !== row.provider
              ? `${row.provider_type} · `
              : ""}
            {credentialLabel(row.credential)}
            {row.quantization ? ` · ${row.quantization}` : ""}
          </span>
        </div>
      ),
    },
    {
      id: "selector",
      header: "Selector",
      cell: (row) => (
        <CopyableValue value={row.selector} label="selector">
          <code className="text-mono-caption">{row.selector}</code>
        </CopyableValue>
      ),
    },
    {
      id: "context",
      header: "Context",
      align: "end",
      cell: (row) => (
        <span className="text-mono-caption">
          {formatContext(row.context_window)}
        </span>
      ),
    },
    {
      id: "output",
      header: "Max out",
      align: "end",
      cell: (row) => (
        <span className="text-mono-caption">
          {formatContext(row.max_output_tokens)}
        </span>
      ),
    },
    {
      id: "input",
      header: "Input / 1M",
      align: "end",
      cell: (row) => (
        <span className="text-mono-caption">
          {rate(row.pricing?.input_price_per_million)}
        </span>
      ),
    },
    {
      id: "outputPrice",
      header: "Output / 1M",
      align: "end",
      cell: (row) => (
        <span className="text-mono-caption">
          {rate(row.pricing?.output_price_per_million)}
        </span>
      ),
    },
    {
      id: "cacheRead",
      header: "Cache read / 1M",
      align: "end",
      cell: (row) => (
        <span className="text-mono-caption">
          {rate(row.pricing?.cache_read_price_per_million)}
        </span>
      ),
    },
    {
      id: "source",
      header: "Price from",
      cell: (row) => <SourceMark source={row.price_source} />,
    },
  ]
  if (canPrice) {
    columns.push({
      id: "actions",
      header: "Actions",
      cell: (row) => (
        <Link
          to="/organization/pricing"
          search={{ model: row.selector }}
          className="text-link hover:text-link-hover"
        >
          Edit rate
        </Link>
      ),
    })
  }
  return columns
}

function Snippets({ selector }: { selector: string }) {
  const deployment = useDeployment()
  const baseUrl = resolveSnippetBaseUrl(deployment)
  if (baseUrl === undefined) {
    // A hosted control plane that has not published its data-plane address:
    // a runnable command aimed at this host would be aimed at the wrong one.
    return (
      <p className="text-sm text-muted">
        This deployment has not published its gateway address, so there is no
        request to copy yet.
      </p>
    )
  }
  const input = { baseUrl, apiKey: "$OTARI_API_KEY", model: selector }
  return (
    <div className="flex flex-col gap-3">
      <CopyField label="cURL" value={buildCurlSnippet(input)} multiline />
      <CopyField
        label="Python (OpenAI SDK)"
        value={buildPythonSnippet(input)}
        multiline
      />
    </div>
  )
}

export function ModelDetailPanel({
  model,
  isLoading,
  error,
  canPrice,
  defaultPricing,
}: {
  model: CatalogModelDetail | undefined
  isLoading: boolean
  error: unknown
  /** Whether this caller may edit a deployment rate; adds the link that leaves the page. */
  canPrice: boolean
  /** Whether an unpriced offering is metered at the genai-prices default. */
  defaultPricing: boolean | undefined
}) {
  if (error) return <ErrorBanner error={error} />
  if (isLoading || !model) return <PageLoading label="Loading model…" />

  const capabilities = CAPABILITY_LABELS.filter(
    ({ key }) => model.capabilities[key],
  )
  const first = defaultOffering(model.offerings)
  const unpriced = model.offerings.filter((o) => o.pricing === null).length

  return (
    <div className="flex flex-col gap-6">
      <div className="flex flex-col gap-2">
        <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1">
          <h2 className="text-heading break-words">{model.name}</h2>
          {model.vendor ? (
            <span className="text-caption">by {model.vendor}</span>
          ) : null}
          {model.deprecated ? (
            <span className="flex items-center gap-2 text-mono-caption text-danger">
              <Dot className="bg-danger" />
              DEPRECATED
            </span>
          ) : null}
        </div>
        {model.description ? (
          <p className="max-w-prose text-sm text-foreground">
            {model.description}
          </p>
        ) : null}
      </div>

      <div className="grid grid-cols-2 gap-x-6 gap-y-3 sm:grid-cols-3 lg:grid-cols-5">
        <Spec
          label="Context"
          value={
            model.context_window == null
              ? "—"
              : `up to ${formatContext(model.context_window)}`
          }
        />
        <Spec
          label="Max output"
          value={
            model.max_output_tokens == null
              ? "—"
              : `up to ${formatContext(model.max_output_tokens)}`
          }
        />
        <Spec label="Released" value={formatReleaseDate(model.release_date)} />
        <Spec label="Knowledge" value={model.knowledge_cutoff ?? "—"} />
        <Spec label="Open weights" value={model.open_weights ? "Yes" : "No"} />
      </div>

      <div className="flex flex-wrap gap-x-4 gap-y-1.5 text-mono-overline text-muted">
        {capabilities.map(({ key, label }) => (
          <span key={key}>{label}</span>
        ))}
        {model.input_modalities.length > 0 ? (
          <span>
            <span className="text-subtle">In </span>
            {model.input_modalities
              .map((m) => MODALITY_LABELS[m] ?? m)
              .join(" · ")}
          </span>
        ) : null}
        {model.output_modalities.length > 0 ? (
          <span>
            <span className="text-subtle">Out </span>
            {model.output_modalities
              .map((m) => MODALITY_LABELS[m] ?? m)
              .join(" · ")}
          </span>
        ) : null}
        {capabilities.length === 0 && model.input_modalities.length === 0 ? (
          <span className="normal-case tracking-normal text-subtle">
            No metadata for this model.
          </span>
        ) : null}
      </div>

      <section
        aria-labelledby="offerings-title"
        className="flex flex-col gap-3"
      >
        <div className="flex items-baseline justify-between gap-3">
          <h3 id="offerings-title" className="text-title">
            Offerings
          </h3>
          <span className="text-caption">
            {model.offering_count} on {model.provider_count}{" "}
            {model.provider_count === 1 ? "provider" : "providers"}, cheapest
            first
          </span>
        </div>
        <TableScrollFrame className="otari-offerings-table">
          <DataTable
            ariaLabel={`Offerings of ${model.name}`}
            columns={offeringColumns(canPrice)}
            rows={model.offerings}
            getRowKey={(row) => row.selector}
            emptyContent="No provider you can use serves this model."
          />
        </TableScrollFrame>
        {unpriced > 0 && defaultPricing === false ? (
          <p className="text-caption">
            Default pricing is off, so an offering with no stored rate is
            unpriced here even where genai-prices publishes one.
            {canPrice ? " Both switches live on Settings." : ""}
          </p>
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

      {first ? (
        <section aria-labelledby="use-title" className="flex flex-col gap-3">
          <div className="flex items-baseline justify-between gap-3">
            <h3 id="use-title" className="text-title">
              Use this model
            </h3>
            <span className="text-caption">
              Through{" "}
              <code className="text-mono-caption">{first.selector}</code>
              {model.offerings.length > 1 ? (
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
          </div>
          <Snippets selector={first.selector} />
        </section>
      ) : null}
    </div>
  )
}
