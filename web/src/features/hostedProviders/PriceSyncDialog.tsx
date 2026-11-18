import { useState } from "react"

import type { HostedPriceResetChange, HostedProvider } from "@/client"
import { ErrorBanner } from "@/design-system/feedback/ErrorBanner"
import { FormDialog } from "@/design-system/feedback/FormDialog"
import { Checkbox } from "@/design-system/forms/Checkbox"
import { formatOfferingPrice } from "@/features/models/pricingUnit"
import {
  useHostedPriceResetPreview,
  useResetHostedModelPrices,
} from "@/shared/api/hostedProviders"

/**
 * Put prices an operator set back on the default rate, after seeing what moves.
 *
 * The preview is a dry run of the reset, asked once per opening. A model
 * whose custom price differs from the default starts ticked; one whose price
 * already matches starts unticked, since resetting it changes nothing today and
 * only lets later default changes apply. Applying resets the ticked models.
 */
export function PriceSyncDialog({
  provider,
  opening,
  vendor,
  isOpen,
  onClose,
  onApplied,
}: {
  provider: HostedProvider
  /** Which opening this is; 0 while the dialog is closed. */
  opening: number
  /** One vendor's models; empty for every vendor. */
  vendor: string
  isOpen: boolean
  onClose: () => void
  /** How many models the server reset, and the ticked ones it kept, having no default now. */
  onApplied: (count: number, kept: string[]) => void
}) {
  const preview = useHostedPriceResetPreview(provider.name, opening, vendor)
  const reset = useResetHostedModelPrices(provider.name)
  // The models whose tick the operator flipped from where it started. Held per
  // opening, since the dialog is keyed on it by nothing else.
  const [flipped, setFlipped] = useState<{
    opening: number
    models: ReadonlySet<string>
  }>({ opening, models: new Set() })
  const flips = flipped.opening === opening ? flipped.models : new Set<string>()

  const changes = preview.data?.changes ?? []
  const isPicked = (change: HostedPriceResetChange) =>
    !change.matches !== flips.has(change.model)
  const picked = changes.filter(isPicked).map((change) => change.model)
  const differs = changes.filter((change) => !change.matches)
  const matches = changes.filter((change) => change.matches)
  const skipped = preview.data?.skipped_no_default.length ?? 0
  const defaultName = provider.provider === "otari" ? "Default" : "models.dev"

  const flip = (model: string) => {
    const next = new Set(flips)
    if (!next.delete(model)) next.add(model)
    setFlipped({ opening, models: next })
  }

  const list = (rows: HostedPriceResetChange[]) => (
    <ul className="flex flex-col divide-y divide-border-subtle border border-border">
      {rows.map((change) => (
        <li key={change.model} className="px-3 py-1.5">
          <Checkbox
            hasTouchTarget
            className="justify-start"
            isSelected={isPicked(change)}
            onChange={() => flip(change.model)}
          >
            <span className="flex min-w-0 flex-col">
              <span className="text-mono-caption break-all">
                {change.model}
              </span>
              <span className="text-caption tabular-nums">
                Custom {formatOfferingPrice(change.current)} · {defaultName}{" "}
                {formatOfferingPrice(change.default)}
              </span>
            </span>
          </Checkbox>
        </li>
      ))}
    </ul>
  )

  const body = () => {
    if (preview.error) return <ErrorBanner error={preview.error} />
    if (preview.isPending || preview.isFetching) {
      return <p className="text-caption">Comparing custom prices…</p>
    }
    return (
      <>
        {differs.length > 0 ? (
          <section
            aria-label="Differs from the default"
            className="flex flex-col gap-2"
          >
            <h3 className="text-overline">
              Differs from the default · {differs.length}
            </h3>
            {list(differs)}
          </section>
        ) : (
          <p className="text-caption">
            No custom price here differs from the default.
          </p>
        )}
        {matches.length > 0 ? (
          <section
            aria-label="Matches the default"
            className="flex flex-col gap-2"
          >
            <h3 className="text-overline">
              Matches the default · {matches.length}
            </h3>
            <p className="text-caption">
              Resetting these changes nothing now, but later changes to the
              default would then apply.
            </p>
            {list(matches)}
          </section>
        ) : null}
        {skipped > 0 ? (
          <p className="text-caption">
            {skipped} custom {skipped === 1 ? "price has" : "prices have"} no
            default price and {skipped === 1 ? "is" : "are"} left alone.
          </p>
        ) : null}
      </>
    )
  }

  return (
    <FormDialog
      isOpen={isOpen}
      onOpenChange={(open) => {
        if (!open) onClose()
      }}
      title={vendor ? `Sync ${vendor} prices` : "Sync prices"}
      description={`Put custom prices back on the ${defaultName} price, which every refresh then keeps current. Requests already billed keep their cost.`}
      size="lg"
      submitLabel={`Use default price for ${picked.length}`}
      isSubmitDisabled={
        picked.length === 0 || !preview.isSuccess || preview.isFetching
      }
      isPending={reset.isPending}
      error={reset.error}
      onSubmit={() =>
        reset.mutate(picked, {
          // The server's own count: a default can go between preview and apply.
          onSuccess: (result) =>
            onApplied(
              result.applied ? result.changes.length : 0,
              result.skipped_no_default.filter((model) =>
                picked.includes(model),
              ),
            ),
        })
      }
    >
      {body()}
    </FormDialog>
  )
}
