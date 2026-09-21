/**
 * The genai-prices catalog, and the operator's review of an update to it.
 *
 * Check for updates fetches upstream and holds the result; the dialog is where
 * an operator accepts or rejects it. A scheduled check can leave one waiting,
 * which is what the pending notice reads.
 *
 * Operator-only: all four reads and writes are `require_deployment_operator`, so
 * the page that mounts this withholds the whole band from anyone else rather
 * than rendering four refusal banners. `PricingRefreshDialog` stays here rather
 * than in a file of its own because the two are useless apart.
 */

import { Button } from "@heroui/react"
import { useState } from "react"

import type { PricingRefreshPreview } from "@/client"
import { Dialog, DialogSection } from "@/design-system/feedback/Dialog"
import { ErrorBanner } from "@/design-system/feedback/ErrorBanner"
import { InfoBanner } from "@/design-system/feedback/InfoBanner"
import { Section } from "@/design-system/layout/Section"
import {
  useConfirmPricingRefresh,
  usePendingPricingRefresh,
  usePreviewPricingRefresh,
  usePricingSnapshots,
  useRejectPricingRefresh,
} from "@/shared/api/pricing"
import { formatRelative } from "@/shared/helpers/format"

function PricingRefreshDialog({
  preview,
  error,
  isPending,
  onAccept,
  onReject,
  onDismiss,
}: {
  preview: PricingRefreshPreview
  error: Error | null
  isPending: boolean
  onAccept: () => void
  onReject: () => void
  onDismiss: () => void
}) {
  return (
    <Dialog
      isOpen
      // Dismissing is "not now", not "reject". The preview is a row the gateway
      // is holding for review, so closing the frame leaves it pending and the
      // notice above offers it again; rejecting discards it server-side and is
      // what the footer's own control is for. Refused outright while a mutation
      // is in flight, which is the same answer the two buttons give.
      onOpenChange={(isOpen) => (isOpen ? undefined : onDismiss())}
      isDismissable={!isPending}
      title="Review default price updates"
      size="lg"
      actions={
        <>
          <Button variant="ghost" isDisabled={isPending} onPress={onReject}>
            Reject changes
          </Button>
          <Button variant="primary" isPending={isPending} onPress={onAccept}>
            Accept price updates
          </Button>
        </>
      }
    >
      <DialogSection>
        <p className="text-sm text-muted">
          {preview.added_count} added, {preview.changed_count} changed, and{" "}
          {preview.removed_count} removed upstream model prices. The accepted
          catalog is saved in the database with source <code>genai-prices</code>{" "}
          and reloads after a restart. Your {preview.protected_model_count}{" "}
          custom model price
          {preview.protected_model_count === 1 ? "" : "s"} remain unchanged.
        </p>
        {preview.changes.length > 0 ? (
          <ul className="max-h-60 list-disc overflow-auto pl-5 text-body">
            {preview.changes.map((change) => (
              <li key={change.model_key}>
                {change.model_key}: {change.change}
              </li>
            ))}
          </ul>
        ) : null}
        {preview.changes_truncated ? (
          <p className="text-caption">Only the first 100 changes are shown.</p>
        ) : null}
        <ErrorBanner error={error} />
      </DialogSection>
    </Dialog>
  )
}

// Who accepted a snapshot, as a word. The schedule is the only non-person.
function acceptedBy(who: string): string {
  return who === "schedule" ? "the scheduled check" : `an ${who}`
}

export function PricingRefreshSection() {
  const previewRefresh = usePreviewPricingRefresh()
  const confirmRefresh = useConfirmPricingRefresh()
  const rejectRefresh = useRejectPricingRefresh()
  // What the scheduled check left for review under `pricing_refresh: review`.
  // It is the same pending row a manual check writes, so the one dialog and
  // the same confirm and reject serve both; the only difference is who fetched.
  const pending = usePendingPricingRefresh()
  const snapshots = usePricingSnapshots()
  const [reviewingPending, setReviewingPending] = useState(false)
  const preview =
    previewRefresh.data ??
    (reviewingPending ? pending.data : undefined) ??
    undefined
  const isPending = confirmRefresh.isPending || rejectRefresh.isPending
  const latest = snapshots.data?.[0]

  const close = () => {
    previewRefresh.reset()
    // The two mutations' errors are the frame's, not the page's, so they go
    // with it: dismissing after a failed accept and reopening would otherwise
    // greet the operator with the banner from the attempt before.
    confirmRefresh.reset()
    rejectRefresh.reset()
    setReviewingPending(false)
  }
  const reject = () => {
    if (preview === undefined || isPending) {
      return
    }
    rejectRefresh.mutate(undefined, { onSuccess: close })
  }

  return (
    <>
      <Section
        aria-labelledby="pricing-catalog-title"
        className="border-y border-border py-5"
        contentClassName="flex flex-col gap-4"
      >
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div className="min-w-0">
            <h2 id="pricing-catalog-title" className="text-title">
              Default pricing catalog
            </h2>
            <p className="mt-1 max-w-3xl text-sm text-muted">
              Fetch the latest upstream catalog, review the proposed change
              summary, then accept or reject it. Accepted data is stored as{" "}
              <code>genai-prices</code>; custom prices remain separate and
              always take precedence.
            </p>
            {latest ? (
              <p className="mt-1 text-caption">
                Last accepted {formatRelative(latest.accepted_at)} by{" "}
                {acceptedBy(latest.accepted_by)}, {latest.model_count} priced
                models.
                {snapshots.data && snapshots.data.length > 1
                  ? ` ${snapshots.data.length} snapshots on record.`
                  : ""}
              </p>
            ) : null}
          </div>
          <Button
            size="sm"
            variant="ghost"
            isDisabled={previewRefresh.isPending || isPending}
            onPress={() => previewRefresh.mutate()}
          >
            {previewRefresh.isPending
              ? "Checking prices…"
              : "Check for price updates"}
          </Button>
        </div>
        <ErrorBanner error={previewRefresh.error} />
        {pending.data ? (
          <div className="flex flex-wrap items-center justify-between gap-3">
            <InfoBanner tone="warning">
              The scheduled check found {pending.data.changed_count} changed,{" "}
              {pending.data.added_count} added and {pending.data.removed_count}{" "}
              removed default prices, fetched{" "}
              {formatRelative(pending.data.fetched_at)}. Nothing changes until
              you accept.
            </InfoBanner>
            <Button
              size="sm"
              variant="primary"
              isDisabled={isPending}
              onPress={() => setReviewingPending(true)}
            >
              Review pending update
            </Button>
          </div>
        ) : null}
      </Section>
      {preview ? (
        <PricingRefreshDialog
          preview={preview}
          error={confirmRefresh.error ?? rejectRefresh.error}
          isPending={isPending}
          onAccept={() =>
            confirmRefresh.mutate(undefined, {
              onSuccess: close,
            })
          }
          onReject={reject}
          onDismiss={close}
        />
      ) : null}
    </>
  )
}

/**
 * Whether an unpriced model is metered at all, which is the catalog's first
 * question and the one that decides what the table below is for.
 *
 * With default pricing on, the table is the exceptions: models whose stored rate
 * overrides an upstream default. With it off, the table is the whole of what can
 * be billed, and `require_pricing` decides whether anything else is refused
 * outright or served for free.
 *
 * Mounted only for a deployment operator, which is why it reads `useSettings()`
 * with no gate of its own: `GET /v1/settings` is operator-only, so an admin who
 * saw this banner would be reading a refusal. Unmounted rather than passed a
 * disabled query, so nothing is left holding a cached answer from a session that
 * used to be an operator's.
 */
