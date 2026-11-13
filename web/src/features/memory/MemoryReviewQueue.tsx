import { useMemo, useState } from "react"

import type {
  HeldMemoryProposal,
  MemoryReviewReason,
  OrganizationMember,
} from "@/client"
import { Button } from "@/design-system/actions/Button"
import { ConfirmDialog } from "@/design-system/feedback/ConfirmDialog"
import { EmptyMessage } from "@/design-system/feedback/EmptyMessage"
import { ErrorBanner } from "@/design-system/feedback/ErrorBanner"
import { Badge } from "@/design-system/indicators/Badge"
import { SettingsGroup } from "@/design-system/layout/SettingsGroup"
import { memberLabel } from "@/features/organization/roles"
import {
  useMemoryReviewQueue,
  useSettleMemoryProposal,
} from "@/shared/api/memory"
import { useOrganizationMembers } from "@/shared/api/organizations"
import { formatDateTimeMinutes } from "@/shared/helpers/format"

const REASONS: Record<MemoryReviewReason, string> = {
  policy: "Held by the review policy",
  conflict: "Contradicts an entry",
  higher_trust: "Changes an admin-reviewed entry",
  guardrail: "Flagged by a mandated guardrail",
  guardrail_unavailable: "Guardrail could not run",
  model: "The consolidation model failed",
}

// What the entry beside a proposal is to it, by why it was held.
function entryHeading(reason: HeldMemoryProposal["reason"]): string {
  return reason === "conflict"
    ? "The entry it contradicts"
    : "The entry it changes"
}

function author(
  proposal: HeldMemoryProposal,
  labels: ReadonlyMap<string, string>,
): string {
  if (!proposal.author_user_id) return "an agent"
  return (
    labels.get(proposal.author_user_id) ??
    `member ${proposal.author_user_id.slice(0, 8)}`
  )
}

function labelsByUserId(
  members: OrganizationMember[] | undefined,
): ReadonlyMap<string, string> {
  return new Map(
    (members ?? []).flatMap((member): [string, string][] =>
      member.user_id ? [[member.user_id, memberLabel(member)]] : [],
    ),
  )
}

function HeldProposal({
  proposal,
  proposedBy,
  canManage,
  isBusy,
  onApprove,
  onDiscard,
}: {
  proposal: HeldMemoryProposal
  proposedBy: string
  canManage: boolean
  isBusy: boolean
  onApprove: () => void
  onDiscard: () => void
}) {
  const entry = proposal.entry
  return (
    <article
      aria-label={`Proposal from ${proposedBy}`}
      className="flex flex-col gap-3 py-3 px-4"
    >
      <div className="flex flex-wrap items-center gap-x-4 gap-y-1">
        <Badge tone="warn">
          {proposal.reason ? REASONS[proposal.reason] : "Held for review"}
        </Badge>
        <span className="text-caption text-subtle">
          Proposed by {proposedBy} on{" "}
          {formatDateTimeMinutes(proposal.created_at)}; deleted{" "}
          {formatDateTimeMinutes(proposal.expires_at)} unless decided
        </span>
      </div>
      <div
        className={
          entry ? "grid grid-cols-1 gap-3 md:grid-cols-2" : "flex flex-col"
        }
      >
        <div className="flex flex-col gap-1">
          <p className="text-caption text-subtle">Proposed</p>
          <p className="whitespace-pre-wrap break-words">{proposal.text}</p>
        </div>
        {entry ? (
          <div className="flex flex-col gap-1">
            <p className="text-caption text-subtle">
              {entryHeading(proposal.reason)}, version {entry.version}
            </p>
            <p className="whitespace-pre-wrap break-words text-muted">
              {entry.text}
            </p>
            <p className="text-caption text-subtle">
              Approving replaces it with the proposal, as its next version.
            </p>
          </div>
        ) : null}
      </div>
      <div className="flex flex-wrap items-center gap-2">
        <Button
          variant="primary"
          className="min-h-11 md:min-h-9"
          onPress={onApprove}
          isDisabled={!canManage || isBusy}
        >
          Approve
        </Button>
        <Button
          variant="ghost"
          className="min-h-11 md:min-h-9"
          onPress={onDiscard}
          isDisabled={!canManage || isBusy}
        >
          Discard
        </Button>
      </div>
    </article>
  )
}

/**
 * The shared proposals the consolidator held for an admin, oldest first.
 *
 * Approving vouches for one: the consolidator's next pass keeps it as reviewed,
 * without holding it again for the review policy or a guardrail, and one held
 * over an entry replaces that entry. Discarding deletes it, so it asks first. A
 * member's own proposals never wait here: no admin reads a member's memory.
 */
export function MemoryReviewQueue({
  workspaceId,
  canManage,
}: {
  workspaceId: string
  canManage: boolean
}) {
  const queue = useMemoryReviewQueue(workspaceId)
  const members = useOrganizationMembers()
  const settle = useSettleMemoryProposal()
  const [discarding, setDiscarding] = useState<HeldMemoryProposal>()
  const labels = useMemo(() => labelsByUserId(members.data), [members.data])
  const held = queue.data?.data ?? []

  const closeDiscard = () => {
    setDiscarding(undefined)
    settle.reset()
  }
  const discard = async () => {
    if (!discarding) return
    try {
      await settle.mutateAsync({
        workspaceId,
        proposalId: discarding.id,
        action: "discard",
      })
      closeDiscard()
    } catch {
      // Shown by the dialog.
    }
  }

  return (
    <>
      <SettingsGroup
        title="Waiting for review"
        count={queue.data ? queue.data.count : undefined}
        description="Shared proposals the consolidator held for an admin. Approving one keeps it as reviewed on the consolidator's next pass; one nobody decides on is deleted when it expires. Members' own memory never waits here."
      >
        {queue.data && held.length === 0 ? (
          <EmptyMessage>Nothing is waiting for review.</EmptyMessage>
        ) : null}
        {held.map((proposal) => (
          <HeldProposal
            key={proposal.id}
            proposal={proposal}
            proposedBy={author(proposal, labels)}
            canManage={canManage}
            isBusy={settle.isPending}
            onApprove={() =>
              settle.mutate({
                workspaceId,
                proposalId: proposal.id,
                action: "approve",
              })
            }
            onDiscard={() => setDiscarding(proposal)}
          />
        ))}
      </SettingsGroup>
      <ErrorBanner
        error={discarding === undefined ? (queue.error ?? settle.error) : null}
      />
      <ConfirmDialog
        isOpen={discarding !== undefined}
        onOpenChange={(open) => {
          if (!open) closeDiscard()
        }}
        heading="Discard this proposal?"
        body={
          <p>
            It is deleted and never becomes memory. This cannot be undone; an
            agent can propose it again.
          </p>
        }
        confirmLabel="Discard proposal"
        confirmVariant="danger"
        isPending={settle.isPending}
        error={settle.error}
        onConfirm={() => void discard()}
      />
    </>
  )
}
