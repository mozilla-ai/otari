import { useState } from "react"
import { FiLock } from "react-icons/fi"

import type {
  TraceContentShare,
  TraceDetail,
  TraceShareAudience,
} from "@/client"
import { Button } from "@/design-system/actions/Button"
import { IconButton } from "@/design-system/actions/IconButton"
import { ErrorBanner } from "@/design-system/feedback/ErrorBanner"
import { FormDialog } from "@/design-system/feedback/FormDialog"
import { InfoBanner } from "@/design-system/feedback/InfoBanner"
import { RadioGroup } from "@/design-system/forms/RadioGroup"
import { TextArea } from "@/design-system/forms/TextArea"
import { Popover } from "@/design-system/overlays/Popover"
import {
  useBreakGlass,
  useEndTraceShare,
  useShareTrace,
  useTraceContentPolicy,
  useTraceShares,
} from "@/shared/api/traces"
import { formatDate, formatDateTime } from "@/shared/helpers/format"

import { exceptionalAccess } from "./exceptionalAccess"

const AUDIENCE_LABELS: Record<TraceShareAudience, string> = {
  organization_admins: "Your organization's owners and admins",
  deployment_operator: "The team that runs this deployment",
}

const AUDIENCE_OPTIONS = [
  {
    value: "organization_admins",
    label: AUDIENCE_LABELS.organization_admins,
  },
  {
    value: "deployment_operator",
    label: AUDIENCE_LABELS.deployment_operator,
    description: "For a support request about this session.",
  },
] as const

// The server refuses a shorter reason; the dialog does not offer to send one.
const BREAK_GLASS_REASON_MIN = 10

function isAudience(value: string): value is TraceShareAudience {
  return value in AUDIENCE_LABELS
}

/**
 * Who can read what a session was about, said where the session is read.
 *
 * Its own user is told it is private to them and can share it; someone reading
 * through a share is told whose say that is and that the read was recorded;
 * anyone else is told the content exists and is not theirs to read. An owner
 * of the organization is offered break-glass for it, recorded with a reason
 * that every owner can see. A session no user owns has no one to share it, so
 * it offers nothing.
 */
export function SessionContent({
  detail,
  onBreakGlass,
}: {
  detail: TraceDetail
  /** Takes the session as break-glass opened it; offered to an organization owner only. */
  onBreakGlass?: (opened: TraceDetail) => void
}) {
  const access = detail.content_access
  const holdsPrivate = detail.spans.some((span) => span.content_private)
  const owned = access === "owner" && detail.summary.user_id !== null

  if (access === "break_glass") {
    return (
      <InfoBanner tone="warning">
        Opened by break-glass, without its owner's share. The opening is
        recorded with your reason, and every owner of the organization can see
        it.
      </InfoBanner>
    )
  }
  if (access === "share" && detail.content_share) {
    const share = detail.content_share
    return (
      <InfoBanner>
        Shared with you by the person whose session this is.
        {share.note ? ` Their note: “${share.note}”.` : ""} They can see that
        you read it, and when.
      </InfoBanner>
    )
  }
  if (holdsPrivate) {
    return (
      <div className="flex flex-col gap-3">
        <InfoBanner>
          What this session was about is private to the person whose session it
          is. They can share it with the organization's admins, or with the team
          that runs this deployment, from this page.
        </InfoBanner>
        {onBreakGlass ? (
          <BreakGlassAction detail={detail} onOpened={onBreakGlass} />
        ) : null}
      </div>
    )
  }
  // On ownership alone, not on the content having opened: during a key-source
  // outage the content reads back empty, and the owner must still be able to
  // stop a share and see who read it.
  if (owned) return <OwnerSharing detail={detail} />
  return null
}

/** Whether the reader owns the session and it kept something of theirs to share. */
function isSharable(detail: TraceDetail): boolean {
  return (
    detail.content_access === "owner" &&
    detail.summary.user_id !== null &&
    !detail.spans.some((span) => span.content_private) &&
    detail.spans.some((span) => Object.keys(span.content).length > 0)
  )
}

/**
 * The owner's share control, for the panel's toolbar: who can read the session,
 * behind a lock that opens it on a press (a tap on a phone, where a tooltip
 * would never show), and the button that shares it.
 */
export function ShareSessionAction({ detail }: { detail: TraceDetail }) {
  const platformBreakGlass = useTraceContentPolicy().policy.platform_break_glass
  const [isSharing, setIsSharing] = useState(false)
  if (!isSharable(detail)) return null
  return (
    <div className="flex items-center gap-1">
      <Popover
        label="Who can read this session"
        placement="bottom end"
        trigger={
          <IconButton label="Who can read this session" size="sm">
            <FiLock aria-hidden="true" className="size-4" />
          </IconButton>
        }
      >
        <p className="max-w-80 text-body">
          Otari shows what this session was about, its prompts, commands and
          file paths, only to you. {exceptionalAccess(platformBreakGlass)} Share
          it when you want someone to look into it.
        </p>
      </Popover>
      <Button size="sm" onPress={() => setIsSharing(true)}>
        Share this session
      </Button>
      {isSharing ? (
        <ShareDialog
          traceId={detail.summary.trace_id}
          workspaceId={detail.summary.workspace_id}
          onClose={() => setIsSharing(false)}
        />
      ) : null}
    </div>
  )
}

function BreakGlassAction({
  detail,
  onOpened,
}: {
  detail: TraceDetail
  onOpened: (opened: TraceDetail) => void
}) {
  const breakGlass = useBreakGlass()
  const [isOpen, setIsOpen] = useState(false)
  const [reason, setReason] = useState("")
  const trimmed = reason.trim()
  // The reason stays for a second try; a failed attempt's error does not.
  const close = () => {
    setIsOpen(false)
    breakGlass.reset()
  }

  const submit = () => {
    if (breakGlass.isPending || trimmed.length < BREAK_GLASS_REASON_MIN) return
    breakGlass.mutate(
      {
        traceId: detail.summary.trace_id,
        workspaceId: detail.summary.workspace_id,
        reason: trimmed,
      },
      {
        onSuccess: (opened) => {
          setIsOpen(false)
          setReason("")
          onOpened(opened)
        },
      },
    )
  }

  return (
    <>
      <div>
        <Button variant="ghost" onPress={() => setIsOpen(true)}>
          Open by break-glass
        </Button>
      </div>
      {isOpen ? (
        <FormDialog
          isOpen
          onOpenChange={(open) => {
            if (!open) close()
          }}
          title="Open this session by break-glass"
          description="For an investigation its owner cannot be asked to share for, such as an HR or legal matter. The opening is recorded with your reason and shown to every owner of this organization. The session's owner is not told."
          submitLabel="Open and record"
          onSubmit={submit}
          isPending={breakGlass.isPending}
          isSubmitDisabled={trimmed.length < BREAK_GLASS_REASON_MIN}
          error={breakGlass.error}
        >
          <TextArea
            label="Reason"
            value={reason}
            onChange={setReason}
            rows={3}
            isRequired
            description="At least ten characters. Name the matter, and who approved it."
          />
        </FormDialog>
      ) : null}
    </>
  )
}

// The owner's shares, with who read them; sharing itself is in the toolbar.
function OwnerSharing({ detail }: { detail: TraceDetail }) {
  const traceId = detail.summary.trace_id
  const workspaceId = detail.summary.workspace_id
  const shares = useTraceShares(traceId, workspaceId, true)
  const endShare = useEndTraceShare()
  // Pages are offsets over an order that moves as shares end, so a share can
  // appear on two of them; each is listed once.
  const items = [
    ...new Map(
      (shares.data?.pages ?? [])
        .flatMap((page) => page.items)
        .map((share) => [share.id, share]),
    ).values(),
  ]

  // A session never shared has nothing to list.
  if (items.length === 0 && !shares.error) return null

  return (
    <div className="flex flex-col gap-3">
      <ErrorBanner error={shares.error ?? endShare.error} />
      {shares.hasNextPage ? (
        <div>
          <Button
            variant="ghost"
            onPress={() => void shares.fetchNextPage()}
            isDisabled={shares.isFetchingNextPage}
          >
            Show older shares
          </Button>
        </div>
      ) : null}
      {items.length > 0 ? (
        // Ended shares stay listed, with who read them.
        <ul className="flex flex-col gap-2" aria-label="Shares of this session">
          {items.map((share) => (
            <ShareRow
              key={share.id}
              share={share}
              isEnding={endShare.isPending}
              onEnd={() =>
                endShare.mutate({ traceId, workspaceId, shareId: share.id })
              }
            />
          ))}
        </ul>
      ) : null}
    </div>
  )
}

function shareState(share: TraceContentShare): string {
  if (share.revoked_at) return `ended ${formatDate(share.revoked_at)}`
  return `since ${formatDate(share.created_at)}`
}

function ShareRow({
  share,
  isEnding,
  onEnd,
}: {
  share: TraceContentShare
  isEnding: boolean
  onEnd: () => void
}) {
  const reads = share.reads ?? []
  const readCount = share.read_count ?? reads.length
  return (
    <li className="flex flex-col gap-1 border border-border p-3">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <span className="text-body">
          {AUDIENCE_LABELS[share.audience]}, {shareState(share)}
        </span>
        {share.active ? (
          <Button variant="ghost" onPress={onEnd} isDisabled={isEnding}>
            Stop sharing
          </Button>
        ) : null}
      </div>
      {share.note ? <span className="text-caption">“{share.note}”</span> : null}
      {reads.length === 0 ? (
        <span className="text-caption">Nobody has read it.</span>
      ) : (
        <ul className="flex flex-col text-caption" aria-label="Who read it">
          {reads.map((read) => (
            <li key={`${read.reader_email}-${read.read_at}`}>
              Read by {read.reader_email} on {formatDateTime(read.read_at)}
            </li>
          ))}
          {readCount > reads.length ? (
            <li>
              The newest {reads.length} of {readCount} reads.
            </li>
          ) : null}
        </ul>
      )}
    </li>
  )
}

function ShareDialog({
  traceId,
  workspaceId,
  onClose,
}: {
  traceId: string
  workspaceId: string
  onClose: () => void
}) {
  const platformBreakGlass = useTraceContentPolicy().policy.platform_break_glass
  const share = useShareTrace()
  const [audience, setAudience] = useState<TraceShareAudience>(
    "organization_admins",
  )
  const [note, setNote] = useState("")

  const submit = () => {
    if (share.isPending) return
    share.mutate(
      {
        traceId,
        workspaceId,
        body: {
          audience,
          note: note.trim() || null,
        },
      },
      { onSuccess: onClose },
    )
  }

  return (
    <FormDialog
      isOpen
      onOpenChange={(open) => {
        if (!open) onClose()
      }}
      title="Share this session"
      description={`Otari shows this session's content only to you. ${exceptionalAccess(platformBreakGlass)} Whoever you share it with can read its prompts, commands and file paths until you stop sharing, you lose access to this workspace, or the session is deleted. Their reads are recorded, at most once an hour each, and listed here.`}
      submitLabel="Share"
      onSubmit={submit}
      isPending={share.isPending}
      error={share.error}
    >
      <RadioGroup
        label="Share with"
        value={audience}
        onChange={(value) => {
          if (isAudience(value)) setAudience(value)
        }}
        options={AUDIENCE_OPTIONS}
      />
      <TextArea
        label="Why (optional)"
        value={note}
        onChange={setNote}
        rows={3}
        description="Shown to whoever reads it."
      />
    </FormDialog>
  )
}
