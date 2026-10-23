import { useState } from "react"

import type {
  TraceContentShare,
  TraceDetail,
  TraceShareAudience,
} from "@/client"
import { Button } from "@/design-system/actions/Button"
import { ErrorBanner } from "@/design-system/feedback/ErrorBanner"
import { FormDialog } from "@/design-system/feedback/FormDialog"
import { InfoBanner } from "@/design-system/feedback/InfoBanner"
import { RadioGroup } from "@/design-system/forms/RadioGroup"
import { Select } from "@/design-system/forms/Select"
import { TextArea } from "@/design-system/forms/TextArea"
import {
  useEndTraceShare,
  useShareTrace,
  useTraceShares,
} from "@/shared/api/traces"
import { formatDate, formatDateTime } from "@/shared/helpers/format"

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

const DAY_OPTIONS = [
  { value: "7", label: "7 days" },
  { value: "30", label: "30 days" },
  { value: "90", label: "90 days" },
] as const

function isAudience(value: string): value is TraceShareAudience {
  return value in AUDIENCE_LABELS
}

/**
 * Who can read what a session was about, said where the session is read.
 *
 * Its own user is told it is private to them and can share it; someone reading
 * through a share is told whose say that is and that the read was recorded;
 * anyone else is told the content exists and is not theirs to read. A session
 * no user owns (a service key's) has no one to share it, so it offers nothing.
 */
export function SessionContent({ detail }: { detail: TraceDetail }) {
  const access = detail.content_access
  const holdsPrivate = detail.spans.some((span) => span.content_private)
  const holdsContent = detail.spans.some(
    (span) => Object.keys(span.content).length > 0,
  )
  const owned = access === "owner" && detail.summary.user_id !== null

  if (access === "share" && detail.content_share) {
    const share = detail.content_share
    return (
      <InfoBanner>
        Shared with you by the person whose session this is, until{" "}
        {formatDate(share.expires_at)}.
        {share.note ? ` Their note: “${share.note}”.` : ""} They can see that
        you read it.
      </InfoBanner>
    )
  }
  if (holdsPrivate) {
    return (
      <InfoBanner>
        What this session was about is private to the person whose session it
        is. They can share it with you from this page.
      </InfoBanner>
    )
  }
  if (owned && holdsContent) {
    return <OwnerSharing detail={detail} />
  }
  return null
}

function OwnerSharing({ detail }: { detail: TraceDetail }) {
  const traceId = detail.summary.trace_id
  const workspaceId = detail.summary.workspace_id
  const shares = useTraceShares(traceId, workspaceId, true)
  const endShare = useEndTraceShare()
  const [isSharing, setIsSharing] = useState(false)
  const active = (shares.data?.items ?? []).filter((share) => share.active)

  return (
    <div className="flex flex-col gap-3">
      <InfoBanner>
        Only you can read what this session was about: its prompts, commands and
        file paths. Share it when you want someone to look into it.
      </InfoBanner>
      <ErrorBanner error={shares.error ?? endShare.error} />
      <div className="flex flex-wrap items-center gap-2">
        <Button variant="primary" onPress={() => setIsSharing(true)}>
          Share this session
        </Button>
      </div>
      {active.length > 0 ? (
        <ul className="flex flex-col gap-2" aria-label="Shares of this session">
          {active.map((share) => (
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
      {isSharing ? (
        <ShareDialog
          traceId={traceId}
          workspaceId={workspaceId}
          onClose={() => setIsSharing(false)}
        />
      ) : null}
    </div>
  )
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
  return (
    <li className="flex flex-col gap-1 border border-border p-3">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <span className="text-body">
          {AUDIENCE_LABELS[share.audience]}, until{" "}
          {formatDate(share.expires_at)}
        </span>
        <Button variant="ghost" onPress={onEnd} isDisabled={isEnding}>
          Stop sharing
        </Button>
      </div>
      {share.note ? <span className="text-caption">“{share.note}”</span> : null}
      <span className="text-caption">
        {reads.length === 0
          ? "Nobody has read it yet."
          : reads
              .map(
                (read) =>
                  `Read by ${read.reader_email} on ${formatDateTime(read.read_at)}`,
              )
              .join(". ")}
      </span>
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
  const share = useShareTrace()
  const [audience, setAudience] = useState<TraceShareAudience>(
    "organization_admins",
  )
  const [note, setNote] = useState("")
  const [days, setDays] = useState("30")

  const submit = () => {
    if (share.isPending) return
    share.mutate(
      {
        traceId,
        workspaceId,
        body: {
          audience,
          note: note.trim() || null,
          days: Number(days),
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
      description="Session content is encrypted and only you can read it. Whoever you share it with can read its prompts, commands and file paths until the share ends or you stop it. Each time they open it, you can see it here."
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
      <Select
        label="For"
        value={days}
        onChange={setDays}
        options={DAY_OPTIONS}
      />
    </FormDialog>
  )
}
