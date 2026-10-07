import { useState } from "react"
import type { SpanContent, TraceSpan } from "@/client"
import { Button } from "@/design-system/actions/Button"
import { EmptyMessage } from "@/design-system/feedback/EmptyMessage"
import { FormDialog } from "@/design-system/feedback/FormDialog"
import { TextArea } from "@/design-system/forms/TextArea"
import { ApiError } from "@/shared/api/client"
import {
  useBreakGlassContent,
  useSpanContent,
  useTraceScope,
} from "@/shared/api/traces"

const FIELD_LABEL: Record<string, string> = {
  input: "Input",
  output: "Model output",
  prior_output: "Model output before this request",
  arguments: "Arguments",
  result: "Result",
}

// The order fields read in, whatever order the server sent them: a field this
// list does not name goes last.
const FIELD_ORDER = Object.keys(FIELD_LABEL)

/** The bounds the server holds a break-glass reason to. */
export const REASON_MIN = 10
export const REASON_MAX = 500

function rank(key: string): number {
  const index = FIELD_ORDER.indexOf(key)
  return index === -1 ? FIELD_ORDER.length : index
}

function hasStatus(error: unknown, status: number): boolean {
  return error instanceof ApiError && error.status === status
}

const NOT_CAPTURED = (
  <EmptyMessage>
    Content was not captured for this span. A workspace admin can turn on
    content capture for the workspace; until then traces record what ran, how
    long it took and what it cost, never the prompt, the output or a tool's
    input and output.
  </EmptyMessage>
)

const UNAVAILABLE = (
  <EmptyMessage>
    This span's content cannot be read right now because the key store is
    unavailable. Try again shortly.
  </EmptyMessage>
)

const GONE = (
  <EmptyMessage>
    This span's content has expired, was purged, or was not captured.
  </EmptyMessage>
)

function ContentFields({ content }: { content: SpanContent }) {
  const fields = Object.entries(content.fields)
    .filter(([, value]) => value !== "")
    .sort(([a], [b]) => rank(a) - rank(b))
  return (
    <div className="flex flex-col gap-3">
      {fields.map(([key, value]) => (
        <section
          key={key}
          aria-label={FIELD_LABEL[key] ?? key}
          className="flex flex-col gap-1"
        >
          <h5 className="text-overline">{FIELD_LABEL[key] ?? key}</h5>
          <pre className="max-h-80 overflow-auto whitespace-pre-wrap break-words rounded-md bg-surface-alt p-3 text-caption font-mono">
            {value}
          </pre>
        </section>
      ))}
    </div>
  )
}

/**
 * The reason a platform operator gives before reading a span's content. Its
 * whole job is to make the read deliberate and leave a record anyone in the
 * workspace's administration can read back.
 */
export function BreakGlassDialog({
  isOpen,
  onOpenChange,
  onConfirm,
  isPending,
  error,
}: {
  isOpen: boolean
  onOpenChange: (open: boolean) => void
  onConfirm: (reason: string) => void
  isPending: boolean
  error: unknown
}) {
  const [reason, setReason] = useState("")
  const length = reason.trim().length
  const isValid = length >= REASON_MIN && length <= REASON_MAX
  return (
    <FormDialog
      isOpen={isOpen}
      onOpenChange={(open) => {
        if (!open) setReason("")
        onOpenChange(open)
      }}
      title="Break glass"
      description="Read this span's captured content as a platform operator. Use it only for a legal or safety request."
      submitLabel="Read content"
      onSubmit={() => onConfirm(reason.trim())}
      isPending={isPending}
      error={error}
      isDirty={reason !== ""}
      isSubmitDisabled={!isValid}
    >
      <TextArea
        label="Reason"
        value={reason}
        onChange={setReason}
        isRequired
        placeholder="Legal hold requested by counsel, case 2026-114"
        description={`Between ${REASON_MIN} and ${REASON_MAX} characters (${length} so far). Recorded with the read and shown to the workspace's admins.`}
        isInvalid={length > REASON_MAX}
        errorMessage={
          length > REASON_MAX
            ? `Keep the reason to ${REASON_MAX} characters.`
            : undefined
        }
        shouldReserveMessage
      />
    </FormDialog>
  )
}

// The operator's deployment-wide view: nothing is read until a reason is given.
function BreakGlassContent({
  traceId,
  span,
}: {
  traceId: string
  span: TraceSpan
}) {
  const [isOpen, setIsOpen] = useState(false)
  const breakGlass = useBreakGlassContent()
  // The mutation outlives a change of span in this same slot, so only its
  // answer for this span counts.
  const isThisSpan =
    breakGlass.variables?.traceId === traceId &&
    breakGlass.variables?.spanId === span.span_id
  if (isThisSpan && breakGlass.data) {
    return <ContentFields content={breakGlass.data} />
  }
  if (isThisSpan && !isOpen && hasStatus(breakGlass.error, 404)) return GONE
  if (isThisSpan && !isOpen && hasStatus(breakGlass.error, 503)) {
    return UNAVAILABLE
  }
  return (
    <div className="flex flex-col items-start gap-3">
      <EmptyMessage>
        Platform operators read a session's content only through a recorded
        break-glass with a stated reason, which the workspace's admins can see.
      </EmptyMessage>
      <Button variant="danger" onPress={() => setIsOpen(true)}>
        Break glass
      </Button>
      <BreakGlassDialog
        isOpen={isOpen}
        onOpenChange={(open) => {
          setIsOpen(open)
          if (!open && !breakGlass.isSuccess) breakGlass.reset()
        }}
        isPending={breakGlass.isPending}
        error={breakGlass.error}
        onConfirm={(reason) =>
          breakGlass.mutate(
            { traceId, spanId: span.span_id, reason },
            {
              onSuccess: () => setIsOpen(false),
              onError: (error) => {
                // These two are answers about the content rather than about the
                // request, so they replace the action instead of sitting in the dialog.
                if (hasStatus(error, 404) || hasStatus(error, 503)) {
                  setIsOpen(false)
                }
              },
            },
          )
        }
      />
    </div>
  )
}

function MemberContent({
  traceId,
  span,
}: {
  traceId: string
  span: TraceSpan
}) {
  const content = useSpanContent(traceId, span.span_id, true)
  if (content.isError && !content.data) {
    if (hasStatus(content.error, 503)) return UNAVAILABLE
    if (hasStatus(content.error, 403)) {
      return (
        <EmptyMessage>
          Only the person who ran this session can read its content, unless a
          workspace admin lets organization admins read it.
        </EmptyMessage>
      )
    }
    return GONE
  }
  if (!content.data) {
    return <EmptyMessage>Loading the content…</EmptyMessage>
  }
  return <ContentFields content={content.data} />
}

// One span's captured content, read when the span is opened. This part fetches for
// itself, against the rule that parts take data, because each read is recorded on
// the server and should happen only for the span a person actually opens.
export function SpanContentSection({
  traceId,
  span,
}: {
  traceId: string
  span: TraceSpan
}) {
  const scope = useTraceScope()
  if (!span.has_content) return NOT_CAPTURED
  if (!scope.isReady) {
    return <EmptyMessage>Loading the content…</EmptyMessage>
  }
  if (scope.isDeploymentWide) {
    return <BreakGlassContent traceId={traceId} span={span} />
  }
  return <MemberContent traceId={traceId} span={span} />
}
