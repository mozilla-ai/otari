import { useState } from "react"
import { Button } from "@/design-system/actions/Button"
import { Dialog, DialogSection } from "@/design-system/feedback/Dialog"
import { ErrorBanner } from "@/design-system/feedback/ErrorBanner"
import {
  endpointHost,
  UPDATE_CHECK_DOC,
  useOperatorUpdateCheck,
} from "@/features/updateCheck/useOperatorUpdateCheck"
import { useAnswerUpdateCheck } from "@/shared/api/updateCheck"
import { docsSourceHref } from "@/shared/helpers/docs"

/**
 * Asks a deployment operator, once, whether to check for new releases and
 * security fixes. Nothing is sent until somebody answers here or config does.
 *
 * Only asked where the answer does something: never on a development build,
 * which is never checked, and never where config already answered.
 *
 * Both answers are stored, so the question does not come back. Closing the
 * frame without answering is neither: it is asked again on the next load,
 * because no answer is not a no.
 */
export function UpdateCheckPrompt() {
  const status = useOperatorUpdateCheck()
  const answer = useAnswerUpdateCheck()
  const [isPostponed, setIsPostponed] = useState(false)

  const isAsking =
    status !== undefined &&
    status.answered_by === null &&
    status.checkable &&
    !isPostponed

  if (!isAsking) {
    return null
  }

  return (
    <Dialog
      isOpen
      onOpenChange={(open) => (open ? undefined : setIsPostponed(true))}
      title="Get notified when a new Otari release is available"
      actions={
        <>
          <Button
            isDisabled={answer.isPending}
            onPress={() => answer.mutate({ enabled: false })}
          >
            No thanks
          </Button>
          <Button
            variant="primary"
            isDisabled={answer.isPending}
            onPress={() => answer.mutate({ enabled: true })}
          >
            Turn on notifications
          </Button>
        </>
      }
    >
      <DialogSection>
        <p className="text-body">
          Keeping Otari up to date helps keep your deployment secure, since new
          releases can include security fixes. We recommend turning this on.
        </p>
        <p className="text-body">
          To do this, Otari sends its version and a random install ID to{" "}
          {endpointHost(status.endpoint)}. Nothing about your users, keys,
          providers or traffic.{" "}
          <a
            href={docsSourceHref(UPDATE_CHECK_DOC, "what-is-sent")}
            target="_blank"
            rel="noreferrer"
            className="underline underline-offset-2"
          >
            See exactly what&rsquo;s sent
          </a>
        </p>
        <p className="text-caption">
          You can change this any time in Settings, or preset it with{" "}
          <code>update_check</code> in config.yml.
        </p>
        <ErrorBanner error={answer.error} />
      </DialogSection>
    </Dialog>
  )
}
