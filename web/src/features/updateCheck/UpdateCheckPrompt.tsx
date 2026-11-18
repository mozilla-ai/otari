import { useState } from "react"
import { Button } from "@/design-system/actions/Button"
import { Dialog, DialogSection } from "@/design-system/feedback/Dialog"
import { ErrorBanner } from "@/design-system/feedback/ErrorBanner"
import {
  endpointHost,
  UPDATE_CHECK_DOC,
  useOperatorUpdateCheck,
} from "@/features/updateCheck/useOperatorUpdateCheck"
import { useHostedProviders } from "@/shared/api/hostedProviders"
import { useProviders } from "@/shared/api/providers"
import { useAnswerUpdateCheck } from "@/shared/api/updateCheck"
import { docsSourceHref } from "@/shared/helpers/docs"

const SNOOZED_KEY = "otari.updateCheckPrompt.snoozedUntil"
// How long closing the question without answering keeps it away.
const SNOOZE_MS = 7 * 24 * 60 * 60 * 1000

function readSnoozedUntil(): number {
  try {
    const parsed: unknown = JSON.parse(
      localStorage.getItem(SNOOZED_KEY) ?? "null",
    )
    return typeof parsed === "object" &&
      parsed !== null &&
      "until" in parsed &&
      typeof parsed.until === "number"
      ? parsed.until
      : 0
  } catch {
    return 0
  }
}

function writeSnoozedUntil(until: number) {
  try {
    localStorage.setItem(SNOOZED_KEY, JSON.stringify({ until }))
  } catch {
    // Storage refused: the question stays away for this tab only.
  }
}

/**
 * Whether the deployment already served a provider when the dashboard loaded,
 * the condition that hides `OnboardingPanel`. Read once rather than live, so
 * adding the first provider does not raise the question straight after.
 */
function useHadProviderAtLoad(enabled: boolean): boolean {
  const configured = useProviders(enabled)
  const hosted = useHostedProviders(enabled)
  const [atLoad, setAtLoad] = useState<boolean | undefined>(undefined)
  if (atLoad === undefined && !configured.isPending && !hosted.isPending) {
    setAtLoad(
      (configured.data?.providers ?? []).length +
        (hosted.data?.data ?? []).length >
        0,
    )
  }
  return atLoad === true
}

/**
 * Asks a deployment operator, once, whether to check for new releases and
 * security fixes. Nothing is sent until somebody answers here or config does.
 *
 * Only asked where the answer does something: never on a development build,
 * which is never checked, and never where config already answered. Never on
 * a first visit either: someone who just deployed came to connect a provider,
 * so the question waits for a load that finds one already set up.
 *
 * Both answers are stored, so the question does not come back. Closing the
 * frame without answering is neither: it is snoozed in this browser for a
 * week, because no answer is not a no.
 */
export function UpdateCheckPrompt() {
  const status = useOperatorUpdateCheck()
  const hadProviderAtLoad = useHadProviderAtLoad(status !== undefined)
  const answer = useAnswerUpdateCheck()
  const [snoozedUntil, setSnoozedUntil] = useState(readSnoozedUntil)

  const isAsking =
    status !== undefined &&
    status.answered_by === null &&
    status.checkable &&
    hadProviderAtLoad &&
    Date.now() >= snoozedUntil

  if (!isAsking) {
    return null
  }

  const pressed = answer.isPending ? answer.variables.enabled : undefined

  return (
    <Dialog
      isOpen
      onOpenChange={(open) => {
        if (!open) {
          const until = Date.now() + SNOOZE_MS
          writeSnoozedUntil(until)
          setSnoozedUntil(until)
        }
      }}
      title="Get alerts for security fixes and new releases?"
      actions={
        <>
          <Button
            isDisabled={pressed === true}
            isPending={pressed === false}
            onPress={() => answer.mutate({ enabled: false })}
          >
            No thanks
          </Button>
          <Button
            variant="primary"
            isDisabled={pressed === false}
            isPending={pressed === true}
            onPress={() => answer.mutate({ enabled: true })}
          >
            Turn on update checks
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
          {endpointHost(status.endpoint)} once a day. Nothing about your users,
          keys, providers or traffic. otari.ai also uses these checks to count
          active installs by version.{" "}
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
