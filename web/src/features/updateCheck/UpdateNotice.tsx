import { useState } from "react"
import { Button } from "@/design-system/actions/Button"
import { InfoBanner } from "@/design-system/feedback/InfoBanner"
import {
  newestVersion,
  useOperatorUpdateCheck,
} from "@/features/updateCheck/useOperatorUpdateCheck"

const DISMISSED_KEY = "otari.updateNotice.dismissed"
// How long a dismissed release keeps any newer one quiet too.
const QUIET_MS = 30 * 24 * 60 * 60 * 1000

interface Dismissed {
  version: string
  at: number
}

function readDismissed(): Dismissed | null {
  try {
    const parsed: unknown = JSON.parse(
      localStorage.getItem(DISMISSED_KEY) ?? "null",
    )
    return typeof parsed === "object" &&
      parsed !== null &&
      "version" in parsed &&
      "at" in parsed &&
      typeof parsed.version === "string" &&
      typeof parsed.at === "number"
      ? { version: parsed.version, at: parsed.at }
      : null
  } catch {
    return null
  }
}

function writeDismissed(dismissed: Dismissed) {
  try {
    localStorage.setItem(DISMISSED_KEY, JSON.stringify(dismissed))
  } catch {
    // Storage refused: the notice is dismissed for this tab only.
  }
}

function isQuiet(dismissed: Dismissed | null, latest: string): boolean {
  return (
    dismissed !== null &&
    (dismissed.version === latest || Date.now() - dismissed.at < QUIET_MS)
  )
}

/**
 * The shell's band for what the last update check heard: a security advisory
 * that covers the running version, or a newer release.
 *
 * An advisory is dismissed for the tab and comes back on the next load while
 * it still applies. A new release is dismissed in this browser for that
 * release, and for 30 days for any later one, so a busy release week does not
 * bring the band back daily.
 */
export function UpdateNotice() {
  const status = useOperatorUpdateCheck()
  const [dismissed, setDismissed] = useState(readDismissed)
  const [isAdvisoryDismissed, setIsAdvisoryDismissed] = useState(false)

  if (status === undefined || status.enabled !== true) {
    return null
  }

  const latest = status.latest_version ?? ""
  const advisories = isAdvisoryDismissed ? [] : status.advisories
  const showsRelease =
    advisories.length === 0 &&
    status.update_available &&
    latest !== "" &&
    !isQuiet(dismissed, latest)

  if (advisories.length === 0 && !showsRelease) {
    return null
  }

  const fixedIn = newestVersion(
    advisories
      .map((advisory) => advisory.fixed_in ?? "")
      .filter((version) => version !== ""),
  )

  return (
    // The same full-bleed band as `PricingWarning`, for the same reasons.
    <div
      data-slot="update-notice"
      className="mx-auto w-full max-w-[112.5rem] shrink-0 px-4 md:px-6"
    >
      <InfoBanner tone={advisories.length > 0 ? "warning" : "info"}>
        <div className="flex flex-wrap items-center justify-between gap-3">
          {advisories.length > 0 ? (
            <span>
              Otari {status.current_version} is affected by{" "}
              {advisories.length === 1
                ? "a security advisory"
                : `${advisories.length} security advisories`}
              :{" "}
              {advisories.map((advisory, index) => (
                <span key={advisory.id}>
                  {index > 0 ? ", " : ""}
                  <a
                    href={advisory.url}
                    target="_blank"
                    rel="noreferrer"
                    className="underline underline-offset-2"
                  >
                    {advisory.summary}
                  </a>{" "}
                  ({advisory.severity})
                </span>
              ))}
              .{" "}
              {fixedIn !== ""
                ? `Upgrade to ${fixedIn} or later.`
                : "Upgrade to the latest release."}
            </span>
          ) : (
            <span>
              Otari {latest} is available. You&rsquo;re running{" "}
              {status.current_version}.{" "}
              {status.release_url ? (
                <a
                  href={status.release_url}
                  target="_blank"
                  rel="noreferrer"
                  className="underline underline-offset-2"
                >
                  What&rsquo;s new
                </a>
              ) : null}
            </span>
          )}
          <Button
            size="sm"
            onPress={() => {
              if (advisories.length > 0) {
                setIsAdvisoryDismissed(true)
                return
              }
              const next = { version: latest, at: Date.now() }
              writeDismissed(next)
              setDismissed(next)
            }}
          >
            Dismiss
          </Button>
        </div>
      </InfoBanner>
    </div>
  )
}
