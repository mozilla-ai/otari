import { useEffect, useRef, useState } from "react"
import { InfoBanner } from "@/design-system/feedback/InfoBanner"
import {
  hasEarnedStarPrompt,
  mayOfferStarPrompt,
  retireStarPrompt,
  STAR_PROMPT_WINDOW_DAYS,
  type StarPromptAnswer,
} from "@/features/overview/starPrompt"
import { useWorkspaceActivation } from "@/shared/api/activation"
import { NO_BREAKDOWNS, useUsageSummary } from "@/shared/api/usage"
import { OTARI_GITHUB_URL } from "@/shared/helpers/community"
import { localMidnightIso } from "@/shared/helpers/timeRange"
import { useSelectedWorkspace } from "@/shared/hooks/SelectedWorkspace"
import { useDeployment } from "@/shared/hooks/useDeployment"
import { TELEMETRY_EVENTS } from "@/shared/telemetry/events"
// By its `@/…` specifier: it is the seam a superset build replaces.
import { useTelemetry } from "@/shared/telemetry/overlayTelemetry"

/** Local midnight at the start of the window, fixed for the life of the page. */
function windowStart(): string {
  const now = new Date()
  return localMidnightIso(
    new Date(
      now.getFullYear(),
      now.getMonth(),
      now.getDate() - (STAR_PROMPT_WINDOW_DAYS - 1),
    ),
  )
}

/**
 * One line inviting an owner or admin to star Otari on GitHub, last in the
 * Overview's status bands, offered once.
 *
 * Everything that decides it is already on this deployment: the operator's
 * setting, the caller's role, the page's own status, the usage it reads, and an
 * answer this browser kept. Nothing is fetched from GitHub, so a self-hosted
 * gateway's dashboard contacts no one to show it, and the star count is
 * deliberately not on it.
 *
 * It yields to everything else a band can say. The page passes `isQuiet` only
 * once its status sources have answered and none of them has anything to
 * report, so the line never asks for a favor beside a provider that is down or
 * a budget at its limit, and never flashes in ahead of a strip that would have
 * replaced it.
 *
 * The setup guide it checks for itself, because usage does not rule it out:
 * imported agent telemetry lands in the usage tables without being a request
 * to this gateway, so a workspace can hold 25 successes while its first-request
 * guide is still on offer (the e2e fixture is exactly that). The getting-started
 * strip needs no check, since it only shows a workspace with no usage at all.
 */
export function StarPrompt({
  canAsk,
  isQuiet,
}: {
  /** Whether the caller is one this asks: an owner, an admin, or the operator. */
  canAsk: boolean
  /** Whether the page has finished loading with nothing needing attention. */
  isQuiet: boolean
}) {
  const { star_prompt_enabled } = useDeployment()
  const { selected: workspace } = useSelectedWorkspace()
  const { recordEvent } = useTelemetry()
  const [isRetired, setIsRetired] = useState(() => !mayOfferStarPrompt())
  // Fixed rather than advanced at midnight like the attention strip's "today":
  // a lookback this long is not wrong by a day, and a fixed bound keeps the
  // query key, and so the cache entry, steady while the page is open.
  const [since] = useState(windowStart)

  const mayAsk = star_prompt_enabled && canAsk && !isRetired
  // Not asked at all once any gate above has said no, so an operator who
  // withdrew it, a member, and a browser that answered cost the gateway nothing.
  const successes = useUsageSummary(
    {
      workspace_id: workspace?.workspace_id,
      start_date: since,
      status: "success",
    },
    "day",
    NO_BREAKDOWNS,
    mayAsk,
  )
  const hasEarned = mayAsk && hasEarnedStarPrompt(successes.data)
  // The same read the setup guide makes, so the two share one cache entry, and
  // made only once the usage has earned the line, so nobody else pays for it.
  // Without a workspace there is no guide to wait for: it renders nothing then.
  const workspaceId = workspace?.workspace_id ?? null
  const activation = useWorkspaceActivation(workspaceId, hasEarned)
  const isGuideSettled = workspaceId === null || !activation.isPending
  // On offer means the guide may be on screen. A failed read leaves the guide
  // with nothing to render, so it does not hold this back.
  const isGuideOnOffer =
    activation.data?.experience_eligible === true &&
    activation.data.status !== "activated"
  const isShown = hasEarned && isQuiet && isGuideSettled && !isGuideOnOffer

  // Once per page view, not per render or per remount within it, so a superset
  // build that records it counts people offered the line rather than renders.
  const wasRecorded = useRef(false)
  useEffect(() => {
    if (isShown && !wasRecorded.current) {
      wasRecorded.current = true
      recordEvent(TELEMETRY_EVENTS.STAR_PROMPT_SHOWN)
    }
  }, [isShown, recordEvent])

  if (!isShown) {
    return null
  }

  const retire = (answer: StarPromptAnswer) => {
    retireStarPrompt(answer)
    setIsRetired(true)
  }

  return (
    <InfoBanner
      dismissLabel="Dismiss the invitation to star Otari"
      onDismiss={() => {
        recordEvent(TELEMETRY_EVENTS.STAR_PROMPT_DISMISSED)
        retire("dismissed")
        // The control that had focus is about to leave the page. The main
        // region is where the skip link lands, it takes focus without joining
        // the tab order, and the line sat at its top, so a keyboard user carries
        // on from roughly where they were rather than from the document's start.
        document.getElementById("main-content")?.focus({ preventScroll: true })
      }}
    >
      Otari is open source. A star helps other developers find it.{" "}
      <a
        href={OTARI_GITHUB_URL}
        target="_blank"
        rel="noopener noreferrer"
        className="text-link underline underline-offset-2 hover:decoration-2"
        // A star happens on GitHub, out of sight, so following the link is the
        // answer this page can see. Retired on the click rather than on a
        // return, which would mean watching for one.
        onClick={() => {
          recordEvent(TELEMETRY_EVENTS.STAR_PROMPT_CLICKED)
          retire("starred")
        }}
      >
        Star Otari on GitHub
      </a>
    </InfoBanner>
  )
}
