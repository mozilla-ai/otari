import { useEffect, useState } from "react"
import { announce } from "@/app/announce"
import { Dot } from "@/design-system/indicators/Dot"
import { useGatewayUnreachable } from "@/shared/api/deployment"
import { formatClockTime } from "@/shared/helpers/format"

/** How long "Reconnected" stays before the band goes. */
export const RECONNECTED_MS = 2_000

export type Band = { state: "down"; since: number } | { state: "back" } | null

/**
 * The gateway not answering, as a band pinned to the top of the content pane.
 *
 * A whole-app condition, so it sits above any page rather than in one, and it
 * is a band in the flow rather than a toast over it: a toast covered the
 * pagination it was floated over, and on a phone it ran to the screen's edge.
 * The band pushes the page down instead and stays in view while it scrolls.
 * `main` scrolls, so this is its first child, ahead of the padded column.
 *
 * Shown only once a liveness probe has confirmed what a failed request
 * suggested (`useGatewayUnreachable`): one dropped request is routine. When
 * the gateway answers again the band says so for a moment, while the pages
 * that failed refetch, and then goes. Not dismissible: it is tied to live
 * state.
 *
 * `InfoBanner`'s anatomy (the dot, the muted sentence, the prose measure)
 * composed here rather than `InfoBanner` itself, because a band pinned under
 * the header takes one rule below it rather than one each side, spans the pane
 * with the page's own gutters, and carries a status word at its end and a
 * success dot on recovery, none of which that component has.
 *
 * Not a live region: the content pane already holds the pages' own loading
 * status, and a second region there is one more thing to talk over it. A
 * screen reader is told the band's sentence once when it appears, and once
 * when it turns to "Reconnected", through the document's one announcer.
 */
export function ConnectionStatus() {
  const down = useGatewayUnreachable()
  const [band, setBand] = useState<Band>(null)

  useEffect(() => {
    if (down) {
      // The time is held from the moment the outage was confirmed: everything
      // on screen loaded before it, and it must not creep forward while later
      // requests keep failing.
      setBand((current) =>
        current?.state === "down"
          ? current
          : { state: "down", since: Date.now() },
      )
      return
    }
    setBand((current) =>
      current?.state === "down" ? { state: "back" } : current,
    )
  }, [down])

  const said = band === null ? null : bandSentence(band)
  useEffect(() => {
    if (said !== null) {
      announce(said)
    }
  }, [said])

  useEffect(() => {
    if (band?.state !== "back") {
      return
    }
    const timer = setTimeout(() => setBand(null), RECONNECTED_MS)
    return () => clearTimeout(timer)
  }, [band])

  return <ConnectionBand band={band} />
}

const DOWN_LEAD = "Can’t reach the gateway."
const BACK_LEAD = "Reconnected."
const BACK_REST = "Refreshing this page."

function downRest(since: number): string {
  return `Showing what loaded before ${formatClockTime(since, { seconds: false })}. Changes won’t save until it answers.`
}

function bandSentence(band: Exclude<Band, null>): string {
  return band.state === "down"
    ? `${DOWN_LEAD} ${downRest(band.since)}`
    : `${BACK_LEAD} ${BACK_REST}`
}

/** The band itself, drawn from `band` alone so each state renders on its own. */
export function ConnectionBand({ band }: { band: Band }) {
  if (band === null) {
    return null
  }
  return (
    <div
      data-connection-band={band.state}
      className="sticky top-0 z-10 flex items-start gap-3 border-b border-border bg-background px-4 py-3 text-sm text-muted md:px-6"
    >
      <Dot
        className={`mt-2 ${band.state === "down" ? "bg-danger" : "bg-success"}`}
      />
      <p className="min-w-0 max-w-prose flex-1">
        <strong className="font-semibold text-foreground">
          {band.state === "down" ? DOWN_LEAD : BACK_LEAD}
        </strong>{" "}
        {band.state === "down" ? downRest(band.since) : BACK_REST}
      </p>
      {band.state === "down" ? (
        <span
          aria-hidden
          className="ml-auto shrink-0 pt-0.5 text-mono-overline text-subtle"
        >
          Retrying
        </span>
      ) : null}
    </div>
  )
}
