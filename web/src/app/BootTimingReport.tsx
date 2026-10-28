import { useEffect, useRef, useState } from "react"

import type { OrganizationContext } from "@/client"
import { useAuth } from "@/features/auth/AuthContext"
import { useCachedOrganizationContext } from "@/shared/api/organizations"
import {
  BOOT_MARKS,
  readBootTiming,
  whenMarked,
} from "@/shared/telemetry/bootTiming"
import { TELEMETRY_EVENTS } from "@/shared/telemetry/events"
import { useTelemetry } from "@/shared/telemetry/overlayTelemetry"
import type { TelemetryProperties } from "@/shared/telemetry/types"

/** The longest a load waits for Overview's figure before reporting without it. */
export const BOOT_REPORT_DEADLINE_MS = 30_000

// Module state, not a ref: once per *load*, and a remount (StrictMode, or a
// boundary keyed on the bootstrap) is the same load.
let reported = false

/** For tests, which render many loads in one module instance. */
export function resetBootTimingReport(): void {
  reported = false
}

type RoleBucket = "admin" | "member" | "operator"

function roleBucket(
  context: OrganizationContext | null | undefined,
): RoleBucket | undefined {
  if (!context) return undefined
  if (context.deployment_operator) return "operator"
  return context.role === "owner" || context.role === "admin"
    ? "admin"
    : "member"
}

/** Whether the load opened on Overview, from the hash it was opened at. */
function landedOnOverview(): boolean {
  const path = window.location.hash.replace(/^#/, "").split("?")[0]
  return path === "" || path === "/"
}

/**
 * Records `Dashboard Boot Timed` once per load, and renders nothing.
 *
 * Mounted by `App` above every branch, so a sign-in screen, a signed-in shell
 * and an unreachable gateway are all reported. It reports when the first of
 * three things happens: Overview's "Spend today" mark is written, the page is
 * hidden, or `BOOT_REPORT_DEADLINE_MS` has passed since navigation start. The
 * marks are read then and only then, so nothing runs while the page boots.
 *
 * A hidden page is a reason to report and not to wait: a tab sent to the
 * background stops presenting frames, so no later mark would arrive, and a tab
 * that is closed may never run the deadline. Such a load says so in
 * `hidden_during_boot`, and the analysis excludes it.
 *
 * `t_kpi_ms` is sent only for a load that was signed in and opened on
 * Overview: anywhere else the mark measures a sign-in or a navigation, not a
 * boot. The base build's tracker records nothing, so in this repository the
 * event goes nowhere; a build that replaces the seam decides whether it is sent.
 */
export function BootTimingReport() {
  const { recordEvent } = useTelemetry()
  const cachedContext = useCachedOrganizationContext()
  const { isAuthenticated } = useAuth()
  // Read at the first render: the event describes the load as it started, so a
  // sign-in that follows must not turn a signed-out boot into a signed-in one.
  const [boot] = useState(() => ({
    signedIn: isAuthenticated,
    overview: landedOnOverview(),
    hidden: document.visibilityState === "hidden",
  }))
  // The tracker and the cache the report reads are whichever are current when
  // it fires, which can be several renders after the one that scheduled it.
  const latest = useRef({ recordEvent, cachedContext })
  useEffect(() => {
    latest.current = { recordEvent, cachedContext }
  })

  useEffect(() => {
    if (reported) return
    let hiddenSince = boot.hidden

    const report = () => {
      if (reported) return
      reported = true
      cleanup()
      const timing = readBootTiming()
      const signedIn = boot.signedIn
      const context = signedIn ? latest.current.cachedContext() : undefined
      const properties: TelemetryProperties = {
        ...timing,
        t_kpi_ms: signedIn && boot.overview ? timing.t_kpi_ms : undefined,
        t_context_ms: signedIn ? timing.t_context_ms : undefined,
        landing_route: boot.overview ? "overview" : "other",
        signed_in: signedIn,
        role_bucket: roleBucket(context),
        hidden_during_boot: hiddenSince,
      }
      latest.current.recordEvent(
        TELEMETRY_EVENTS.DASHBOARD_BOOT_TIMED,
        properties,
      )
    }

    const onVisibilityChange = () => {
      if (document.visibilityState !== "hidden") return
      hiddenSince = true
      report()
    }

    const stopWaiting = whenMarked(BOOT_MARKS.KPIS, report)
    const deadline = window.setTimeout(
      report,
      Math.max(0, BOOT_REPORT_DEADLINE_MS - performance.now()),
    )
    document.addEventListener("visibilitychange", onVisibilityChange)

    function cleanup() {
      stopWaiting()
      window.clearTimeout(deadline)
      document.removeEventListener("visibilitychange", onVisibilityChange)
    }
    return cleanup
  }, [boot])

  return null
}
