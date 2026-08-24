import { useEffect } from "react"

import { useOrganizationContext } from "@/shared/api/hooks"
import { useDeployment } from "@/shared/hooks/useDeployment"
import { useTelemetry } from "@/shared/telemetry/overlayTelemetry"

/**
 * The identity half of the telemetry seam: it names who the recorded events
 * belong to, and renders nothing.
 *
 * Mounted by `AppShell`, so it exists exactly while a session does. The actor
 * and the active organization arrive together from `GET /v1/organizations/me`,
 * which is the only management route that reports anything about the caller,
 * so there is nothing to send until that resolves and nothing to re-send when
 * it does not.
 *
 * **Withheld until consent is stored.** The identity is the one call that hands
 * a tracker something durable about a person, so this refuses to make it on an
 * undecided browser rather than leaving that decision to whichever module the
 * build resolved the seam to. The base build answers `"unknown"` and always
 * will, so nothing is sent here; a replacement that stores a decision is what
 * turns this on, and flipping that decision re-runs this effect. This is
 * `otari-ai/frontend/src/app/AnalyticsIdentity.tsx`'s gate, kept on the base
 * side of the seam.
 *
 * The fields are read out before the effect rather than assembled into an
 * object in the render body, so the effect's dependencies are the values
 * themselves. An object literal would be a fresh reference on every render and
 * would re-identify on each one, and leaning on the React Compiler to memoize
 * it would make an infinite loop depend on whether the compiler bailed out of
 * this component.
 */
export function TelemetryIdentity() {
  const { session_type } = useDeployment()
  const { data } = useOrganizationContext()
  const { consent, identify } = useTelemetry()

  const actorId = data?.organization_member_id
  const organization = data?.organization
  const organizationId = organization?.id
  const organizationName = organization?.name
  const role = data?.role

  useEffect(() => {
    if (consent !== "granted") {
      return
    }
    if (!actorId || !organizationId || !organizationName || !role) {
      // Nothing to say yet, and `identify(null)` is not the thing to say: that
      // is the sign-out signal and `AuthContext` owns it. Sending it here would
      // clear a tracker's actor once on every mount, before the context this is
      // waiting for had a chance to answer.
      return
    }
    identify({
      actorId,
      sessionType: session_type,
      organizationId,
      organizationName,
      role,
    })
  }, [
    consent,
    identify,
    actorId,
    organizationId,
    organizationName,
    role,
    session_type,
  ])

  return null
}
