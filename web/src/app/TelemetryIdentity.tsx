import { useEffect, useRef } from "react"

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
 * side of the seam. Withdrawing a decision is the other half of it and forgets
 * an identity already sent, which is the case a gate written only as an early
 * return would miss.
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
  // Whether this component has handed the tracker an actor. Revoking consent
  // has to take it back, and there is nothing else that knows one was sent.
  const identified = useRef(false)

  const actorId = data?.organization_member_id
  const organization = data?.organization
  const organizationId = organization?.id
  const organizationName = organization?.name
  const role = data?.role

  useEffect(() => {
    if (consent !== "granted") {
      // Revocation, not just absence: a decision withdrawn after an identity
      // was sent has to forget it, or the tracker keeps holding an actor it no
      // longer has permission to hold. `identify(null)` means "forget the last
      // actor", which is the same thing a sign-out needs and is why one call
      // serves both.
      if (identified.current) {
        identified.current = false
        identify(null)
      }
      return
    }
    if (!actorId || !organizationId || !organizationName || !role) {
      // Nothing to say yet, and `identify(null)` is not the thing to say: that
      // is the sign-out signal and `AuthContext` owns it. Sending it here would
      // clear a tracker's actor once on every mount, before the context this is
      // waiting for had a chance to answer.
      return
    }
    identified.current = true
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
