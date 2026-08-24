/**
 * The contract the telemetry seam is written against.
 *
 * Base-owned and not swapped, for the reason `events.ts` is not: a replacement
 * module implements this shape, so both halves of the swap have to read one
 * declaration of it.
 */

import type { DeploymentBootstrap } from "@/client"

import type { TELEMETRY_EVENTS } from "./events"

/** One of the names `events.ts` declares, and nothing else. */
export type TelemetryEventName =
  (typeof TELEMETRY_EVENTS)[keyof typeof TELEMETRY_EVENTS]

/**
 * What a call site may attach to an event.
 *
 * Bounded, non-user-controlled values only: a property name is a schema an
 * analytics backend indexes, and a value taken from a text box is unbounded
 * cardinality at best and someone's password at worst. Nothing here carries a
 * credential, an address, or a server error message; a failure is described by
 * a code and a status, which is what `Login.tsx` and `SignupPage.tsx` send.
 */
export type TelemetryProperties = Readonly<
  Record<string, string | number | boolean | readonly string[] | undefined>
>

/**
 * Whether a decision to record telemetry is stored for this browser.
 *
 * Part of the seam rather than an assumption inside it. Consent is what gates
 * every call in the platform's working implementation, and a seam that did not
 * carry it would leave the base able to hand an identity to a tracker that had
 * no permission to hold one. `"unknown"` is "nothing decided", which is what a
 * build with no consent UI answers and is not a synonym for `"granted"`.
 */
export type TelemetryConsent = "granted" | "denied" | "unknown"

/**
 * Who the events being recorded belong to.
 *
 * The active organization travels with the actor rather than as a later call,
 * because the two arrive together: `GET /v1/organizations/me` is the one
 * management route that reports anything about the caller, and it answers with
 * the membership and its organization in a single response. A build that could
 * name the user but not the tenant would report a funnel it cannot segment.
 */
export interface TelemetryIdentity {
  /**
   * The caller's membership in the active organization
   * (`organization_member_id`), which is the only per-caller id this deployment
   * publishes. Deliberately not an email or a name: this is an id to group by,
   * not a person to describe.
   */
  actorId: string
  /** Which kind of session this deployment issues, from `GET /v1/bootstrap`. */
  sessionType: DeploymentBootstrap["session_type"]
  organizationId: string
  organizationName: string
  /** The actor's role in that organization ("owner", "admin", "member"). */
  role: string
}

/**
 * The tracker a call site reaches, whichever build supplied it.
 *
 * **Which half of the consent check is whose.** The base checks consent for
 * `identify` and not for `recordEvent`, and the split is deliberate rather than
 * an oversight: an identity is the one call that hands a tracker something
 * durable about a person, so `TelemetryIdentity` withholds it here, while every
 * `recordEvent` call site fires unconditionally and a replacement owns that
 * gate. That is where the platform's own gate lives (`trackEvent` in
 * `otari-ai/frontend/src/shared/helpers/mixpanel.ts` checks `hasConsent`
 * itself), and it is why `consent` is on this interface at all: an
 * implementation cannot gate what it is never told.
 */
export interface Telemetry {
  /** The stored decision, for a caller that must not assume one. */
  readonly consent: TelemetryConsent
  /**
   * Record one event. **Not consent-gated by the base**: a replacement must
   * check `consent` itself before sending anything anywhere.
   */
  readonly recordEvent: (
    event: TelemetryEventName,
    properties?: TelemetryProperties,
  ) => void
  /**
   * Name the actor these events belong to, or `null` to forget the last one.
   *
   * `null` is the sign-out signal: it is what keeps a tracker from attributing
   * the next session in this tab to the identity that just left.
   */
  readonly identify: (identity: TelemetryIdentity | null) => void
}
