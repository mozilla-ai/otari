/**
 * Mixpanel-backed tracker. Loaded only through a dynamic import, and only
 * after `readMixpanelToken()` has returned a value, so a build with no key
 * never fetches this module or `mixpanel-browser`.
 *
 * The SDK's ESM build exports only a default (`export { mixpanel as default }`).
 * Named imports type-check against its `.d.ts` and are `undefined` at runtime.
 */

import mixpanel from "mixpanel-browser"

import type {
  Telemetry,
  TelemetryIdentity,
  TelemetryProperties,
} from "@/shared/telemetry/types"

/**
 * Init options that keep Mixpanel from inventing events of its own.
 *
 * Autocapture, pageviews, and session replay would record names that are not
 * in `TELEMETRY_EVENTS`. The catalog is the whole of what this dashboard
 * sends, so those features stay off.
 */
const MIXPANEL_INIT = {
  autocapture: false,
  track_pageview: false,
  record_sessions_percent: 0,
  verbose: false,
  debug: false,
} as const

function toDict(
  properties: TelemetryProperties | undefined,
): Record<string, string | number | boolean | readonly string[]> | undefined {
  if (properties === undefined) {
    return undefined
  }
  const dict: Record<string, string | number | boolean | readonly string[]> = {}
  for (const [key, value] of Object.entries(properties)) {
    if (value !== undefined) {
      dict[key] = value
    }
  }
  return dict
}

function peopleProperties(identity: TelemetryIdentity): Record<string, string> {
  return {
    session_type: identity.sessionType,
    organization_id: identity.organizationId,
    organization_name: identity.organizationName,
    role: identity.role,
  }
}

export function createMixpanelTelemetry(token: string): Telemetry {
  mixpanel.init(token, MIXPANEL_INIT)
  // `actorId` is the active organization's membership id, so a switch
  // changes it. mixpanel-browser 2.82.1 `identify` always tracks `$identify`
  // with `$anon_distinct_id` set to the previous `distinct_id`, and the
  // server merges those two. That is the anonymous-to-first-user link we
  // want on the first identify (and after `reset`). It is not what we want
  // when the previous id is another membership: that would cluster two
  // actors. Remember the last identified membership and `reset` before a
  // different one so the previous id is a fresh `$device:` id, not a peer.
  let identifiedActor: string | undefined

  return {
    // A Mixpanel key is the deployment opt-in. This build has no consent UI,
    // so the key is the gate: present means granted, absent never reaches here.
    consent: "granted",
    recordEvent: (event, properties) => {
      mixpanel.track(event, toDict(properties))
    },
    identify: (identity) => {
      if (identity === null) {
        identifiedActor = undefined
        mixpanel.reset()
        return
      }
      if (
        identifiedActor !== undefined &&
        identifiedActor !== identity.actorId
      ) {
        mixpanel.reset()
      }
      identifiedActor = identity.actorId
      mixpanel.identify(identity.actorId)
      mixpanel.people.set(peopleProperties(identity))
    },
  }
}
