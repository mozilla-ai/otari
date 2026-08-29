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

  return {
    // A Mixpanel key is the deployment opt-in. This build has no consent UI,
    // so the key is the gate: present means granted, absent never reaches here.
    consent: "granted",
    recordEvent: (event, properties) => {
      mixpanel.track(event, toDict(properties))
    },
    identify: (identity) => {
      if (identity === null) {
        mixpanel.reset()
        return
      }
      mixpanel.identify(identity.actorId)
      mixpanel.people.set(peopleProperties(identity))
    },
  }
}
