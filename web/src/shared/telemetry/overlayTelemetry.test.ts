import { readdirSync, readFileSync } from "node:fs"
import { join } from "node:path"

import { describe, expect, it } from "vitest"

import { TELEMETRY_EVENTS } from "./events"
import { useTelemetry } from "./overlayTelemetry"

// Resolved from the Vitest root (web/) rather than from import.meta.url, which
// the jsdom environment reports as an http URL. Same reason as
// `src/architecture.test.ts`.
const WEB = process.cwd()

/**
 * Names an analytics, error-reporting or support vendor whose SDK would put
 * telemetry in the bundle. Substrings, because a package can arrive scoped
 * (`@sentry/react`) or suffixed (`mixpanel-browser`).
 */
const VENDORS = [
  "mixpanel",
  "matomo",
  "piwik",
  "intercom",
  "sentry",
  "posthog",
  "amplitude",
  "segment",
  "analytics",
  "heap",
  "fullstory",
  "logrocket",
  "datadog",
  "hotjar",
  "gtag",
  "google-analytics",
]

describe("the base telemetry seam", () => {
  it("records nothing", () => {
    const telemetry = useTelemetry()

    // A no-op has no observable effect to assert, so what is asserted is that
    // calling it neither throws nor answers with anything: an implementation
    // that started returning a queue, a client, or a promise would be doing
    // something, and this build is meant to do nothing.
    expect(
      telemetry.recordEvent(TELEMETRY_EVENTS.LOGIN_SUCCESS, {
        authentication_method: "password",
      }),
    ).toBeUndefined()
    expect(
      telemetry.identify({
        actorId: "member-1",
        sessionType: "local_operator",
        organizationId: "org-1",
        organizationName: "Default Organization",
        role: "owner",
      }),
    ).toBeUndefined()
    expect(telemetry.identify(null)).toBeUndefined()
  })

  it("reports no stored consent decision, which is not consent", () => {
    // "unknown" rather than "granted": a build with no consent UI has asked
    // nobody. `TelemetryIdentity` is what turns that into a withheld identity.
    expect(useTelemetry().consent).toBe("unknown")
  })

  it("answers with one stable object, as a replacement must", () => {
    // `TelemetryIdentity` holds `identify` and `consent` in an effect's
    // dependencies, so a tracker handing out a fresh object each call would
    // re-identify forever. The base default is the reference for that contract.
    expect(useTelemetry()).toBe(useTelemetry())
  })

  it("imports nothing but its own contract", () => {
    const source = readFileSync(
      join(WEB, "src", "shared", "telemetry", "overlayTelemetry.ts"),
      "utf8",
    )
    const imports = [...source.matchAll(/from\s+"([^"]+)"/g)].map(
      (match) => match[1],
    )

    // The seam is what an OSS build ships, so it may not reach a vendor SDK
    // even lazily. The contract is the whole of it; `events.ts` is not needed
    // here because this module names no event. Asserted in its `@/…` form
    // rather than as `./types`, which is the point of the note on that import:
    // a replacement lives in the overlay's tree, where a relative sibling path
    // would resolve somewhere else entirely.
    expect(imports).toEqual(["@/shared/telemetry/types"])
  })
})

describe("the base build's dependencies", () => {
  const manifest = JSON.parse(
    readFileSync(join(WEB, "package.json"), "utf8"),
  ) as {
    dependencies?: Record<string, string>
    devDependencies?: Record<string, string>
  }

  it("declare no telemetry vendor", () => {
    // The other half of "the base emits nothing": a dashboard that shipped an
    // analytics SDK and merely declined to call it would still hand every
    // operator's browser a script from a third party. devDependencies are
    // checked too, because Vite bundles from the import graph and not from
    // which section of the manifest a package was declared in.
    const declared = [
      ...Object.keys(manifest.dependencies ?? {}),
      ...Object.keys(manifest.devDependencies ?? {}),
    ]

    expect(
      declared.filter((name) =>
        VENDORS.some((vendor) => name.toLowerCase().includes(vendor)),
      ),
    ).toEqual([])
  })
})

describe("the event catalog", () => {
  it("carries the twelve names the platform already records under", () => {
    // Pinned as strings, not just as keys: a superset build reports into the
    // analytics workspace otari-ai already reports into, so renaming one here
    // splits its funnel in two rather than failing anything.
    expect(TELEMETRY_EVENTS).toEqual({
      SIGNUP_STARTED: "Signup Started",
      SIGNUP_SUCCESS: "Signup Success",
      SIGNUP_FAILED: "Signup Failed",
      LOGIN_SUCCESS: "Login Success",
      LOGIN_FAILED: "Login Failed",
      LOGOUT: "Logout",
      EMAIL_VERIFICATION_SUCCESS: "Email Verification Success",
      EMAIL_VERIFICATION_FAILED: "Email Verification Failed",
      RESEND_VERIFICATION_CLICKED: "Resend Verification Clicked",
      FORM_VALIDATION_FAILED: "Form Validation Failed",
      TAB_CHANGED: "Tab Changed",
      CHECKOUT_CANCELLED: "Checkout Cancelled",
    })
  })
})

/**
 * Every non-test source file under `web/src`, as a path/contents pair.
 *
 * `src/tests/` is excluded along with the `.test.` files: it is the harness
 * directory, so a spy there naming an event would otherwise satisfy "fired from
 * a base source file" without any base code firing it.
 */
function baseSources(directory: string): { path: string; source: string }[] {
  const found: { path: string; source: string }[] = []
  for (const entry of readdirSync(directory, { withFileTypes: true })) {
    // `architecture.test.ts` plants and deletes files in these while this runs,
    // and Vitest runs the two together, so reading one is a race.
    if (entry.name === "__boundary_probe__" || entry.name === "tests") {
      continue
    }
    const path = join(directory, entry.name)
    if (entry.isDirectory()) {
      found.push(...baseSources(path))
      continue
    }
    if (!/\.tsx?$/.test(entry.name) || /\.test\.tsx?$/.test(entry.name)) {
      continue
    }
    found.push({ path, source: readFileSync(path, "utf8") })
  }
  return found
}

describe("the wiring behind the catalog", () => {
  const sources = baseSources(join(WEB, "src"))

  /**
   * Named here and fired by nobody, deliberately, and the only such name.
   *
   * Listed rather than left silently outside the assertion below: an unfired
   * name reads as dead, and the next person to tidy one away would be breaking
   * a contract rather than removing a leftover. `events.ts` says why this one
   * is here.
   */
  const NOT_FIRED_HERE = ["CHECKOUT_CANCELLED"]

  it.each(
    Object.keys(TELEMETRY_EVENTS).filter(
      (name) => !NOT_FIRED_HERE.includes(name),
    ),
  )("records %s from a base source file", (name) => {
    // A name nothing fires is a step of the funnel a superset build would
    // silently lose: the base default records nothing either way, so no test
    // that mocks the seam fails when a call site quietly goes away.
    expect(
      sources.some(({ source }) => source.includes(`TELEMETRY_EVENTS.${name}`)),
    ).toBe(true)
  })

  it.each(NOT_FIRED_HERE)("leaves %s to the overlay that owns it", (name) => {
    expect(
      sources.filter(({ source }) =>
        source.includes(`TELEMETRY_EVENTS.${name}`),
      ),
    ).toEqual([])
  })
})
