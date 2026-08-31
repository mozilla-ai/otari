import { readdirSync, readFileSync } from "node:fs"
import { join } from "node:path"

import { describe, expect, it, vi } from "vitest"

import { TELEMETRY_EVENTS } from "./events"
import {
  createTelemetry,
  loadMixpanelClient,
  useTelemetry,
} from "./overlayTelemetry"
import type { Telemetry } from "./types"

// Resolved from the Vitest root (web/) rather than from import.meta.url, which
// the jsdom environment reports as an http URL. Same reason as
// `src/architecture.test.ts`.
const WEB = process.cwd()

const TELEMETRY_DIR = join(WEB, "src", "shared", "telemetry")

/**
 * Names an analytics, error-reporting or support vendor whose SDK would put
 * telemetry in the bundle. Substrings, because a package can arrive scoped
 * (`@sentry/react`) or suffixed (`mixpanel-browser`).
 *
 * `mixpanel-browser` is the one allowed vendor: it may be declared, and it
 * may only load when a Mixpanel key is present.
 */
const OTHER_VENDORS = [
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

const ALLOWED_MIXPANEL = "mixpanel-browser"

function identity() {
  return {
    actorId: "member-1",
    sessionType: "local_operator" as const,
    organizationId: "org-1",
    organizationName: "Default Organization",
    role: "owner",
  }
}

describe("createTelemetry without a key", () => {
  it("records nothing and does not load the SDK", () => {
    const loadClient = vi.fn<(token: string) => Promise<Telemetry>>()
    const log = vi.fn()
    const telemetry = createTelemetry(undefined, {
      isLocal: false,
      loadClient,
      log,
    })

    expect(
      telemetry.recordEvent(TELEMETRY_EVENTS.LOGIN_SUCCESS, {
        authentication_method: "password",
      }),
    ).toBeUndefined()
    expect(telemetry.identify(identity())).toBeUndefined()
    expect(telemetry.identify(null)).toBeUndefined()
    expect(loadClient).not.toHaveBeenCalled()
    expect(log).not.toHaveBeenCalled()
  })

  it("reports no stored consent decision, which is not consent", () => {
    expect(
      createTelemetry(undefined, { isLocal: false, log: vi.fn() }).consent,
    ).toBe("unknown")
  })

  it("logs Mixpanel not initialized once on a local dashboard, then stops", () => {
    const loadClient = vi.fn<(token: string) => Promise<Telemetry>>()
    const log = vi.fn()
    const telemetry = createTelemetry(undefined, {
      isLocal: true,
      loadClient,
      log,
    })

    telemetry.recordEvent(TELEMETRY_EVENTS.LOGOUT)
    telemetry.recordEvent(TELEMETRY_EVENTS.LOGIN_FAILED)
    telemetry.identify(null)

    expect(log).toHaveBeenCalledTimes(1)
    expect(log).toHaveBeenCalledWith("Mixpanel not initialized")
    expect(loadClient).not.toHaveBeenCalled()
  })

  it("stays silent on a deployed host without a key", () => {
    const log = vi.fn()
    createTelemetry(undefined, { isLocal: false, log, loadClient: vi.fn() })
    expect(log).not.toHaveBeenCalled()
  })
})

describe("createTelemetry with a key", () => {
  it("loads the Mixpanel client and forwards events", async () => {
    const recordEvent = vi.fn<(event: string, properties?: object) => void>()
    const identify = vi.fn()
    const loaded: Telemetry = {
      consent: "granted",
      recordEvent,
      identify,
    }
    const loadClient = vi.fn(async () => loaded)
    const log = vi.fn()

    const telemetry = createTelemetry("mp-test-token", {
      isLocal: true,
      loadClient,
      log,
    })

    expect(telemetry.consent).toBe("granted")
    expect(loadClient).toHaveBeenCalledWith("mp-test-token")
    expect(log).not.toHaveBeenCalled()

    telemetry.recordEvent(TELEMETRY_EVENTS.LOGIN_SUCCESS, {
      authentication_method: "password",
    })
    telemetry.identify(identity())

    await vi.waitFor(() => {
      expect(recordEvent).toHaveBeenCalledWith("Login Success", {
        authentication_method: "password",
      })
      expect(identify).toHaveBeenCalledWith(identity())
    })
  })

  it("does not load Mixpanel when the key is absent even if loadClient is supplied", () => {
    const loadClient = vi.fn<(token: string) => Promise<Telemetry>>()
    createTelemetry(undefined, { isLocal: false, loadClient, log: vi.fn() })
    expect(loadClient).not.toHaveBeenCalled()
  })

  it("discards the in-flight queue and later calls after load failure", async () => {
    let rejectLoad!: (reason: unknown) => void
    const loadClient = vi.fn(
      () =>
        new Promise<Telemetry>((_, reject) => {
          rejectLoad = reject
        }),
    )

    const telemetry = createTelemetry("mp-test-token", {
      isLocal: false,
      loadClient,
      log: vi.fn(),
    })

    telemetry.recordEvent(TELEMETRY_EVENTS.LOGIN_SUCCESS, {
      authentication_method: "password",
    })
    telemetry.identify(identity())

    const load = loadClient.mock.results[0]?.value as Promise<Telemetry>
    rejectLoad(new Error("chunk failed"))
    await expect(load).rejects.toThrow("chunk failed")

    expect(() => {
      telemetry.recordEvent(TELEMETRY_EVENTS.LOGOUT)
      telemetry.identify(null)
    }).not.toThrow()
    expect(loadClient).toHaveBeenCalledTimes(1)
  })
})

describe("the base telemetry seam", () => {
  it("records nothing when this process has no Mixpanel key", () => {
    const telemetry = useTelemetry()

    expect(
      telemetry.recordEvent(TELEMETRY_EVENTS.LOGIN_SUCCESS, {
        authentication_method: "password",
      }),
    ).toBeUndefined()
    expect(telemetry.identify(identity())).toBeUndefined()
    expect(telemetry.identify(null)).toBeUndefined()
  })

  it("reports no stored consent decision without a key", () => {
    expect(useTelemetry().consent).toBe("unknown")
  })

  it("answers with one stable object, as a replacement must", () => {
    expect(useTelemetry()).toBe(useTelemetry())
  })

  it("does not statically import mixpanel-browser", () => {
    const source = readFileSync(
      join(TELEMETRY_DIR, "overlayTelemetry.ts"),
      "utf8",
    )

    expect(source).not.toMatch(/from\s+["']mixpanel-browser["']/)
    expect(source).not.toMatch(/import\s+["']mixpanel-browser["']/)
  })

  it("loads the Mixpanel client only through a dynamic import", () => {
    const source = readFileSync(
      join(TELEMETRY_DIR, "overlayTelemetry.ts"),
      "utf8",
    )

    expect(source).toMatch(
      /import\(\s*["']@\/shared\/telemetry\/mixpanelClient["']\s*\)/,
    )
    expect(source).not.toMatch(
      /from\s+["']@\/shared\/telemetry\/mixpanelClient["']/,
    )
  })
})

describe("the base build's dependencies", () => {
  const manifest = JSON.parse(
    readFileSync(join(WEB, "package.json"), "utf8"),
  ) as {
    dependencies?: Record<string, string>
    devDependencies?: Record<string, string>
  }

  it("may declare mixpanel-browser and no other telemetry vendor", () => {
    const declared = [
      ...Object.keys(manifest.dependencies ?? {}),
      ...Object.keys(manifest.devDependencies ?? {}),
    ]

    expect(declared).toContain(ALLOWED_MIXPANEL)
    expect(
      declared.filter(
        (name) =>
          name !== ALLOWED_MIXPANEL &&
          OTHER_VENDORS.some((vendor) => name.toLowerCase().includes(vendor)),
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

describe("always-loaded telemetry modules", () => {
  it("never statically import mixpanel-browser", () => {
    const alwaysLoaded = [
      "overlayTelemetry.ts",
      "mixpanelToken.ts",
      "events.ts",
      "types.ts",
      "errorCode.ts",
    ]
    for (const name of alwaysLoaded) {
      const source = readFileSync(join(TELEMETRY_DIR, name), "utf8")
      expect(source, name).not.toMatch(/from\s+["']mixpanel-browser["']/)
      expect(source, name).not.toMatch(/import\s+["']mixpanel-browser["']/)
    }
  })
})

describe("loadMixpanelClient", () => {
  it("is the dynamic import used only after a key is present", () => {
    // The function exists so Vite puts mixpanel-browser in an async chunk.
    // Calling it without a key is what the gate above forbids; this only
    // asserts the export stays a function the queued tracker can call.
    expect(typeof loadMixpanelClient).toBe("function")
  })
})
