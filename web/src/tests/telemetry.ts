import { vi } from "vitest"

import type {
  Telemetry,
  TelemetryConsent,
  TelemetryEventName,
  TelemetryIdentity,
  TelemetryProperties,
} from "@/shared/telemetry/types"

/**
 * A recording stand-in for the telemetry seam.
 *
 * The base module is a genuine no-op, so a test asserting that an event fires
 * has nothing to observe unless it replaces the module the way a superset build
 * does. That replacement is what this is: one shared spy, so the six consumers
 * that record something do not each invent their own shape for the tracker and
 * then agree with themselves about it.
 *
 * A test reaches it through the async factory form, because `vi.mock` is hoisted
 * above the imports and a factory referencing one directly would run before it
 * is initialized:
 *
 * ```ts
 * vi.mock("@/shared/telemetry/overlayTelemetry", async () => {
 *   const { telemetrySpy } = await import("@/tests/telemetry")
 *   return { useTelemetry: vi.fn(() => telemetrySpy) }
 * })
 * ```
 *
 * Mocked by the `@/…` specifier, which is the one a build-time override
 * replaces, so the test exercises the same resolution the superset build does.
 * The hook is wrapped in `vi.fn()` rather than handed over as a bare arrow, so
 * it keeps a function identity lint and the React tooling can follow; see
 * "Test mocking" in `.github/skills/frontend-standards/imports-and-modules.md`.
 */

export const recordEvent =
  vi.fn<(event: TelemetryEventName, properties?: TelemetryProperties) => void>()
export const identify = vi.fn<(identity: TelemetryIdentity | null) => void>()

// Granted by default, so a test about an event does not have to say anything
// about consent; the identity test is the one that varies it.
let consent: TelemetryConsent = "granted"

/**
 * One object for the life of the module, because `Telemetry` promises a stable
 * reference: `TelemetryIdentity` holds `identify` in an effect's dependencies,
 * and a fresh object per render would re-identify on every one.
 */
export const telemetrySpy: Telemetry = {
  get consent() {
    return consent
  },
  recordEvent,
  identify,
}

/** Set the stored decision this spy reports, before rendering. */
export function setTelemetryConsent(next: TelemetryConsent): void {
  consent = next
}

/** Forget every recorded call and return consent to its default. */
export function resetTelemetrySpy(): void {
  recordEvent.mockClear()
  identify.mockClear()
  consent = "granted"
}
