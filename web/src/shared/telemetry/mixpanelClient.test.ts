import { beforeEach, describe, expect, it, vi } from "vitest"

import { TELEMETRY_EVENTS } from "./events"
import type { TelemetryIdentity } from "./types"

const init = vi.fn()
const track = vi.fn()
const identify = vi.fn()
const reset = vi.fn()
const peopleSet = vi.fn()

vi.mock("mixpanel-browser", () => ({
  default: {
    init,
    track,
    identify,
    reset,
    people: { set: peopleSet },
  },
}))

function member(actorId: string): TelemetryIdentity {
  return {
    actorId,
    sessionType: "local_operator",
    organizationId: "org-1",
    organizationName: "Default Organization",
    role: "owner",
  }
}

describe("createMixpanelTelemetry", () => {
  beforeEach(() => {
    init.mockClear()
    track.mockClear()
    identify.mockClear()
    reset.mockClear()
    peopleSet.mockClear()
  })

  it("inits without autocapture or invented pageviews", async () => {
    const { createMixpanelTelemetry } = await import("./mixpanelClient")
    createMixpanelTelemetry("mp-test-token")

    expect(init).toHaveBeenCalledWith(
      "mp-test-token",
      expect.objectContaining({
        autocapture: false,
        track_pageview: false,
        record_sessions_percent: 0,
      }),
    )
  })

  it("forwards recordEvent as mixpanel.track", async () => {
    const { createMixpanelTelemetry } = await import("./mixpanelClient")
    const telemetry = createMixpanelTelemetry("mp-test-token")

    telemetry.recordEvent(TELEMETRY_EVENTS.LOGIN_SUCCESS, {
      authentication_method: "password",
    })

    expect(track).toHaveBeenCalledWith("Login Success", {
      authentication_method: "password",
    })
  })

  it("identifies the actor and sets people properties", async () => {
    const { createMixpanelTelemetry } = await import("./mixpanelClient")
    const telemetry = createMixpanelTelemetry("mp-test-token")

    telemetry.identify(member("member-1"))

    expect(identify).toHaveBeenCalledWith("member-1")
    expect(peopleSet).toHaveBeenCalledWith({
      session_type: "local_operator",
      organization_id: "org-1",
      organization_name: "Default Organization",
      role: "owner",
    })
  })

  it("does not reset before the first identify", async () => {
    const { createMixpanelTelemetry } = await import("./mixpanelClient")
    const telemetry = createMixpanelTelemetry("mp-test-token")

    telemetry.identify(member("member-1"))

    expect(reset).not.toHaveBeenCalled()
    expect(identify).toHaveBeenCalledWith("member-1")
  })

  it("does not reset when identifying the same actor again", async () => {
    const { createMixpanelTelemetry } = await import("./mixpanelClient")
    const telemetry = createMixpanelTelemetry("mp-test-token")

    telemetry.identify(member("member-1"))
    reset.mockClear()
    identify.mockClear()

    telemetry.identify(member("member-1"))

    expect(reset).not.toHaveBeenCalled()
    expect(identify).toHaveBeenCalledWith("member-1")
  })

  it("resets before identifying a different already-identified actor", async () => {
    const { createMixpanelTelemetry } = await import("./mixpanelClient")
    const telemetry = createMixpanelTelemetry("mp-test-token")

    telemetry.identify(member("member-1"))
    reset.mockClear()
    identify.mockClear()

    telemetry.identify(member("member-2"))

    expect(reset).toHaveBeenCalledOnce()
    expect(identify).toHaveBeenCalledWith("member-2")
    expect(reset.mock.invocationCallOrder[0]).toBeLessThan(
      identify.mock.invocationCallOrder[0],
    )
  })

  it("does not reset on the first identify after a logout reset", async () => {
    const { createMixpanelTelemetry } = await import("./mixpanelClient")
    const telemetry = createMixpanelTelemetry("mp-test-token")

    telemetry.identify(member("member-1"))
    telemetry.identify(null)
    reset.mockClear()
    identify.mockClear()

    telemetry.identify(member("member-2"))

    expect(reset).not.toHaveBeenCalled()
    expect(identify).toHaveBeenCalledWith("member-2")
  })

  it("resets Mixpanel when identify is handed null", async () => {
    const { createMixpanelTelemetry } = await import("./mixpanelClient")
    const telemetry = createMixpanelTelemetry("mp-test-token")

    telemetry.identify(null)

    expect(reset).toHaveBeenCalledOnce()
    expect(identify).not.toHaveBeenCalled()
  })
})
