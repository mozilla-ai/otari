import { beforeEach, describe, expect, it, vi } from "vitest"

import { TELEMETRY_EVENTS } from "./events"

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

    telemetry.identify({
      actorId: "member-1",
      sessionType: "local_operator",
      organizationId: "org-1",
      organizationName: "Default Organization",
      role: "owner",
    })

    expect(identify).toHaveBeenCalledWith("member-1")
    expect(peopleSet).toHaveBeenCalledWith({
      session_type: "local_operator",
      organization_id: "org-1",
      organization_name: "Default Organization",
      role: "owner",
    })
  })

  it("resets Mixpanel when identify is handed null", async () => {
    const { createMixpanelTelemetry } = await import("./mixpanelClient")
    const telemetry = createMixpanelTelemetry("mp-test-token")

    telemetry.identify(null)

    expect(reset).toHaveBeenCalledOnce()
    expect(identify).not.toHaveBeenCalled()
  })
})
