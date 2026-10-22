import { act, render, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import {
  BOOT_REPORT_DEADLINE_MS,
  BootTimingReport,
  resetBootTimingReport,
} from "@/app/BootTimingReport"
import { Provider } from "@/app/provider"
import { useOrganizationContext } from "@/shared/api/organizations"
import { BOOT_MARKS, markOnce } from "@/shared/telemetry/bootTiming"
import { TELEMETRY_EVENTS } from "@/shared/telemetry/events"
import { organizationContext } from "@/tests/fixtures"
import { recordEvent, resetTelemetrySpy } from "@/tests/telemetry"

vi.mock("@/shared/telemetry/overlayTelemetry", async () => {
  const { telemetrySpy } = await import("@/tests/telemetry")
  return { useTelemetry: vi.fn(() => telemetrySpy) }
})

const SESSION_KEY = "otari.dashboard.hasSession"

function setVisibility(state: DocumentVisibilityState) {
  Object.defineProperty(document, "visibilityState", {
    configurable: true,
    get: () => state,
  })
}

function renderReport() {
  return render(
    <Provider>
      <BootTimingReport />
    </Provider>,
  )
}

function ContextReader() {
  useOrganizationContext()
  return null
}

function reported() {
  return recordEvent.mock.calls.filter(
    ([event]) => event === TELEMETRY_EVENTS.DASHBOARD_BOOT_TIMED,
  )
}

describe("BootTimingReport", () => {
  beforeEach(() => {
    performance.clearMarks()
    resetTelemetrySpy()
    resetBootTimingReport()
    window.localStorage.setItem(SESSION_KEY, "1")
    window.location.hash = "#/"
    setVisibility("visible")
  })

  afterEach(() => {
    window.localStorage.clear()
    Reflect.deleteProperty(document, "visibilityState")
    vi.useRealTimers()
    vi.restoreAllMocks()
  })

  it("reports once Overview's figure is marked, and only once", async () => {
    performance.mark(BOOT_MARKS.BOOTSTRAP, { startTime: 412 })
    performance.mark(BOOT_MARKS.SHELL, { startTime: 655 })
    renderReport()
    expect(reported()).toHaveLength(0)

    await act(async () => {
      markOnce(BOOT_MARKS.CONTEXT)
      markOnce(BOOT_MARKS.KPIS)
    })

    expect(reported()).toHaveLength(1)
    expect(reported()[0][1]).toMatchObject({
      t_bootstrap_ms: 410,
      t_shell_ms: 660,
      t_kpi_ms: expect.any(Number),
      t_context_ms: expect.any(Number),
      landing_route: "overview",
      signed_in: true,
      kept_bootstrap: "off",
      kept_context: "off",
      remounted: false,
      hidden_during_boot: false,
    })

    act(() => {
      setVisibility("hidden")
      document.dispatchEvent(new Event("visibilitychange"))
    })
    expect(reported()).toHaveLength(1)
  })

  it("reports once across a remount, which is the same load", async () => {
    const first = renderReport()
    first.unmount()
    renderReport()
    await act(async () => {
      markOnce(BOOT_MARKS.KPIS)
    })
    expect(reported()).toHaveLength(1)
  })

  it("reports what it has when the page is hidden", () => {
    renderReport()
    act(() => {
      setVisibility("hidden")
      document.dispatchEvent(new Event("visibilitychange"))
    })
    expect(reported()).toHaveLength(1)
    expect(reported()[0][1]).toMatchObject({
      hidden_during_boot: true,
      t_kpi_ms: undefined,
    })
  })

  it("says a load that started hidden was hidden during boot", () => {
    vi.useFakeTimers()
    setVisibility("hidden")
    renderReport()
    act(() => {
      vi.advanceTimersByTime(BOOT_REPORT_DEADLINE_MS)
    })
    expect(reported()).toHaveLength(1)
    expect(reported()[0][1]).toMatchObject({ hidden_during_boot: true })
  })

  it("reports at the deadline when Overview never shows a figure", () => {
    vi.useFakeTimers()
    renderReport()
    act(() => {
      vi.advanceTimersByTime(BOOT_REPORT_DEADLINE_MS - performance.now() - 1)
    })
    expect(reported()).toHaveLength(0)
    act(() => {
      vi.advanceTimersByTime(BOOT_REPORT_DEADLINE_MS)
    })
    expect(reported()).toHaveLength(1)
    expect(reported()[0][1]).toMatchObject({ hidden_during_boot: false })
  })

  it("sends no figure time for a load that did not open on Overview", async () => {
    window.location.hash = "#/keys"
    renderReport()
    await act(async () => {
      markOnce(BOOT_MARKS.KPIS)
    })
    expect(reported()[0][1]).toMatchObject({
      landing_route: "other",
      t_kpi_ms: undefined,
    })
  })

  it("keeps a signed-out boot signed out, and sends no figure or context", async () => {
    window.localStorage.removeItem(SESSION_KEY)
    renderReport()
    await act(async () => {
      markOnce(BOOT_MARKS.CONTEXT)
      markOnce(BOOT_MARKS.KPIS)
    })
    expect(reported()[0][1]).toMatchObject({
      signed_in: false,
      t_kpi_ms: undefined,
      t_context_ms: undefined,
      role_bucket: undefined,
    })
  })

  it.each([
    ["owner", false, "admin"],
    ["member", false, "member"],
    ["member", true, "operator"],
  ])("buckets a %s (operator: %s) as %s", async (role, operator, bucket) => {
    vi.spyOn(globalThis, "fetch").mockImplementation(async () =>
      Response.json(
        organizationContext({ role, deployment_operator: operator }),
      ),
    )
    render(
      <Provider>
        <ContextReader />
        <BootTimingReport />
      </Provider>,
    )
    await waitFor(() => {
      expect(performance.getEntriesByName(BOOT_MARKS.CONTEXT)).toHaveLength(1)
    })
    await act(async () => {
      markOnce(BOOT_MARKS.KPIS)
    })
    expect(reported()[0][1]).toMatchObject({ role_bucket: bucket })
  })
})
