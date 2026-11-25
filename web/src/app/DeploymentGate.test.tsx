import { act, render, screen, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { announce } from "@/app/announce"
import { type BootLoader, createBootLoader, firstSettled } from "@/app/boot"
import { DeploymentGate } from "@/app/DeploymentGate"
import { Provider } from "@/app/provider"
import { ApiError, RedirectedError } from "@/shared/api/client"
import type { WireBootstrap } from "@/shared/helpers/bootstrap"
import { bootstrap } from "@/tests/fixtures"

vi.mock("@/app/announce", () => ({ announce: vi.fn() }))

// The seam this build answers "same origin" through, mutable so a case can
// stand in for a build whose API is on another host.
const policy = vi.hoisted(() => ({
  origin: "",
  credentials: "same-origin" as RequestCredentials,
}))
vi.mock("@/shared/api/overlayRequestPolicy", () => ({
  prepareRequests: async () => {},
  requestPolicy: () => policy,
}))

const DOWN = new ApiError(0, "Network error: could not reach the gateway.")
const UNREACHABLE = "Can’t reach the gateway"

let loader: BootLoader | undefined

type Load = (signal: AbortSignal) => Promise<WireBootstrap>

// Mounted the way `main.tsx` mounts it: after the first attempt settles.
async function mountGate(
  load: Load,
  prepare: () => Promise<void> = async () => {},
) {
  loader = createBootLoader({ prepare, load, random: () => 0 })
  await act(async () => {
    await firstSettled(loader as BootLoader)
  })
  return render(
    <Provider>
      <DeploymentGate loader={loader} />
    </Provider>,
  )
}

/** A read that stays open until the test settles it. */
function pending() {
  let settle: {
    answer: (value: WireBootstrap) => void
    fail: (error: unknown) => void
  } = { answer: () => {}, fail: () => {} }
  const promise = new Promise<WireBootstrap>((resolve, reject) => {
    settle = { answer: resolve, fail: reject }
  })
  return { promise, settle: () => settle }
}

function retryButton() {
  return screen.getByRole("button", { name: /Retry now|Retrying…/ })
}

function lastTry() {
  const row = screen.getByText("Last try")
  return row.nextElementSibling?.textContent ?? ""
}

describe("DeploymentGate", () => {
  beforeEach(() => {
    vi.useFakeTimers({ shouldAdvanceTime: true })
    vi.mocked(announce).mockClear()
  })

  afterEach(() => {
    loader?.dispose()
    loader = undefined
    vi.useRealTimers()
    vi.restoreAllMocks()
    window.localStorage.clear()
    window.history.replaceState(null, "", "/")
    document.title = ""
    policy.origin = ""
  })

  it("says the gateway is unreachable instead of guessing a deployment", async () => {
    await mountGate(async () => {
      throw DOWN
    })

    expect(
      screen.getByRole("heading", { level: 1, name: UNREACHABLE }),
    ).toBeInTheDocument()
    expect(screen.getByText("GATEWAY UNREACHABLE")).toBeInTheDocument()
    expect(screen.getByText(window.location.host)).toBeInTheDocument()
    expect(lastTry()).toMatch(/^\d\d:\d\d:\d\d · no response$/)
    expect(retryButton()).toHaveTextContent("Retry now")
    expect(document.title).toBe("Can’t reach the gateway · Otari")
    expect(
      screen.queryByRole("heading", { name: "Sign in to Otari" }),
    ).toBeNull()
    expect(screen.queryByRole("navigation")).toBeNull()
  })

  // The table the screen is built from: what selects each state, its words,
  // its action, and whether the card keeps sweeping.
  it.each<{
    name: string
    error: unknown
    offline?: boolean
    heading: string
    mark: string
    reason: string | null
    action: string
    title: string
    sweeps: boolean
  }>([
    {
      name: "a 502",
      error: new ApiError(502, "Bad Gateway"),
      heading: UNREACHABLE,
      mark: "GATEWAY UNREACHABLE",
      reason: "HTTP 502",
      action: "Retry now",
      title: "Can’t reach the gateway · Otari",
      sweeps: true,
    },
    {
      name: "an offline device",
      error: DOWN,
      offline: true,
      heading: UNREACHABLE,
      mark: "GATEWAY UNREACHABLE",
      reason: "offline",
      action: "Retry now",
      title: "Can’t reach the gateway · Otari",
      sweeps: true,
    },
    {
      name: "a 401",
      error: new ApiError(401, "Unauthorized"),
      heading: "Sign in to continue",
      mark: "SIGN-IN REQUIRED",
      reason: "HTTP 401",
      action: "Reload",
      title: "Sign-in required · Otari",
      sweeps: false,
    },
    {
      name: "a redirect",
      error: new RedirectedError(),
      heading: "Sign in to continue",
      mark: "SIGN-IN REQUIRED",
      reason: "sent to sign-in",
      action: "Reload",
      title: "Sign-in required · Otari",
      sweeps: false,
    },
    {
      name: "a 403",
      error: new ApiError(403, "Forbidden"),
      heading: "This browser isn’t allowed in",
      mark: "ACCESS REFUSED",
      reason: "HTTP 403",
      action: "Reload",
      title: "Access refused · Otari",
      sweeps: false,
    },
    {
      name: "a 404",
      error: new ApiError(404, "Not Found"),
      heading: "This address didn’t answer like a gateway",
      mark: "UNEXPECTED ANSWER",
      reason: "HTTP 404",
      action: "Retry",
      title: "Unexpected answer · Otari",
      sweeps: false,
    },
  ])("reports $name as its own state", async (row) => {
    if (row.offline) {
      vi.spyOn(navigator, "onLine", "get").mockReturnValue(false)
    }
    const { container } = await mountGate(async () => {
      throw row.error
    })

    expect(
      screen.getByRole("heading", { level: 1, name: row.heading }),
    ).toBeInTheDocument()
    expect(screen.getByText(row.mark)).toBeInTheDocument()
    expect(lastTry()).toMatch(new RegExp(`· ${row.reason}$`))
    expect(screen.getByRole("button", { name: row.action })).toBeInTheDocument()
    expect(document.title).toBe(row.title)
    expect(
      container
        .querySelector("[data-boot-card]")
        ?.classList.contains("otari-scan-border"),
    ).toBe(row.sweeps)
    // The gateway's or a proxy's own words never reach the screen.
    expect(container).not.toHaveTextContent(
      /Unauthorized|Forbidden|Bad Gateway/,
    )
  })

  it("says no gateway was chosen, with nothing to diagnose, when the deployment could not be chosen", async () => {
    const load = vi.fn<Load>()
    await mountGate(load, async () => {
      throw new Error("no deployment selected")
    })

    expect(
      screen.getByRole("heading", {
        level: 1,
        name: "Couldn’t choose a gateway",
      }),
    ).toBeInTheDocument()
    expect(screen.queryByText("Last try")).toBeNull()
    expect(screen.getByRole("button", { name: "Retry" })).toBeInTheDocument()
    expect(document.title).toBe("No gateway chosen · Otari")
    expect(load).not.toHaveBeenCalled()
  })

  it("reloads the page for a sign-in, rather than asking again from here", async () => {
    const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime })
    const reload = vi.fn()
    vi.spyOn(window, "location", "get").mockReturnValue({
      ...window.location,
      host: window.location.host,
      reload,
    })
    const load = vi.fn<Load>(async () => {
      throw new RedirectedError()
    })
    await mountGate(load)

    const before = load.mock.calls.length
    await user.click(screen.getByRole("button", { name: "Reload" }))

    expect(reload).toHaveBeenCalledTimes(1)
    // The press reloads; it does not start an attempt of its own.
    expect(load).toHaveBeenCalledTimes(before)
  })

  it("opens the app from the sign-in screen, without a press, once the gateway answers", async () => {
    const { container } = await mountGate(
      vi
        .fn<Load>()
        .mockRejectedValueOnce(new RedirectedError())
        .mockResolvedValue(bootstrap()),
    )
    expect(
      screen.getByRole("heading", { level: 1, name: "Sign in to continue" }),
    ).toBeInTheDocument()
    // Retrying quietly: no sweep on a screen that asks the reader to act.
    expect(
      container
        .querySelector("[data-boot-card]")
        ?.classList.contains("otari-scan-border"),
    ).toBe(false)

    await act(async () => {
      await vi.advanceTimersByTimeAsync(500)
    })

    expect(
      await screen.findByRole("heading", { name: "Sign in to Otari" }),
    ).toBeInTheDocument()
  })

  it("keeps Retry pressable and unpending while background attempts run", async () => {
    const attempts = [pending(), pending()]
    const load = vi
      .fn<Load>()
      .mockRejectedValueOnce(DOWN)
      .mockImplementationOnce(() => attempts[0].promise)
      .mockImplementationOnce(() => attempts[1].promise)
    await mountGate(load)

    // The first background retry is running.
    await act(async () => {
      await vi.advanceTimersByTimeAsync(500)
    })
    expect(load).toHaveBeenCalledTimes(2)
    expect(retryButton()).toHaveTextContent("Retry now")
    expect(retryButton()).not.toHaveAttribute("data-pending")
    expect(retryButton()).toBeEnabled()

    // It fails; the row updates without saying anything.
    await act(async () => {
      attempts[0].settle().fail(DOWN)
    })
    expect(retryButton()).toHaveTextContent("Retry now")
    expect(lastTry()).toMatch(/· no response$/)
    expect(announce).toHaveBeenCalledTimes(1)
  })

  it("goes pending on a press, adopting the attempt already running, and says when it fails", async () => {
    const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime })
    const running = pending()
    const load = vi
      .fn<Load>()
      .mockRejectedValueOnce(DOWN)
      .mockImplementationOnce(() => running.promise)
    await mountGate(load)
    await act(async () => {
      await vi.advanceTimersByTimeAsync(500)
    })
    expect(load).toHaveBeenCalledTimes(2)

    await user.click(retryButton())

    // Answered by the attempt that was already running, not a third.
    expect(load).toHaveBeenCalledTimes(2)
    expect(retryButton()).toHaveTextContent("Retrying…")
    expect(retryButton()).toHaveAttribute("data-pending", "true")

    await act(async () => {
      running.settle().fail(DOWN)
    })
    expect(retryButton()).toHaveTextContent("Retry now")
    expect(lastTry()).toMatch(/^\d\d:\d\d:\d\d · still no answer$/)
    expect(vi.mocked(announce).mock.calls.map(([message]) => message)).toEqual([
      UNREACHABLE,
      "Still no answer.",
    ])
  })

  it("announces the first failure and a change of failure, and nothing else", async () => {
    const load = vi
      .fn<Load>()
      .mockRejectedValueOnce(DOWN)
      .mockRejectedValueOnce(DOWN)
      .mockRejectedValueOnce(new ApiError(503, "Service Unavailable"))
      .mockRejectedValueOnce(DOWN)
    await mountGate(load)

    // Three background attempts: same, same family but another reason, then
    // back. The reasons differ; the failure, as announced, does not.
    await act(async () => {
      await vi.advanceTimersByTimeAsync(500 + 1_000 + 2_000)
    })
    expect(load).toHaveBeenCalledTimes(4)
    expect(vi.mocked(announce).mock.calls.map(([message]) => message)).toEqual([
      UNREACHABLE,
    ])
  })

  it("announces a failure that changes kind", async () => {
    const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime })
    const load = vi
      .fn<Load>()
      .mockRejectedValueOnce(new ApiError(404, "Not Found"))
      .mockRejectedValueOnce(new ApiError(401, "Unauthorized"))
    await mountGate(load)

    await user.click(screen.getByRole("button", { name: "Retry" }))

    expect(
      await screen.findByRole("heading", { name: "Sign in to continue" }),
    ).toBeInTheDocument()
    expect(vi.mocked(announce).mock.calls.map(([message]) => message)).toEqual([
      "This address didn’t answer like a gateway",
      "Sign in to continue",
    ])
  })

  it("names the address it asked, where that is not the page's own", async () => {
    policy.origin = "https://api.example.com:8443"
    await mountGate(async () => {
      throw DOWN
    })

    expect(screen.getByText("api.example.com:8443")).toBeInTheDocument()
    expect(screen.queryByText(window.location.host)).toBeNull()
  })

  it("shows the status a pressed Retry got, in place of saying it still failed", async () => {
    const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime })
    const load = vi
      .fn<Load>()
      .mockRejectedValueOnce(DOWN)
      .mockRejectedValueOnce(new ApiError(503, "Service Unavailable"))
    await mountGate(load)

    await user.click(retryButton())

    await vi.waitFor(() =>
      expect(lastTry()).toMatch(/^\d\d:\d\d:\d\d · HTTP 503$/),
    )
    expect(vi.mocked(announce).mock.calls.at(-1)).toEqual(["Still no answer."])
  })

  it("has no live region in the card", async () => {
    const { container } = await mountGate(async () => {
      throw DOWN
    })
    const card = container.querySelector("[data-boot-card]") as HTMLElement
    expect(card.querySelector("[aria-live], [role=status], [role=alert]")).toBe(
      null,
    )
    expect(within(card).queryByRole("status")).toBeNull()
  })

  it("becomes the app, without a reload, when a background retry answers", async () => {
    const load = vi
      .fn<Load>()
      .mockRejectedValueOnce(DOWN)
      .mockResolvedValue(bootstrap())
    await mountGate(load)
    expect(screen.getByRole("heading", { name: UNREACHABLE })).toBeVisible()

    await act(async () => {
      await vi.advanceTimersByTimeAsync(500)
    })

    expect(
      await screen.findByRole("heading", { name: "Sign in to Otari" }),
    ).toBeInTheDocument()
    expect(screen.queryByRole("button", { name: /Retry/ })).toBeNull()
    expect(document.title).not.toMatch(/reach the gateway/)
  })

  it("asks again when Retry is pressed after a refusal", async () => {
    const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime })
    const second = pending()
    const load = vi
      .fn<Load>()
      .mockRejectedValueOnce(new ApiError(404, "Not Found"))
      .mockImplementationOnce(() => second.promise)
    await mountGate(load)

    await user.click(screen.getByRole("button", { name: "Retry" }))

    expect(load).toHaveBeenCalledTimes(2)
    expect(
      screen.getByRole("button", { name: "Retrying…" }),
    ).toBeInTheDocument()

    await act(async () => {
      second.settle().answer(bootstrap())
    })
    expect(
      await screen.findByRole("heading", { name: "Sign in to Otari" }),
    ).toBeInTheDocument()
  })
})
