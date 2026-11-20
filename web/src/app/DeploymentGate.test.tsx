import { act, render, screen } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { type BootLoader, createBootLoader, firstSettled } from "@/app/boot"
import { DeploymentGate } from "@/app/DeploymentGate"
import { Provider } from "@/app/provider"
import { ApiError } from "@/shared/api/client"
import type { WireBootstrap } from "@/shared/helpers/bootstrap"
import { bootstrap } from "@/tests/fixtures"

const DOWN = new ApiError(0, "Network error: could not reach the gateway.")

let loader: BootLoader | undefined

// Mounted the way `main.tsx` mounts it: after the first attempt settles.
async function mountGate(
  load: (signal: AbortSignal) => Promise<WireBootstrap>,
) {
  loader = createBootLoader({
    prepare: async () => {},
    load,
    random: () => 0,
  })
  await act(async () => {
    await firstSettled(loader as BootLoader)
  })
  return render(
    <Provider>
      <DeploymentGate loader={loader} />
    </Provider>,
  )
}

describe("DeploymentGate", () => {
  beforeEach(() => {
    vi.useFakeTimers({ shouldAdvanceTime: true })
  })

  afterEach(() => {
    loader?.dispose()
    loader = undefined
    vi.useRealTimers()
    window.localStorage.clear()
    window.history.replaceState(null, "", "/")
  })

  it("says the gateway is unreachable instead of guessing a deployment", async () => {
    await mountGate(async () => {
      throw DOWN
    })

    expect(screen.getByRole("alert")).toHaveTextContent(
      /does not know what it is connected to/,
    )
    expect(screen.getByRole("status")).toHaveTextContent(
      /keeps trying in the background/,
    )
    expect(screen.getByRole("button", { name: "Retry" })).toBeInTheDocument()
    expect(
      screen.queryByRole("heading", { name: "Sign in to Otari" }),
    ).toBeNull()
    expect(screen.queryByRole("navigation")).toBeNull()
  })

  it("shows the gateway's own refusal rather than calling it unreachable", async () => {
    await mountGate(async () => {
      throw new ApiError(404, "Not Found")
    })

    expect(screen.getByRole("alert")).toHaveTextContent("Not Found")
    expect(screen.getByRole("status")).toHaveTextContent(
      /The gateway refused the request/,
    )
  })

  it("says no gateway was asked when the deployment could not be chosen", async () => {
    const load = vi.fn<(signal: AbortSignal) => Promise<WireBootstrap>>()
    loader = createBootLoader({
      prepare: async () => {
        throw new Error("no deployment selected")
      },
      load,
      random: () => 0,
    })
    await act(async () => {
      await firstSettled(loader as BootLoader)
    })
    render(
      <Provider>
        <DeploymentGate loader={loader} />
      </Provider>,
    )

    expect(screen.getByRole("alert")).toHaveTextContent(
      /could not decide which gateway to reach/,
    )
    expect(screen.getByRole("status")).not.toHaveTextContent(/refused/)
    expect(load).not.toHaveBeenCalled()
  })

  it("becomes the app, without a reload, when a background retry answers", async () => {
    const load = vi
      .fn<(signal: AbortSignal) => Promise<WireBootstrap>>()
      .mockRejectedValueOnce(DOWN)
      .mockResolvedValue(bootstrap())
    await mountGate(load)
    expect(screen.getByRole("alert")).toBeInTheDocument()

    await act(async () => {
      await vi.advanceTimersByTimeAsync(500)
    })

    expect(
      await screen.findByRole("heading", { name: "Sign in to Otari" }),
    ).toBeInTheDocument()
    expect(screen.queryByRole("button", { name: "Retry" })).toBeNull()
  })

  it("asks again when Retry is pressed", async () => {
    const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime })
    let answer: (value: WireBootstrap) => void = () => {}
    const load = vi
      .fn<(signal: AbortSignal) => Promise<WireBootstrap>>()
      .mockRejectedValueOnce(new ApiError(404, "Not Found"))
      .mockImplementationOnce(
        () =>
          new Promise((resolve) => {
            answer = resolve
          }),
      )
    await mountGate(load)
    // A refusal is not retried on its own, so only the button reaches the
    // second call.
    expect(screen.getByRole("status")).toHaveTextContent(/then retry/)

    await user.click(screen.getByRole("button", { name: "Retry" }))

    expect(load).toHaveBeenCalledTimes(2)
    expect(screen.getByRole("status")).toHaveTextContent("Trying again…")

    await act(async () => {
      answer(bootstrap())
    })
    expect(
      await screen.findByRole("heading", { name: "Sign in to Otari" }),
    ).toBeInTheDocument()
  })
})
