import { afterEach, describe, expect, it, vi } from "vitest"
import { KEPT_ANSWER_HOURS } from "@/shared/api/keptAnswers"
import {
  type RequestPolicy,
  SAME_ORIGIN_POLICY,
} from "@/shared/api/requestPolicy"
import type { WireBootstrap } from "@/shared/helpers/bootstrap"
import { type KeptBootstrap, openKeptBootstrap, startDeployment } from "./boot"

const BOOTSTRAP: WireBootstrap = { surfaces: [], maintenance_mode: false }
const MAINTENANCE: WireBootstrap = { ...BOOTSTRAP, maintenance_mode: true }
const KEPT_KEY = "otari.dashboard.bootstrap"

function keeping(origin = ""): RequestPolicy {
  return {
    origin,
    credentials: origin ? "include" : "same-origin",
    keptAnswers: { bootstrap: true, organizationContext: false },
  }
}

function stubKept(bootstrap?: WireBootstrap) {
  return {
    read: vi.fn(() => bootstrap),
    write: vi.fn(),
  } satisfies KeptBootstrap
}

const prepareNothing = async () => {}

function marks(): string[] {
  return performance
    .getEntriesByType("mark")
    .map((entry) => entry.name)
    .filter((name) => name.startsWith("boot:"))
}

afterEach(() => {
  window.localStorage.clear()
  performance.clearMarks()
  vi.useRealTimers()
})

describe("startDeployment", () => {
  it("reads the bootstrap once the request policy is settled", async () => {
    const order: string[] = []
    const prepare = vi.fn(async () => {
      order.push("prepare")
    })
    const load = vi.fn(async () => {
      order.push("load")
      return BOOTSTRAP
    })
    const render = vi.fn()

    await startDeployment(prepare, load, () => null, render)

    expect(order).toEqual(["prepare", "load"])
    expect(render.mock.calls).toEqual([[BOOTSTRAP]])
  })

  it("renders null and never asks for a bootstrap when the deployment could not be chosen", async () => {
    const load = vi.fn(async () => BOOTSTRAP)
    const keep = vi.fn(() => stubKept(BOOTSTRAP))
    const render = vi.fn()

    await startDeployment(
      async () => {
        throw new Error("no directory")
      },
      load,
      keep,
      render,
    )

    expect(render.mock.calls).toEqual([[null]])
    expect(load).not.toHaveBeenCalled()
    expect(keep).not.toHaveBeenCalled()
  })

  it("waits for the bootstrap and marks the kept one off where the policy keeps none", async () => {
    const render = vi.fn()

    await startDeployment(
      prepareNothing,
      async () => null,
      () => null,
      render,
    )

    expect(render.mock.calls).toEqual([[null]])
    expect(marks()).toEqual(["boot:kept-bootstrap:off"])
  })

  it("renders a kept bootstrap before the current one arrives, and once when they agree", async () => {
    let answer: (bootstrap: WireBootstrap) => void = () => {}
    const load = () =>
      new Promise<WireBootstrap>((resolve) => {
        answer = resolve
      })
    const render = vi.fn()

    const started = startDeployment(
      prepareNothing,
      load,
      () => stubKept(BOOTSTRAP),
      render,
    )

    await vi.waitFor(() => expect(render.mock.calls).toEqual([[BOOTSTRAP]]))
    // The same answer with its fields in another order is the same answer.
    answer({ maintenance_mode: false, surfaces: [] })
    await started
    expect(render).toHaveBeenCalledOnce()
    expect(marks()).toEqual(["boot:kept-bootstrap:hit"])
  })

  it("renders the current bootstrap when it differs from the kept one, and keeps it", async () => {
    const kept = stubKept(BOOTSTRAP)
    const render = vi.fn()

    await startDeployment(
      prepareNothing,
      async () => MAINTENANCE,
      () => kept,
      render,
    )

    expect(render.mock.calls).toEqual([[BOOTSTRAP], [MAINTENANCE]])
    expect(kept.write).toHaveBeenCalledWith(MAINTENANCE)
    expect(marks()).toEqual(["boot:kept-bootstrap:hit", "boot:remount"])
  })

  it("leaves a kept bootstrap on screen and forgets it when the current one never arrives", async () => {
    const kept = stubKept(BOOTSTRAP)
    const render = vi.fn()

    await startDeployment(
      prepareNothing,
      async () => null,
      () => kept,
      render,
    )

    expect(render.mock.calls).toEqual([[BOOTSTRAP]])
    expect(kept.write).toHaveBeenCalledWith(undefined)
  })

  it("marks a miss and renders the current bootstrap when nothing is kept yet", async () => {
    const kept = stubKept()
    const render = vi.fn()

    await startDeployment(
      prepareNothing,
      async () => BOOTSTRAP,
      () => kept,
      render,
    )

    expect(render.mock.calls).toEqual([[BOOTSTRAP]])
    expect(kept.write).toHaveBeenCalledWith(BOOTSTRAP)
    expect(marks()).toEqual(["boot:kept-bootstrap:miss"])
  })

  it("keeps the bootstrap under the origin it was asked from", async () => {
    let policy = keeping("https://eu.api.otari.test")
    let answer: ((bootstrap: WireBootstrap) => void) | undefined
    const load = () =>
      new Promise<WireBootstrap>((resolve) => {
        answer = resolve
      })

    const started = startDeployment(
      prepareNothing,
      load,
      () => openKeptBootstrap(policy, ""),
      vi.fn(),
    )
    await vi.waitFor(() => expect(answer).toBeDefined())
    // The request policy moves while the request is in flight.
    policy = keeping()
    answer?.(BOOTSTRAP)
    await started

    expect(
      openKeptBootstrap(keeping("https://eu.api.otari.test"), "")?.read(),
    ).toEqual(BOOTSTRAP)
    expect(openKeptBootstrap(keeping(), "")?.read()).toBeUndefined()
  })
})

describe("openKeptBootstrap", () => {
  it("keeps nothing, and forgets what an earlier build kept, where the policy keeps no bootstrap", () => {
    openKeptBootstrap(keeping(), "")?.write(BOOTSTRAP)
    expect(window.localStorage.getItem(KEPT_KEY)).not.toBeNull()

    expect(openKeptBootstrap(SAME_ORIGIN_POLICY, "")).toBeNull()
    expect(window.localStorage.getItem(KEPT_KEY)).toBeNull()

    // A replacing policy written before the switch existed reads as off.
    openKeptBootstrap(keeping(), "")?.write(BOOTSTRAP)
    expect(
      openKeptBootstrap({ origin: "", credentials: "same-origin" }, ""),
    ).toBeNull()
    expect(window.localStorage.getItem(KEPT_KEY)).toBeNull()
  })

  it("answers the bootstrap kept for the same origin, and nothing for another", () => {
    openKeptBootstrap(keeping("https://eu.api.otari.test"), "")?.write(
      BOOTSTRAP,
    )

    expect(
      openKeptBootstrap(keeping("https://eu.api.otari.test"), "")?.read(),
    ).toEqual(BOOTSTRAP)
    expect(openKeptBootstrap(keeping(), "")?.read()).toBeUndefined()
  })

  it("forgets what it kept when the bootstrap never arrived", () => {
    const kept = openKeptBootstrap(keeping(), "")
    kept?.write(BOOTSTRAP)
    kept?.write(undefined)

    expect(kept?.read()).toBeUndefined()
    expect(window.localStorage.getItem(KEPT_KEY)).toBeNull()
  })

  it("answers nothing for a kept value it cannot read", () => {
    window.localStorage.setItem(KEPT_KEY, "{not json")

    expect(openKeptBootstrap(keeping(), "")?.read()).toBeUndefined()
  })

  it("answers nothing once the entry is older than a dashboard session lives", () => {
    vi.useFakeTimers({ toFake: ["Date"] })
    openKeptBootstrap(keeping(), "")?.write(BOOTSTRAP)

    vi.advanceTimersByTime(KEPT_ANSWER_HOURS * 60 * 60 * 1000 - 1)
    expect(openKeptBootstrap(keeping(), "")?.read()).toEqual(BOOTSTRAP)
    vi.advanceTimersByTime(1)
    expect(openKeptBootstrap(keeping(), "")?.read()).toBeUndefined()
  })

  it("reads none on an OAuth callback, which spends its code on mount, but keeps the current one", () => {
    openKeptBootstrap(keeping(), "")?.write(BOOTSTRAP)

    const kept = openKeptBootstrap(
      keeping(),
      "#/auth/github/callback?code=c&state=s",
    )
    expect(kept?.read()).toBeUndefined()
    kept?.write(MAINTENANCE)

    expect(openKeptBootstrap(keeping(), "#/")?.read()).toEqual(MAINTENANCE)
    expect(
      openKeptBootstrap(keeping(), "#/verify-email?token=t")?.read(),
    ).toEqual(MAINTENANCE)
  })

  it("keeps nothing when storage refuses, and boots as if nothing were kept", async () => {
    vi.spyOn(Storage.prototype, "getItem").mockImplementation(() => {
      throw new DOMException("denied", "SecurityError")
    })
    vi.spyOn(Storage.prototype, "setItem").mockImplementation(() => {
      throw new DOMException("full", "QuotaExceededError")
    })
    const render = vi.fn()

    await startDeployment(
      prepareNothing,
      async () => BOOTSTRAP,
      () => openKeptBootstrap(keeping(), ""),
      render,
    )

    expect(render.mock.calls).toEqual([[BOOTSTRAP]])
    vi.restoreAllMocks()
  })
})
