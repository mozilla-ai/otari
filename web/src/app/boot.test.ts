import { afterEach, describe, expect, it, vi } from "vitest"
import type { WireBootstrap } from "@/shared/helpers/bootstrap"
import { type KeptBootstrap, openKeptBootstrap, startDeployment } from "./boot"

const BOOTSTRAP = { deployment_type: "standalone" } as WireBootstrap
const HOSTED = {
  deployment_type: "hosted",
  maintenance_mode: false,
} as WireBootstrap
const MAINTENANCE = { ...HOSTED, maintenance_mode: true } as WireBootstrap

function stubKept(bootstrap: WireBootstrap | null = null) {
  return {
    read: vi.fn(() => bootstrap),
    write: vi.fn(),
  } satisfies KeptBootstrap
}

const prepareNothing = async () => {}

afterEach(() => {
  window.localStorage.clear()
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

    await startDeployment(prepare, load, () => stubKept(), render)

    expect(order).toEqual(["prepare", "load"])
    expect(render.mock.calls).toEqual([[BOOTSTRAP]])
  })

  it("renders null and never asks for a bootstrap when the deployment could not be chosen", async () => {
    const load = vi.fn(async () => BOOTSTRAP)
    const keep = vi.fn(() => stubKept(HOSTED))
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

  it("renders null when nothing is kept and the bootstrap never arrives", async () => {
    const render = vi.fn()

    await startDeployment(
      prepareNothing,
      async () => null,
      () => stubKept(),
      render,
    )

    expect(render.mock.calls).toEqual([[null]])
  })

  it("renders a kept bootstrap before the current one arrives", async () => {
    let answer: (bootstrap: WireBootstrap) => void = () => {}
    const load = () =>
      new Promise<WireBootstrap>((resolve) => {
        answer = resolve
      })
    const render = vi.fn()

    const started = startDeployment(
      prepareNothing,
      load,
      () => stubKept(HOSTED),
      render,
    )

    await vi.waitFor(() => expect(render.mock.calls).toEqual([[HOSTED]]))
    answer(structuredClone(HOSTED))
    await started
    expect(render).toHaveBeenCalledOnce()
  })

  it("renders the current bootstrap when it differs from the kept one", async () => {
    const kept = stubKept(HOSTED)
    const render = vi.fn()

    await startDeployment(
      prepareNothing,
      async () => MAINTENANCE,
      () => kept,
      render,
    )

    expect(render.mock.calls).toEqual([[HOSTED], [MAINTENANCE]])
    expect(kept.write).toHaveBeenCalledWith(MAINTENANCE)
  })

  it("leaves a kept bootstrap on screen and forgets it when the current one never arrives", async () => {
    const kept = stubKept(HOSTED)
    const render = vi.fn()

    await startDeployment(
      prepareNothing,
      async () => null,
      () => kept,
      render,
    )

    expect(render.mock.calls).toEqual([[HOSTED]])
    expect(kept.write).toHaveBeenCalledWith(null)
  })

  it("keeps the bootstrap under the origin it was asked from", async () => {
    let origin = "https://eu.api.otari.test"
    let answer: ((bootstrap: WireBootstrap) => void) | undefined
    const load = () =>
      new Promise<WireBootstrap>((resolve) => {
        answer = resolve
      })

    const started = startDeployment(
      prepareNothing,
      load,
      () => openKeptBootstrap(origin),
      vi.fn(),
    )
    await vi.waitFor(() => expect(answer).toBeDefined())
    // The request policy moves while the request is in flight.
    origin = ""
    answer?.(HOSTED)
    await started

    expect(openKeptBootstrap("https://eu.api.otari.test").read()).toEqual(
      HOSTED,
    )
    expect(openKeptBootstrap("").read()).toBeNull()
  })
})

describe("openKeptBootstrap", () => {
  it("answers the hosted bootstrap kept for the same origin", () => {
    openKeptBootstrap("https://eu.api.otari.test").write(HOSTED)

    expect(openKeptBootstrap("https://eu.api.otari.test").read()).toEqual(
      HOSTED,
    )
  })

  it("answers nothing for another origin", () => {
    openKeptBootstrap("https://eu.api.otari.test").write(HOSTED)

    expect(openKeptBootstrap("").read()).toBeNull()
  })

  it("keeps no bootstrap but a hosted one", () => {
    const kept = openKeptBootstrap("")
    kept.write(HOSTED)
    kept.write(BOOTSTRAP)

    expect(kept.read()).toBeNull()
  })

  it("forgets what it kept when the bootstrap never arrived", () => {
    const kept = openKeptBootstrap("")
    kept.write(HOSTED)
    kept.write(null)

    expect(kept.read()).toBeNull()
  })

  it("answers nothing for a kept value it cannot read", () => {
    window.localStorage.setItem("otari.dashboard.bootstrap", "{not json")

    expect(openKeptBootstrap("").read()).toBeNull()
  })
})
