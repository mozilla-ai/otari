import { afterEach, describe, expect, it, vi } from "vitest"
import type { WireBootstrap } from "@/shared/helpers/bootstrap"
import { type KeptBootstrap, keptBootstrapAt, startDeployment } from "./boot"

const BOOTSTRAP = { deployment_type: "standalone" } as WireBootstrap
const HOSTED = {
  deployment_type: "hosted",
  maintenance_mode: false,
} as WireBootstrap
const MAINTENANCE = { ...HOSTED, maintenance_mode: true } as WireBootstrap

function keeping(bootstrap: WireBootstrap | null = null) {
  return {
    read: vi.fn(() => bootstrap),
    write: vi.fn(),
  } satisfies KeptBootstrap
}

const settled = async () => {}

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

    await startDeployment(prepare, load, keeping(), render)

    expect(order).toEqual(["prepare", "load"])
    expect(render.mock.calls).toEqual([[BOOTSTRAP]])
  })

  it("renders null and never asks for a bootstrap when the deployment could not be chosen", async () => {
    const load = vi.fn(async () => BOOTSTRAP)
    const kept = keeping(HOSTED)
    const render = vi.fn()

    await startDeployment(
      async () => {
        throw new Error("no directory")
      },
      load,
      kept,
      render,
    )

    expect(render.mock.calls).toEqual([[null]])
    expect(load).not.toHaveBeenCalled()
    expect(kept.read).not.toHaveBeenCalled()
  })

  it("renders null when nothing is kept and the bootstrap never arrives", async () => {
    const render = vi.fn()

    await startDeployment(settled, async () => null, keeping(), render)

    expect(render.mock.calls).toEqual([[null]])
  })

  it("renders a kept bootstrap before the current one arrives", async () => {
    let answer: (bootstrap: WireBootstrap) => void = () => {}
    const load = () =>
      new Promise<WireBootstrap>((resolve) => {
        answer = resolve
      })
    const render = vi.fn()

    const started = startDeployment(settled, load, keeping(HOSTED), render)

    await vi.waitFor(() => expect(render.mock.calls).toEqual([[HOSTED]]))
    answer(structuredClone(HOSTED))
    await started
    expect(render).toHaveBeenCalledOnce()
  })

  it("renders the current bootstrap when it differs from the kept one", async () => {
    const kept = keeping(HOSTED)
    const render = vi.fn()

    await startDeployment(settled, async () => MAINTENANCE, kept, render)

    expect(render.mock.calls).toEqual([[HOSTED], [MAINTENANCE]])
    expect(kept.write).toHaveBeenCalledWith(MAINTENANCE)
  })

  it("renders null over a kept bootstrap when the current one never arrives", async () => {
    const kept = keeping(HOSTED)
    const render = vi.fn()

    await startDeployment(settled, async () => null, kept, render)

    expect(render.mock.calls).toEqual([[HOSTED], [null]])
    expect(kept.write).toHaveBeenCalledWith(null)
  })
})

describe("keptBootstrapAt", () => {
  afterEach(() => {
    window.localStorage.clear()
  })

  it("answers the hosted bootstrap kept for the same origin", () => {
    keptBootstrapAt(() => "https://eu.api.otari.test").write(HOSTED)

    expect(keptBootstrapAt(() => "https://eu.api.otari.test").read()).toEqual(
      HOSTED,
    )
  })

  it("answers nothing for another origin", () => {
    keptBootstrapAt(() => "https://eu.api.otari.test").write(HOSTED)

    expect(keptBootstrapAt(() => "").read()).toBeNull()
  })

  it("keeps no bootstrap but a hosted one", () => {
    const kept = keptBootstrapAt(() => "")
    kept.write(HOSTED)
    kept.write(BOOTSTRAP)

    expect(kept.read()).toBeNull()
  })

  it("forgets what it kept when the bootstrap never arrived", () => {
    const kept = keptBootstrapAt(() => "")
    kept.write(HOSTED)
    kept.write(null)

    expect(kept.read()).toBeNull()
  })

  it("answers nothing for a kept value it cannot read", () => {
    window.localStorage.setItem("otari.dashboard.bootstrap", "{not json")

    expect(keptBootstrapAt(() => "").read()).toBeNull()
  })
})
