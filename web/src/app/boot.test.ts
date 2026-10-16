import { describe, expect, it, vi } from "vitest"
import type { WireBootstrap } from "@/shared/helpers/bootstrap"
import { loadDeployment } from "./boot"

const BOOTSTRAP = { deployment_type: "standalone" } as WireBootstrap

describe("loadDeployment", () => {
  it("reads the bootstrap once the request policy is settled", async () => {
    const order: string[] = []
    const prepare = vi.fn(async () => {
      order.push("prepare")
    })
    const load = vi.fn(async () => {
      order.push("load")
      return BOOTSTRAP
    })

    await expect(loadDeployment(prepare, load)).resolves.toBe(BOOTSTRAP)
    expect(order).toEqual(["prepare", "load"])
  })

  it("answers null and never asks for a bootstrap when the deployment could not be chosen", async () => {
    const load = vi.fn(async () => BOOTSTRAP)

    await expect(
      loadDeployment(async () => {
        throw new Error("no directory")
      }, load),
    ).resolves.toBeNull()
    expect(load).not.toHaveBeenCalled()
  })
})
