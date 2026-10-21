import { describe, expect, it } from "vitest"

import {
  GATEWAY_ROWS,
  instanceRows,
  isSetInConfig,
  perCall,
  storedFromPerCall,
} from "./toolRates"

describe("instanceRows", () => {
  it("lists the configured instances a stored one does not shadow, then the stored", () => {
    const rows = instanceRows("search", {
      stored: [
        {
          name: "exa",
          kind: "search",
          provider: "exa",
          options: {},
          decryptable: true,
          shadows_config: true,
        },
      ],
      config: [
        {
          name: "exa",
          kind: "search",
          provider: "exa",
          has_api_key: true,
          shadowed: true,
        },
        {
          name: "brave",
          kind: "search",
          provider: "brave",
          has_api_key: true,
          shadowed: false,
        },
      ],
    })

    expect(
      rows.map((row) => [row.kind, row.modelKey, row.isConfigured]),
    ).toEqual([
      ["search", "brave", true],
      ["search", "exa", false],
    ])
  })

  it("lists nothing before the instances are read", () => {
    expect(instanceRows("fetch", undefined)).toEqual([])
  })
})

describe("isSetInConfig", () => {
  const [configured] = instanceRows("search", {
    stored: [],
    config: [
      {
        name: "brave",
        kind: "search",
        provider: "brave",
        has_api_key: true,
        shadowed: false,
      },
    ],
  })

  it("holds a configured instance's rate written from its price_per_call", () => {
    expect(isSetInConfig(configured, "config")).toBe(true)
  })

  it("leaves a configured instance with an API rate editable", () => {
    expect(isSetInConfig(configured, "api")).toBe(false)
    expect(isSetInConfig(configured, undefined)).toBe(false)
  })

  it("never holds a gateway row, which no config entry prices", () => {
    expect(isSetInConfig(GATEWAY_ROWS.code_execution[0], "config")).toBe(false)
  })
})

describe("per-call rates", () => {
  it("reads and writes dollars per call against a per-million store", () => {
    expect(perCall(10_000)).toBe(0.01)
    expect(storedFromPerCall("0.01")).toBe(10_000)
    expect(storedFromPerCall(" 0.07 ")).toBe(70_000)
    expect(storedFromPerCall("0")).toBe(0)
  })

  it.each(["", "1e3", "0x10", "-1", ".5", "abc"])("refuses %j", (text) => {
    expect(storedFromPerCall(text)).toBeUndefined()
  })
})
