import { describe, expect, it } from "vitest"

import { pricingResponse } from "@/tests/fixtures"

import {
  GATEWAY_ROWS,
  hasUnpricedRow,
  pairWithRates,
  parsePerCallPrice,
  type RateRead,
  toPerCall,
} from "./toolRates"

describe("per-call rates", () => {
  it("reads and writes dollars per call against a per-million store", () => {
    expect(toPerCall(10_000)).toBe(0.01)
    expect(parsePerCallPrice("0.01")).toBe(10_000)
    expect(parsePerCallPrice(" 0.07 ")).toBe(70_000)
    expect(parsePerCallPrice("0")).toBe(0)
  })

  it.each(["", "1e3", "0x10", "-1", ".5", "abc"])("refuses %j", (text) => {
    expect(parsePerCallPrice(text)).toBeUndefined()
  })
})

describe("pairWithRates", () => {
  const [codeExecution] = GATEWAY_ROWS.code_execution
  const priced: RateRead = {
    data: pricingResponse({
      model_key: "otari:code_execution",
      input_price_per_million: 10_000,
    }),
    isPending: false,
    isError: false,
  }
  const unpriced: RateRead = { data: null, isPending: false, isError: false }

  it("calls a row unpriced once its rate is read and absent", () => {
    expect(hasUnpricedRow(pairWithRates([codeExecution], [unpriced]))).toBe(
      true,
    )
    expect(hasUnpricedRow(pairWithRates([codeExecution], [priced]))).toBe(false)
  })

  it("does not call a row unpriced before its rate is read", () => {
    const pending: RateRead = { isPending: true, isError: false }
    const failed: RateRead = { isPending: false, isError: true }
    expect(hasUnpricedRow(pairWithRates([codeExecution], [pending]))).toBe(
      false,
    )
    expect(hasUnpricedRow(pairWithRates([codeExecution], [failed]))).toBe(false)
  })
})
