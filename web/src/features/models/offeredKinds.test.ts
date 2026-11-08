import { describe, expect, it } from "vitest"

import { kindFilterOf, kindTabs, offeredPrice } from "./offeredKinds"

describe("kindTabs", () => {
  it("leads with every model, then each kind offered, in a fixed order", () => {
    expect(kindTabs({ video: 3, chat: 40, unknown: 5 }, "")).toEqual([
      { kind: "", label: "All", count: 48 },
      { kind: "chat", label: "Text", count: 40 },
      { kind: "video", label: "Video", count: 3 },
      { kind: "unknown", label: "Unknown", count: 5 },
    ])
  })

  it("keeps the kind shown even when it has no models left", () => {
    expect(kindTabs({ chat: 2 }, "image").map((tab) => tab.kind)).toEqual([
      "",
      "chat",
      "image",
    ])
  })

  it("is only All before the counts arrive", () => {
    expect(kindTabs(undefined, "")).toEqual([
      { kind: "", label: "All", count: 0 },
    ])
  })
})

describe("kindFilterOf", () => {
  it("reads a kind, and anything else as every kind", () => {
    expect(kindFilterOf("video")).toBe("video")
    expect(kindFilterOf("unknown")).toBe("unknown")
    expect(kindFilterOf("bogus")).toBe("")
    expect(kindFilterOf("")).toBe("")
  })
})

describe("offeredPrice", () => {
  const tokens = {
    unit: "token",
    input_price_per_million: 0.28,
    output_price_per_million: 1.1,
    cache_read_price_per_million: 0.056,
    cache_write_price_per_million: null,
  }

  it("shows a token price bare, per million tokens", () => {
    expect(offeredPrice(tokens, "input")).toEqual({ amount: "$0.28" })
    expect(offeredPrice(tokens, "output")).toEqual({ amount: "$1.10" })
    expect(offeredPrice(tokens, "cache_read")).toEqual({ amount: "$0.056" })
    expect(offeredPrice(tokens, "cache_write")).toBeUndefined()
  })

  it("puts a video model's per-second price under output, naming its unit", () => {
    const video = {
      type: "video",
      unit: "second",
      input_price_per_million: 45_000,
    }
    expect(offeredPrice(video, "input")).toBeUndefined()
    expect(offeredPrice(video, "output")).toEqual({
      amount: "$0.045",
      title: "Video output",
      per: "/second",
      rate: "$0.045/second",
    })
  })

  it("keeps any other model's per-second price on its audio side", () => {
    const transcription = {
      type: "transcription",
      unit: "second",
      input_price_per_million: 100,
    }
    expect(offeredPrice(transcription, "input")).toEqual({
      amount: "$0.0001",
      title: "Audio input",
      per: "/second",
      rate: "$0.0001/second",
    })
    expect(offeredPrice(transcription, "output")).toBeUndefined()
  })

  it("puts an image price under output", () => {
    expect(
      offeredPrice(
        { unit: "image", input_price_per_million: 40_000 },
        "output",
      ),
    ).toEqual({
      amount: "$0.04",
      title: "Image output",
      per: "/image",
      rate: "$0.04/image",
    })
  })

  it("keeps an audio price on the side it is stored on", () => {
    const transcription = { unit: "minute", input_price_per_million: 6_000 }
    expect(offeredPrice(transcription, "input")).toEqual({
      amount: "$0.006",
      title: "Audio input",
      per: "/minute",
      rate: "$0.006/minute",
    })
    expect(offeredPrice(transcription, "output")).toBeUndefined()
  })

  it("names the unit of a per-request, per-character or per-search price", () => {
    expect(
      offeredPrice({ unit: "request", input_price_per_million: 2000 }, "input"),
    ).toEqual({
      amount: "$0.002",
      title: "Request",
      per: "/request",
      rate: "$0.002/request",
    })
    expect(
      offeredPrice({ unit: "character", input_price_per_million: 15 }, "input"),
    ).toEqual({
      amount: "$15.00",
      title: "Text input",
      per: "/1M characters",
      rate: "$15.00/1M characters",
    })
    expect(
      offeredPrice(
        { unit: "web_search", input_price_per_million: 10_000 },
        "input",
      ),
    ).toEqual({
      amount: "$10.00",
      title: "Web search",
      per: "/1K searches",
      rate: "$10.00/1K searches",
    })
  })

  it("has no cache price for a unit other than tokens", () => {
    expect(
      offeredPrice(
        {
          unit: "request",
          input_price_per_million: 2000,
          cache_read_price_per_million: 1,
        },
        "cache_read",
      ),
    ).toBeUndefined()
  })
})
