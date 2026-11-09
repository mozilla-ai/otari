import { describe, expect, it } from "vitest"

import { priceRules } from "@/features/models/priceRules"

describe("priceRules", () => {
  it("names an override's price, the price it replaces and its condition", () => {
    expect(
      priceRules({
        text: {
          input: {
            price: 1.25,
            overrides: [{ when: { context: { gt: 272000 } }, price: 2.5 }],
          },
          output: { price: 10 },
        },
      }),
    ).toEqual([
      {
        change: "$2.50 / 1M for text input",
        when: "context is over 272,000 tokens",
      },
    ])
  })

  it("names a price in its own unit", () => {
    expect(
      priceRules({
        video: {
          output: {
            price: 0.02,
            unit: "second",
            overrides: [
              {
                when: { resolution: { equals: "1080p" } },
                price: 0.04,
              },
            ],
          },
        },
        web_search: {
          price: 10,
          overrides: [{ when: { region: { in: ["eu", "us"] } }, price: 12 }],
        },
      }),
    ).toEqual([
      {
        change: "$0.04 per second for video output",
        when: "resolution is 1080p",
      },
      {
        change: "$12.00 / 1K searches for web search",
        when: "region is one of eu or us",
      },
    ])
  })

  it("reads a list under 'one of' apart from the tests beside it", () => {
    // Commas between both read two resolutions as three.
    expect(
      priceRules({
        image: {
          output: {
            price: 0.011,
            unit: "image",
            overrides: [
              {
                when: {
                  all: [
                    { resolution: { in: ["1024x1536", "1536x1024"] } },
                    { quality: { equals: "low" } },
                  ],
                },
                price: 0.006,
              },
              {
                when: { quality: { in: ["low", "medium", "high"] } },
                price: 0.02,
              },
            ],
          },
        },
      }),
    ).toEqual([
      {
        change: "$0.006 per image for image output",
        when: "resolution is one of 1024x1536 or 1536x1024 and quality is low",
      },
      {
        change: "$0.02 per image for image output",
        when: "quality is one of low, medium or high",
      },
    ])
  })

  it("names a multiplier's factor and the prices it scales", () => {
    expect(
      priceRules({
        multipliers: [
          { when: { service_tier: { equals: "flex" } }, factor: 0.5 },
          {
            when: { service_tier: { equals: "priority" } },
            factor: { text: { output: 2 }, image: 1.5 },
          },
        ],
      }),
    ).toEqual([
      { change: "×0.5 on every price", when: "service tier is flex" },
      {
        change: "×2 on text output, ×1.5 on image",
        when: "service tier is priority",
      },
    ])
  })

  it("reads the combinators", () => {
    expect(
      priceRules({
        multipliers: [
          {
            when: {
              all: [
                { utc_days: { in: ["sat", "sun"] } },
                { not: { generate_audio: { equals: true } } },
              ],
            },
            factor: 0.8,
          },
          {
            when: {
              any: [
                { duration: { gte: 10 } },
                { aspect_ratio: { equals: "9:16" } },
              ],
              quality: { lte: "high" },
            },
            factor: 1.2,
          },
        ],
      }),
    ).toEqual([
      {
        change: "×0.8 on every price",
        when: "UTC day is one of sat or sun and not (generate audio is true)",
      },
      {
        change: "×1.2 on every price",
        when: "(duration is at least 10 seconds or aspect ratio is 9:16) and quality is at most high",
      },
    ])
  })

  it("keeps the bounds of a group nested in another", () => {
    expect(
      priceRules({
        multipliers: [
          {
            when: {
              any: [
                {
                  all: [
                    { service_tier: { equals: "flex" } },
                    { context: { gte: 1000 } },
                  ],
                },
                { quality: { equals: "low" } },
              ],
            },
            factor: 0.5,
          },
        ],
      }),
    ).toEqual([
      {
        change: "×0.5 on every price",
        when: "(service tier is flex and context is at least 1,000 tokens) or quality is low",
      },
    ])
  })

  it("names a cache and reasoning price an override sets", () => {
    expect(
      priceRules({
        text: {
          input: {
            price: 1,
            overrides: [
              {
                when: { context: { gte: 200000 } },
                cache: { read: 0.2, write: 2 },
                reasoning: 3,
              },
            ],
          },
        },
      }),
    ).toEqual([
      {
        change:
          "$3.00 / 1M for text reasoning, $0.20 / 1M for text input cache reads, $2.00 / 1M for text input cache writes",
        when: "context is at least 200,000 tokens",
      },
    ])
  })

  it("names every cache price an override sets: each TTL, storage and explicit caching", () => {
    expect(
      priceRules({
        text: {
          input: {
            price: 1,
            overrides: [
              {
                when: { context: { gte: 200000 } },
                cache: {
                  write: { "5m": 1.25, "1h": 2 },
                  storage: 4.5,
                  explicit: { read: 0.1 },
                },
              },
            ],
          },
        },
      }),
    ).toEqual([
      {
        change:
          "$1.25 / 1M for text input cache writes (5m), $2.00 / 1M for text input cache writes (1h), $4.50 / 1M for text input cache storage, $0.10 / 1M for text input explicit cache reads",
        when: "context is at least 200,000 tokens",
      },
    ])
  })

  it("reads nothing from a block with no rules, or no block", () => {
    expect(priceRules({ text: { input: { price: 1 } } })).toEqual([])
    expect(priceRules(undefined)).toEqual([])
    expect(priceRules("not a block")).toEqual([])
  })
})
