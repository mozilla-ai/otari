import { describe, expect, it } from "vitest"

import { hostedRegion } from "@/tests/hostedProviders"

import { switchOffConsequence } from "./RegionsCard"

const regions = [
  hostedRegion({ slug: "us", label: "West US", position: 0 }),
  hostedRegion({ slug: "sg", label: "Southeast Asia", position: 1 }),
  hostedRegion({ slug: "cn", label: "China", position: 2 }),
]

describe("switchOffConsequence", () => {
  it("names the next serving region only as the likely one", () => {
    const text = switchOffConsequence(regions, regions[0], "tokenhub")
    expect(text).toContain(
      "go to the first other serving region that admits them",
    )
    expect(text).toContain("are refused where none does")
    expect(text).toContain(
      "Southeast Asia, the next serving region in order, is the likely one.",
    )
    expect(text).toContain(
      "A request that pins openai@us is refused while it is off.",
    )
  })

  it("skips a switched-off region when naming the likely one", () => {
    const withSgOff = regions.map((region) =>
      region.slug === "sg" ? { ...region, enabled: false } : region,
    )
    expect(switchOffConsequence(withSgOff, withSgOff[0], "tokenhub")).toContain(
      "China, the next serving region in order, is the likely one.",
    )
  })

  it("names no likely region after the last one", () => {
    const text = switchOffConsequence(regions, regions[2], "tokenhub")
    expect(text).not.toContain("likely")
    expect(text).toContain("refused where none does")
    expect(text).toContain("openai@cn is refused while it is off.")
  })

  it("says every request is refused when no other region serves", () => {
    const alone = [regions[0], { ...regions[1], enabled: false }]
    expect(switchOffConsequence(alone, alone[0], "tokenhub")).toBe(
      "No region serves tokenhub then, so every request to it is refused.",
    )
  })
})
