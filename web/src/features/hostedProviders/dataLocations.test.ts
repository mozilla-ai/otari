import { describe, expect, it } from "vitest"

import { hostedProvider } from "@/tests/hostedProviders"

import {
  DATA_LOCATION_OPTIONS,
  dataLocationLabel,
  matchDataLocations,
  primaryLabelFor,
  primarySlugFor,
  providerDataLocations,
  REGION_SLUG_PATTERN,
} from "./dataLocations"

const region = (slug: string, position: number, dataLocation?: string) => ({
  slug,
  selector: `openai@${slug}`,
  label: slug,
  position,
  enabled: true,
  has_api_key: false,
  decryptable: true,
  data_location: dataLocation ?? null,
})

describe("dataLocations", () => {
  it("names a location by its region, the EU and global", () => {
    expect(dataLocationLabel("DE")).toBe("Germany")
    expect(dataLocationLabel("EU")).toBe("European Union")
    expect(dataLocationLabel("GLOBAL")).toBe("Global")
  })

  it("offers Global and the EU ahead of the countries", () => {
    expect(DATA_LOCATION_OPTIONS.slice(0, 2).map((o) => o.value)).toEqual([
      "GLOBAL",
      "EU",
    ])
    expect(matchDataLocations("germ").map((o) => o.value)).toEqual(["DE"])
    expect(matchDataLocations("sg").map((o) => o.value)).toEqual(["SG"])
    expect(matchDataLocations("xk").map((o) => o.label)).toEqual(["Kosovo"])
  })

  it("keeps the selected location in a capped list", () => {
    const shown = matchDataLocations("", 10, "ZW")
    expect(shown).toHaveLength(11)
    expect(shown[0].value).toBe("ZW")
    expect(matchDataLocations("", 10, "GLOBAL")).toHaveLength(10)
  })

  it("lists every distinct effective location, a region inheriting the provider's", () => {
    const provider = hostedProvider({
      data_location: "CN",
      regions: [
        region("us", 2, "US"),
        region("cn", 0),
        region("sg", 1, "SG"),
        region("cn2", 3),
      ],
    })
    expect(providerDataLocations(provider)).toEqual(["CN", "SG", "US"])
    // A switched-off region serves nothing, so its location is not listed.
    const withSgOff = {
      ...provider,
      regions: provider.regions?.map((row) =>
        row.slug === "sg" ? { ...row, enabled: false } : row,
      ),
    }
    expect(providerDataLocations(withSgOff)).toEqual(["CN", "US"])
    expect(providerDataLocations(hostedProvider())).toEqual([])
    expect(
      providerDataLocations(hostedProvider({ data_location: "EU" })),
    ).toEqual(["EU"])
  })

  it("names the current connection after the provider's location", () => {
    expect(primarySlugFor(hostedProvider({ data_location: "CN" }))).toBe("cn")
    expect(primaryLabelFor(hostedProvider({ data_location: "CN" }))).toBe(
      "China",
    )
    expect(primarySlugFor(hostedProvider())).toBe("primary")
    expect(primaryLabelFor(hostedProvider())).toBe("Primary")
  })

  it("accepts a region id the API accepts", () => {
    expect(REGION_SLUG_PATTERN.test("sg")).toBe(true)
    expect(REGION_SLUG_PATTERN.test("us-west-2")).toBe(true)
    expect(REGION_SLUG_PATTERN.test("-sg")).toBe(false)
    expect(REGION_SLUG_PATTERN.test("SG")).toBe(false)
    expect(REGION_SLUG_PATTERN.test("a".repeat(33))).toBe(false)
  })
})
