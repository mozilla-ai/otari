import { describe, expect, it } from "vitest"

import {
  groupModelsByVendor,
  OTHER_VENDOR,
  vendorFor,
} from "@/features/workspaces/access/modelVendors"

const CATALOG = new Map([["DeepSeek-V3-0324", "deepseek"]])

describe("vendorFor", () => {
  it("takes the catalog's vendor, then the name before a slash, then Other", () => {
    expect(vendorFor("DeepSeek-V3-0324", CATALOG)).toBe("deepseek")
    expect(vendorFor("meta-llama/Llama-3.3-70B", CATALOG)).toBe("meta-llama")
    expect(vendorFor("mystery-1", CATALOG)).toBe(OTHER_VENDOR)
  })
})

describe("groupModelsByVendor", () => {
  it("groups by vendor in name order, Other last", () => {
    expect(
      groupModelsByVendor(
        [
          "mystery-1",
          "meta-llama/Llama-3.3-70B",
          "DeepSeek-V3-0324",
          "deepseek-ai/DeepSeek-R1",
        ],
        CATALOG,
      ),
    ).toEqual([
      { vendor: "deepseek", models: ["DeepSeek-V3-0324"] },
      { vendor: "deepseek-ai", models: ["deepseek-ai/DeepSeek-R1"] },
      { vendor: "meta-llama", models: ["meta-llama/Llama-3.3-70B"] },
      { vendor: OTHER_VENDOR, models: ["mystery-1"] },
    ])
  })
})
