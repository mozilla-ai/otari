import { describe, expect, it } from "vitest"

import {
  type AccessEntry,
  filterGroups,
  groupByProvider,
  summarizeGroups,
} from "@/features/workspaces/access/providerGroups"

function entry(overrides: Partial<AccessEntry> = {}): AccessEntry {
  return {
    id: "id",
    provider: "openai",
    name: "openai / Production",
    isCustomized: false,
    isOff: false,
    ...overrides,
  }
}

const ENTRIES = [
  entry({ id: "a", provider: "openai", name: "openai / Production" }),
  entry({
    id: "b",
    provider: "anthropic",
    name: "anthropic / team",
    isCustomized: true,
  }),
  entry({ id: "c", provider: "openai", name: "openai / Staging", isOff: true }),
  entry({ id: "d", provider: "mistral", name: "mistral" }),
]

describe("groupByProvider", () => {
  it("folds each provider's sources into one group, in name order", () => {
    expect(
      groupByProvider(ENTRIES).map((group) => [
        group.provider,
        group.entries.map((member) => member.id),
      ]),
    ).toEqual([
      ["anthropic", ["b"]],
      ["mistral", ["d"]],
      ["openai", ["a", "c"]],
    ])
  })
})

describe("filterGroups", () => {
  const groups = groupByProvider(ENTRIES)

  it("keeps a whole group when the query names its provider", () => {
    expect(
      filterGroups(groups, "OPEN", "all").map((group) => group.entries.length),
    ).toEqual([2])
  })

  it("keeps only the key a query names", () => {
    expect(
      filterGroups(groups, "staging", "all").flatMap((group) =>
        group.entries.map((member) => member.id),
      ),
    ).toEqual(["c"])
  })

  it("narrows to what was customized or switched off", () => {
    const ids = (filter: "customized" | "off") =>
      filterGroups(groups, "", filter).flatMap((group) =>
        group.entries.map((member) => member.id),
      )
    expect(ids("customized")).toEqual(["b"])
    expect(ids("off")).toEqual(["c"])
  })
})

describe("summarizeGroups", () => {
  it("counts providers and leaves out what is zero", () => {
    expect(summarizeGroups(groupByProvider(ENTRIES))).toBe(
      "3 providers · 1 customized · 1 off",
    )
    expect(summarizeGroups(groupByProvider([entry()]))).toBe("1 provider")
  })
})
