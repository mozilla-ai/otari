import { describe, expect, it } from "vitest"

import {
  AVAILABILITY_ORDER,
  availabilityFields,
  availabilityOf,
} from "@/features/tools/mcpAvailability"

describe("availability", () => {
  it("maps each choice onto the two stored fields and back", () => {
    for (const choice of AVAILABILITY_ORDER) {
      expect(availabilityOf(availabilityFields(choice))).toBe(choice)
    }
  })

  it("reads a disabled server as off, attached or not", () => {
    expect(availabilityOf({ enabled: false, attach_by_default: true })).toBe(
      "off",
    )
    expect(availabilityFields("off")).toEqual({
      enabled: false,
      attach_by_default: false,
    })
  })
})
