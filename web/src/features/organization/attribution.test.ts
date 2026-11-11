import { describe, expect, it } from "vitest"

import { organizationMember } from "@/tests/fixtures"

import { attributionId, memberLabelsByAttributionId } from "./attribution"

describe("attributionId", () => {
  it("is the member's request-plane id once they have one", () => {
    expect(
      attributionId(
        organizationMember({
          user_id: "identity-bo",
          attribution_user_id: "bo",
        }),
      ),
    ).toBe("bo")
  })

  it("is the id the row will take for a member with none yet", () => {
    const keyless = organizationMember({
      user_id: "identity-ana",
      attribution_user_id: null,
      full_name: "Ana",
    })

    expect(attributionId(keyless)).toBe("identity-ana")
    expect(memberLabelsByAttributionId([keyless]).get("identity-ana")).toBe(
      "Ana",
    )
  })
})
