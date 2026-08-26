import { describe, expect, it } from "vitest"

import {
  accountLabel,
  accountLockoutReason,
  organizationSummary,
} from "@/features/admin/accounts"
import { deploymentUser } from "@/tests/fixtures"

describe("accountLockoutReason", () => {
  it("allows the change on an ordinary account", () => {
    expect(accountLockoutReason(deploymentUser())).toBeUndefined()
  })

  it("blocks the caller's own row", () => {
    expect(accountLockoutReason(deploymentUser({ is_self: true }))).toContain(
      "your own account",
    )
  })

  it("blocks the bootstrap operator", () => {
    expect(
      accountLockoutReason(deploymentUser({ is_bootstrap_operator: true })),
    ).toContain("bootstrap operator")
  })

  it("names the bootstrap reason first when it is both", () => {
    // Which it is on every standalone deployment today, where the one operator
    // identity is the marked one. The self reason ends in a remedy ("another
    // operator has to make this change") that nobody can carry out on this row,
    // because the server refuses this change from every operator; the bootstrap
    // reason is the one that stays true whoever is reading.
    expect(
      accountLockoutReason(
        deploymentUser({ is_self: true, is_bootstrap_operator: true }),
      ),
    ).toContain("bootstrap operator")
  })
})

describe("accountLabel", () => {
  it("prefers the name", () => {
    expect(accountLabel(deploymentUser())).toBe("Ada Lovelace")
  })

  it("falls back to the address, then to the id", () => {
    expect(accountLabel(deploymentUser({ full_name: null }))).toBe(
      "ada@example.com",
    )
    // A local operator identity has neither, which is the ordinary standalone
    // first-boot row rather than an edge case.
    const bare = deploymentUser({ full_name: null, email: null })
    expect(accountLabel(bare)).toBe(bare.id)
  })

  it("ignores a whitespace-only name", () => {
    expect(accountLabel(deploymentUser({ full_name: "   " }))).toBe(
      "ada@example.com",
    )
  })
})

describe("organizationSummary", () => {
  it("says None for an account in no organization", () => {
    expect(organizationSummary(deploymentUser({ organizations: [] }))).toBe(
      "None",
    )
  })

  it("marks a membership that is not active and leaves an active one plain", () => {
    const account = deploymentUser({
      organizations: [
        {
          organization_id: "1",
          name: "Acme",
          slug: "acme",
          role: "owner",
          status: "active",
        },
        {
          organization_id: "2",
          name: "Umbrella",
          slug: "umbrella",
          role: "member",
          status: "suspended",
        },
      ],
    })

    expect(organizationSummary(account)).toBe(
      "Acme (owner), Umbrella (member, suspended)",
    )
  })
})
