import { describe, expect, it } from "vitest"
import type { OrganizationMember } from "@/client"
import { personActions } from "@/features/people/personActions"
import { organizationContext, organizationMember } from "@/tests/fixtures"

function member(
  id: string,
  overrides: Partial<OrganizationMember> = {},
): OrganizationMember {
  return organizationMember({
    organization_member_id: `om-${id}`,
    user_id: `user-${id}`,
    role: "member",
    status: "active",
    ...overrides,
  })
}

const context = (role: string, id = "om-me") =>
  organizationContext({ role, organization_member_id: id })

const owner = member("owner", { role: "owner" })
const wei = member("wei")
const me = member("me", { role: "admin" })
const members = [owner, wei, me]

describe("personActions", () => {
  it("offers both to an owner or admin on someone else's row", () => {
    expect(
      personActions({ member: wei, context: context("admin"), members }),
    ).toEqual({
      changeRole: { blocked: undefined },
      remove: { blocked: undefined },
    })
  })

  it("offers nothing to a member or viewer", () => {
    for (const role of ["member", "viewer"]) {
      expect(
        personActions({ member: wei, context: context(role), members }),
      ).toEqual({})
    }
  })

  it("lists what the server would refuse, with its reason", () => {
    const actions = personActions({
      member: owner,
      context: context("admin"),
      members,
    })
    expect(actions.remove?.blocked).toMatch(/Only an owner/)
    expect(actions.changeRole?.blocked).toMatch(/Only an owner/)
  })

  it("never offers removing yourself, and leaves out a refused role change", () => {
    const lastOwner = personActions({
      member: owner,
      context: context("owner", "om-owner"),
      members,
    })
    expect(lastOwner).toEqual({})
    const admin = personActions({
      member: me,
      context: context("admin"),
      members,
    })
    expect(admin).toEqual({ changeRole: {} })
  })

  it("withdraws an invitation and has no role to change on one", () => {
    const actions = personActions({
      member: member("ana", {
        status: "invited",
        organization_member_id: null,
        invitation_id: "inv-ana",
      }),
      context: context("owner"),
      members,
    })
    expect(actions.remove).toEqual({})
    expect(actions.changeRole?.blocked).toMatch(/pending invitation/)
  })
})
