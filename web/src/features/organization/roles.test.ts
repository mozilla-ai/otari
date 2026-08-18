import { describe, expect, it } from "vitest"

import {
  activeOwnerCount,
  canManage,
  isOwner,
  memberLabel,
  membershipChangeBlockedReason,
} from "@/features/organization/roles"
import { organizationContext, organizationMember } from "@/tests/fixtures"

describe("tenancy roles", () => {
  it("treats owners and admins as managers and nobody else", () => {
    expect(canManage(organizationContext({ role: "owner" }))).toBe(true)
    expect(canManage(organizationContext({ role: "admin" }))).toBe(true)
    expect(canManage(organizationContext({ role: "member" }))).toBe(false)
    expect(canManage(organizationContext({ role: "viewer" }))).toBe(false)
    // A context that has not loaded is not a manager: the controls stay off
    // until the server has said what the caller is.
    expect(canManage(undefined)).toBe(false)
  })

  it("reserves ownership for the owner role", () => {
    expect(isOwner(organizationContext({ role: "owner" }))).toBe(true)
    expect(isOwner(organizationContext({ role: "admin" }))).toBe(false)
  })

  it("counts only active owners", () => {
    expect(
      activeOwnerCount([
        organizationMember({ role: "owner", status: "active" }),
        organizationMember({ role: "owner", status: "suspended" }),
        organizationMember({ role: "admin", status: "active" }),
      ]),
    ).toBe(1)
  })
})

describe("membershipChangeBlockedReason", () => {
  const owner = organizationMember({
    organization_member_id: "owner-membership",
    role: "owner",
  })
  const member = organizationMember({
    organization_member_id: "member-membership",
    role: "member",
  })

  it("refuses a caller who cannot manage the organization", () => {
    expect(
      membershipChangeBlockedReason({
        member,
        context: organizationContext({ role: "viewer" }),
        members: [owner, member],
      }),
    ).toMatch(/owners and admins/)
  })

  it("refuses an admin acting on an owner", () => {
    // Only an owner outranks an owner, which is what the server enforces.
    expect(
      membershipChangeBlockedReason({
        member: owner,
        context: organizationContext({ role: "admin" }),
        members: [owner, member],
      }),
    ).toMatch(/Only an owner/)
  })

  it("refuses the change that would leave no active owner", () => {
    expect(
      membershipChangeBlockedReason({
        member: owner,
        context: organizationContext({ role: "owner" }),
        members: [owner, member],
      }),
    ).toMatch(/last active owner/)
  })

  it("allows an owner to act on another owner once there are two", () => {
    const second = organizationMember({
      organization_member_id: "second-owner",
      role: "owner",
    })
    expect(
      membershipChangeBlockedReason({
        member: owner,
        context: organizationContext({ role: "owner" }),
        members: [owner, second],
      }),
    ).toBeUndefined()
  })

  it("refuses a row that has no membership behind it yet", () => {
    // An invitation is a roster row with no organization_member_id; the
    // endpoints that change a membership have nothing to address.
    expect(
      membershipChangeBlockedReason({
        member: organizationMember({
          organization_member_id: null,
          invitation_id: "pending",
          role: "member",
        }),
        context: organizationContext({ role: "owner" }),
        members: [owner],
      }),
    ).toMatch(/pending invitation/)
  })
})

describe("memberLabel", () => {
  it("prefers a name, then an address, then the identity", () => {
    expect(memberLabel(organizationMember({ full_name: "Ada" }))).toBe("Ada")
    expect(
      memberLabel(
        organizationMember({ full_name: null, email: "ada@example.com" }),
      ),
    ).toBe("ada@example.com")
    // The standalone operator has neither, and an opaque UUID is worse than a
    // short prefix of it.
    expect(
      memberLabel(
        organizationMember({
          full_name: null,
          email: null,
          user_id: "abcdef01-2345-6789-abcd-ef0123456789",
        }),
      ),
    ).toBe("Identity abcdef01")
  })
})
