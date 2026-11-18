import { describe, expect, it } from "vitest"
import type {
  BulkInviteOrganizationMembersResult,
  InviteOrganizationMemberResult,
} from "@/client"
import {
  addressProblem,
  inviteLabel,
  needsOutcome,
  outcomeCopy,
  teamHelp,
  teamRoleFor,
} from "@/features/people/invite"

describe("teamRoleFor and teamHelp", () => {
  it("places a viewer as a viewer and everyone else as a member", () => {
    expect(teamRoleFor("viewer")).toBe("viewer")
    expect(teamRoleFor("member")).toBe("member")
    expect(teamRoleFor("admin")).toBe("member")
    expect(teamRoleFor("owner")).toBe("member")
  })

  it("says what the team means for the role picked", () => {
    expect(teamHelp("member")).toBe(
      "Where their keys and sessions go. They join it as a member.",
    )
    expect(teamHelp("viewer")).toBe(
      "Where their keys and sessions go. They join it as a viewer.",
    )
    expect(teamHelp("admin")).toBe(
      "Where their own keys go. Owners and admins see every workspace.",
    )
  })
})

describe("addressProblem", () => {
  it("asks for an address when there is none", () => {
    expect(addressProblem("")).toBe("Enter at least one email address.")
    expect(addressProblem(" ,\n ")).toBe("Enter at least one email address.")
  })

  it("passes addresses separated by commas, spaces or new lines", () => {
    expect(
      addressProblem("ana@northwind.test, li@northwind.test\nwei@x.io"),
    ).toBeUndefined()
  })

  it("names the one address that would not go", () => {
    expect(addressProblem("ana.lima@northwind.test, li.wei@northwind")).toBe(
      "li.wei@northwind is not an email address. Correct it or take it out.",
    )
  })

  it("names every address that would not go", () => {
    expect(addressProblem("ana, li@northwind, wei@x.io")).toBe(
      "ana, li@northwind are not email addresses. Correct them or take them out.",
    )
  })
})

describe("inviteLabel", () => {
  it("counts people past one", () => {
    expect(inviteLabel(0)).toBe("Invite")
    expect(inviteLabel(1)).toBe("Invite")
    expect(inviteLabel(2)).toBe("Invite 2 people")
  })
})

const invited = (
  email: string,
  mail_sent: boolean,
): InviteOrganizationMemberResult => ({
  email,
  mail_sent,
  mail_limited: false,
  invitation_id: `i-${email}`,
  organization_member_id: `om-${email}`,
  role: "member",
  status: "invited",
  created_at: "2026-11-01T00:00:00Z",
  expires_at: "2026-11-08T00:00:00Z",
  accept_link: mail_sent ? null : `/invitations/accept?token=${email}`,
})

const result = (
  sent: InviteOrganizationMemberResult[],
  failed: { email: string; detail: string }[],
): BulkInviteOrganizationMembersResult => ({ invited: sent, failed })

describe("outcome", () => {
  it("just closes when every email went out", () => {
    expect(needsOutcome(result([invited("a@x.io", true)], []))).toBe(false)
  })

  it("says which one went and which did not", () => {
    const outcome = result(
      [invited("ana.lima@northwind.test", true)],
      [{ email: "li.wei@northwind.test", detail: "Already a member." }],
    )
    expect(needsOutcome(outcome)).toBe(true)
    expect(outcomeCopy(outcome)).toEqual({
      title: "1 of 2 invited",
      description:
        "An invitation email is on its way to ana.lima@northwind.test. This address was not invited:",
    })
  })

  it("counts several that went and several that did not", () => {
    const outcome = result(
      [invited("a@x.io", true), invited("b@x.io", true)],
      [
        { email: "c@x.io", detail: "Already a member." },
        { email: "d@x.io", detail: "Blocked." },
      ],
    )
    expect(outcomeCopy(outcome)).toEqual({
      title: "2 of 4 invited",
      description:
        "Invitation emails are on their way to 2 people. These addresses were not invited:",
    })
  })

  it("hands over the links when no mail went", () => {
    const outcome = result([invited("a@x.io", false)], [])
    expect(needsOutcome(outcome)).toBe(true)
    expect(outcomeCopy(outcome)).toEqual({
      title: "1 of 1 invited",
      description: "Share this link with the person it names.",
    })
  })
})
