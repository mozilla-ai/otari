import { describe, expect, it } from "vitest"

import { organizationUsers } from "@/features/users/organizationUsers"
import { apiKey, user } from "@/tests/fixtures"

const MEMBER = "33333333-3333-3333-3333-333333333333"
const OTHER_TENANT = "44444444-4444-4444-4444-444444444444"
const ROSTER = new Map([[MEMBER, "Alice Example"]])

describe("organizationUsers", () => {
  it("keeps the organization's own members", () => {
    expect(
      organizationUsers([user({ user_id: MEMBER })], ROSTER, []).map(
        (u) => u.user_id,
      ),
    ).toEqual([MEMBER])
  })

  it("keeps an id that already owns a key here, member or not", () => {
    // `ci-bot` is nobody on the roster, and dropping it would take the service
    // identities an operator names out of their own picker.
    const users = [user({ user_id: "ci-bot" })]
    const keys = [apiKey({ id: "key-1", user_id: "ci-bot" })]
    expect(
      organizationUsers(users, ROSTER, keys).map((u) => u.user_id),
    ).toEqual(["ci-bot"])
  })

  it("drops another organization's people", () => {
    const users = [user({ user_id: MEMBER }), user({ user_id: OTHER_TENANT })]
    expect(organizationUsers(users, ROSTER, []).map((u) => u.user_id)).toEqual([
      MEMBER,
    ])
  })

  it("ignores a key with no owner rather than matching users with none", () => {
    const users = [user({ user_id: OTHER_TENANT })]
    expect(
      organizationUsers(users, new Map(), [
        apiKey({ id: "key-1", user_id: null }),
      ]),
    ).toEqual([])
  })

  it("offers nothing while the two scoping reads are still empty", () => {
    // Both arrive asynchronously, and an empty scope has to read as "nothing to
    // offer yet" rather than falling open to the deployment's whole list.
    expect(
      organizationUsers([user({ user_id: MEMBER })], new Map(), []),
    ).toEqual([])
  })
})
