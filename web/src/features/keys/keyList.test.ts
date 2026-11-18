import { describe, expect, it } from "vitest"
import type { ApiKey, UsageGroupRow } from "@/client"
import {
  type Caller,
  canAddWorkloadIdentity,
  compareRows,
  countLine,
  type KeyRow,
  keyAccess,
  keyMenu,
  keyRow,
  keyState,
  listsOtherPeoplesKeys,
  parseStateFilter,
  rowsInState,
} from "@/features/keys/keyList"
import { apiKey } from "@/tests/fixtures"

const NOW = Date.parse("2026-11-11T12:00:00Z")

function key(overrides: Partial<ApiKey> = {}): ApiKey {
  return apiKey({
    id: "k1",
    key_name: "claude-code on my-laptop",
    is_active: true,
    expires_at: null,
    disabled_reason: null,
    is_service_key: false,
    is_organization_service_key: false,
    user_id: "me",
    workspace_id: "platform",
    ...overrides,
  })
}

const row = (overrides: Partial<ApiKey>, spend: number | null = 0): KeyRow => {
  const k = key(overrides)
  return { key: k, state: keyState(k, NOW), spend, tokens: 0 }
}

const caller = (
  role: string | undefined,
  workspaces: Record<string, string> = { platform: "member" },
  isOperator = false,
): Caller => ({
  isOperator,
  role,
  workspaceRoles: new Map(Object.entries(workspaces)),
})

const ids = (items: { id: string }[]) => items.map((item) => item.id)

describe("keyState", () => {
  it("reads a switched-off key as Disabled, even past its expiry", () => {
    expect(
      keyState(
        key({ is_active: false, expires_at: "2026-11-01T00:00:00Z" }),
        NOW,
      ),
    ).toBe("disabled")
  })

  it("reads an operator-disabled key as Disabled", () => {
    expect(
      keyState(key({ is_active: false, disabled_reason: "operator" }), NOW),
    ).toBe("disabled")
  })

  it("reads an active key past its expiry as Expired", () => {
    expect(keyState(key({ expires_at: "2026-11-08T00:00:00Z" }), NOW)).toBe(
      "expired",
    )
  })

  it("reads a key with a future expiry, or none, as Active", () => {
    expect(keyState(key({ expires_at: "2026-12-01T00:00:00Z" }), NOW)).toBe(
      "active",
    )
    expect(keyState(key(), NOW)).toBe("active")
  })
})

describe("the State filter", () => {
  const rows = [
    row({ id: "a" }),
    row({ id: "b" }),
    row({ id: "d", is_active: false }),
    row({ id: "e", expires_at: "2026-11-08T00:00:00Z" }),
  ]

  it("opens on Active, and falls back to it for a value it does not know", () => {
    expect(parseStateFilter("")).toBe("active")
    expect(parseStateFilter("revoked")).toBe("active")
    expect(parseStateFilter("all")).toBe("all")
  })

  it("lists the keys in the state picked, or every key", () => {
    expect(rowsInState(rows, "active").map((r) => r.key.id)).toEqual(["a", "b"])
    expect(rowsInState(rows, "disabled").map((r) => r.key.id)).toEqual(["d"])
    expect(rowsInState(rows, "expired").map((r) => r.key.id)).toEqual(["e"])
    expect(rowsInState(rows, "all")).toHaveLength(4)
  })

  it("counts in the words of each filter", () => {
    expect(countLine(rows, "active")).toBe("2 active of 4")
    expect(countLine(rows, "all")).toBe("4 keys, 2 active")
    expect(countLine(rows, "disabled")).toBe("1 disabled")
    expect(countLine(rows, "expired")).toBe("1 expired")
    expect(countLine([rows[0]], "all")).toBe("1 key, 1 active")
  })

  it("orders Active, Expired, Disabled, then spend, then name", () => {
    const sorted = [
      row({ id: "off", is_active: false }, 50),
      row({ id: "old", expires_at: "2026-11-08T00:00:00Z" }, 40),
      row({ id: "b", key_name: "b" }, 1),
      row({ id: "a", key_name: "a" }, 1),
      row({ id: "big" }, 9),
    ].sort(compareRows)
    expect(sorted.map((r) => r.key.id)).toEqual(["big", "a", "b", "old", "off"])
  })

  it("puts a key whose spend is unknown after every key whose spend is known", () => {
    const sorted = [
      row({ id: "a", key_name: "a" }, null),
      row({ id: "b", key_name: "b" }, 1),
      row({ id: "c", key_name: "c" }, 9),
    ].sort(compareRows)
    expect(sorted.map((r) => r.key.id)).toEqual(["c", "b", "a"])
  })
})

describe("a key's spend", () => {
  const usage = new Map([["spent", { cost: 4.2, tokens: 1200 }]])

  it("is what the usage read says, and nothing for a key it has no row for", () => {
    expect(keyRow(key({ id: "spent" }), usage, NOW)).toMatchObject({
      spend: 4.2,
      tokens: 1200,
    })
    expect(keyRow(key({ id: "idle" }), usage, NOW)).toMatchObject({
      spend: 0,
      tokens: 0,
    })
  })

  it("is unknown, not zero, for a key the summary folded into its Other row", () => {
    // Keyed as the keys page keys the summary's rows. The fold has no key and
    // comes last, after any real row with none (the Playground's).
    const rows: UsageGroupRow[] = [
      {
        key: "spent",
        label: null,
        cost: 4.2,
        tokens: 1200,
        requests: 3,
        is_other: false,
      },
      {
        key: null,
        label: null,
        cost: 1,
        tokens: 10,
        requests: 1,
        is_other: false,
      },
      {
        key: null,
        label: null,
        cost: 0.5,
        tokens: 5,
        requests: 9,
        is_other: true,
      },
    ]
    const folded = new Map(rows.map((one) => [one.key, one]))
    expect(keyRow(key({ id: "idle" }), folded, NOW)).toMatchObject({
      spend: null,
      tokens: null,
    })
    expect(keyRow(key({ id: "spent" }), folded, NOW)).toMatchObject({
      spend: 4.2,
    })
    const whole = new Map(rows.slice(0, 2).map((one) => [one.key, one]))
    expect(keyRow(key({ id: "idle" }), whole, NOW)).toMatchObject({
      spend: 0,
      tokens: 0,
    })
  })

  it("is unknown, not zero, until the usage read answers", () => {
    expect(keyRow(key(), null, NOW)).toMatchObject({
      spend: null,
      tokens: null,
    })
  })
})

describe("the Owner column", () => {
  it("shows only when the list holds someone else's key", () => {
    expect(listsOtherPeoplesKeys([key(), key({ user_id: "wei" })], "me")).toBe(
      true,
    )
    expect(listsOtherPeoplesKeys([key()], "me")).toBe(false)
  })

  it("does not count the organization's service keys", () => {
    expect(
      listsOtherPeoplesKeys(
        [key(), key({ user_id: "org-svc", is_organization_service_key: true })],
        "me",
      ),
    ).toBe(false)
  })
})

describe("keyAccess", () => {
  it("lets the deployment operator write every key", () => {
    expect(keyAccess(key({ user_id: "wei" }), caller("owner", {}, true))).toBe(
      "write",
    )
  })

  it("lets an owner or admin write what they are listed", () => {
    expect(keyAccess(key(), caller("owner", {}))).toBe("write")
    expect(keyAccess(key(), caller("admin", {}))).toBe("write")
  })

  it("lets a member write their key in a team they contribute to", () => {
    expect(keyAccess(key(), caller("member"))).toBe("write")
    expect(keyAccess(key(), caller("member", { platform: "admin" }))).toBe(
      "write",
    )
  })

  it("refuses a viewer every write, in either layer", () => {
    expect(keyAccess(key(), caller("viewer"))).toBe("read")
    expect(keyAccess(key(), caller("member", { platform: "viewer" }))).toBe(
      "read",
    )
    expect(keyAccess(key(), caller("viewer", {}))).toBe("read")
  })

  it("refuses an organization viewer every write, an admin of the key's team included", () => {
    expect(keyAccess(key(), caller("viewer", { platform: "admin" }))).toBe(
      "read",
    )
  })

  it("leaves a member removed from the key's team only switching it off or deleting it", () => {
    expect(keyAccess(key(), caller("member", { data: "member" }))).toBe(
      "revoke",
    )
  })

  it("offers nothing until the role is known", () => {
    expect(keyAccess(key(), caller(undefined))).toBe("read")
  })
})

describe("canAddWorkloadIdentity", () => {
  it("is for owners, admins and the operator, where the deployment allows it", () => {
    expect(canAddWorkloadIdentity(caller("owner"), true)).toBe(true)
    expect(canAddWorkloadIdentity(caller("admin"), true)).toBe(true)
    expect(canAddWorkloadIdentity(caller("member", {}, true), true)).toBe(true)
    expect(canAddWorkloadIdentity(caller("member"), true)).toBe(false)
    expect(canAddWorkloadIdentity(caller("viewer"), true)).toBe(false)
    expect(canAddWorkloadIdentity(caller("owner"), false)).toBe(false)
  })
})

describe("keyMenu", () => {
  const menu = (
    k: ApiKey,
    access: "write" | "revoke" | "read",
    flags: { isOperator?: boolean; canFederate?: boolean } = {},
  ) =>
    keyMenu({
      key: k,
      access,
      isOperator: flags.isOperator ?? false,
      canFederate: flags.canFederate ?? false,
    })

  it("gives an owner or admin every action on their own active key, Delete held", () => {
    const items = menu(key(), "write", { canFederate: true })
    expect(ids(items)).toEqual([
      "edit",
      "disable",
      "workload-identity",
      "regenerate",
      "sessions",
      "delete",
    ])
    expect(items.at(-1)).toMatchObject({
      hint: "Disable it first",
      isDisabled: true,
      isDestructive: true,
    })
  })

  it("adds End users on a service key", () => {
    expect(
      ids(menu(key({ is_service_key: true }), "write", { canFederate: true })),
    ).toEqual([
      "edit",
      "disable",
      "end-users",
      "workload-identity",
      "regenerate",
      "sessions",
      "delete",
    ])
  })

  it("gives a member no Workload identity", () => {
    expect(ids(menu(key(), "write"))).toEqual([
      "edit",
      "disable",
      "regenerate",
      "sessions",
      "delete",
    ])
  })

  it("offers Enable and a pressable Delete on a disabled key", () => {
    const items = menu(key({ is_active: false }), "write")
    expect(ids(items)).toEqual([
      "edit",
      "enable",
      "regenerate",
      "sessions",
      "delete",
    ])
    expect(items.find((i) => i.id === "enable")?.isDisabled).toBe(false)
    expect(items.find((i) => i.id === "delete")).toMatchObject({
      hint: "Cannot be undone",
      isDisabled: false,
    })
  })

  it("holds Enable on a key an operator switched off, unless the caller is one", () => {
    const off = key({ is_active: false, disabled_reason: "operator" })
    expect(menu(off, "write").find((i) => i.id === "enable")).toMatchObject({
      hint: "Disabled by an operator",
      isDisabled: true,
    })
    expect(
      menu(off, "write", { isOperator: true }).find((i) => i.id === "enable")
        ?.isDisabled,
    ).toBe(false)
  })

  it("holds Enable on a revoked key, even for an operator", () => {
    const revoked = key({
      is_active: false,
      disabled_reason: "revoked",
      operator_can_enable: false,
    })
    expect(
      menu(revoked, "write", { isOperator: true }).find(
        (i) => i.id === "enable",
      ),
    ).toMatchObject({ hint: "Revoked permanently", isDisabled: true })
  })

  it("gives a viewer only See its sessions, and End users on a service key", () => {
    expect(ids(menu(key(), "read"))).toEqual(["sessions"])
    expect(ids(menu(key({ is_service_key: true }), "read"))).toEqual([
      "end-users",
      "sessions",
    ])
  })

  it("leaves a member removed from the team only Disable and Delete", () => {
    expect(ids(menu(key(), "revoke"))).toEqual([
      "disable",
      "sessions",
      "delete",
    ])
    expect(ids(menu(key({ is_active: false }), "revoke"))).toEqual([
      "sessions",
      "delete",
    ])
  })
})
