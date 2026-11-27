import { describe, expect, it } from "vitest"

import type { MailReadiness } from "@/client"
import type { WireBootstrap } from "@/shared/helpers/bootstrap"
import {
  mailCanSend,
  mailDelivers,
  normalizeBootstrap,
} from "@/shared/helpers/bootstrap"
import { bootstrap } from "@/tests/fixtures"

// A payload from a gateway that predates a field: the key is absent, not null.
// Built by deleting rather than by writing a literal, so the case stays the one
// the wire produces even as the bootstrap grows fields.
function omitting(...fields: string[]): WireBootstrap {
  const wire = { ...bootstrap() } as Record<string, unknown>
  for (const field of fields) {
    delete wire[field]
  }
  return wire as WireBootstrap
}

describe("normalizeBootstrap", () => {
  it("leaves a current gateway's answer alone", () => {
    const current = bootstrap({
      surfaces: ["settings"],
      sign_in_methods: ["password"],
      oauth_providers: ["github"],
      docs_url: "https://docs.example.com",
      mail: "delivers",
    })

    expect(normalizeBootstrap(current)).toEqual(current)
  })

  it("empties a collection the gateway did not publish", () => {
    const completed = normalizeBootstrap(
      omitting("surfaces", "sign_in_methods", "oauth_providers"),
    )

    expect(completed.surfaces).toEqual([])
    expect(completed.sign_in_methods).toEqual([])
    expect(completed.oauth_providers).toEqual([])
  })

  it("reads an absent flag as off and an absent link as none", () => {
    const completed = normalizeBootstrap(
      omitting(
        "mail",
        "feedback_enabled",
        "federation_member_rules",
        "passkeys_ready",
        "maintenance_mode",
        "docs_url",
        "terms_url",
        "privacy_url",
      ),
    )

    expect(completed.feedback_enabled).toBe(false)
    expect(completed.federation_member_rules).toBe(false)
    expect(completed.mail).toBe("none")
    expect(completed.passkeys_ready).toBe(false)
    expect(completed.maintenance_mode).toBe(false)
    expect(completed.docs_url).toBeNull()
    expect(completed.terms_url).toBeNull()
    expect(completed.privacy_url).toBeNull()
  })

  it("reads an absent switch over an existing feature as on", () => {
    // The two flags that default the other way, because a gateway old enough to
    // omit either had no way to turn the feature off. Reading absence as off
    // would withdraw a page and a sign-in option from a deployment serving both.
    const completed = normalizeBootstrap(
      omitting("passkeys_enabled", "organization_oidc_enabled"),
    )

    expect(completed.passkeys_enabled).toBe(true)
    expect(completed.organization_oidc_enabled).toBe(true)
  })

  it("keeps a false flag rather than treating it as absent", () => {
    // A published `false` and a published empty list are answers, not absences,
    // and neither may be overwritten by the default that stands in for one.
    const completed = normalizeBootstrap(
      bootstrap({ maintenance_mode: false, sign_in_methods: [] }),
    )

    expect(completed.maintenance_mode).toBe(false)
    expect(completed.sign_in_methods).toEqual([])
  })

  it("leaves an older deployment reading as it did before `mail` existed", () => {
    // Both questions were gated on the retired flag there, so this mapping
    // changes nothing for such a deployment. The narrower reading would be a
    // worse answer on every delivering one, which is most of them.
    const older: WireBootstrap = { ...omitting("mail"), mail_ready: true }

    expect(normalizeBootstrap(older).mail).toBe("delivers")
  })

  it("reads an older deployment's retired mail flag as off when it was off", () => {
    const older: WireBootstrap = { ...omitting("mail"), mail_ready: false }

    expect(normalizeBootstrap(older).mail).toBe("none")
  })
})

describe("mail predicates", () => {
  it("answers both questions from one published value", () => {
    const states: MailReadiness[] = ["none", "sends_only", "delivers"]

    expect(states.map((state) => mailCanSend(state))).toEqual([
      false,
      true,
      true,
    ])
    expect(states.map((state) => mailDelivers(state))).toEqual([
      false,
      false,
      true,
    ])
  })
})
