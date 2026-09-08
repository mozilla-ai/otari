import { describe, expect, it } from "vitest"

import type { WireBootstrap } from "@/shared/helpers/bootstrap"
import { normalizeBootstrap } from "@/shared/helpers/bootstrap"
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
      mail_ready: true,
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
        "mail_ready",
        "passkeys_ready",
        "maintenance_mode",
        "docs_url",
        "management_url",
        "data_plane_url",
        "terms_url",
        "privacy_url",
      ),
    )

    expect(completed.mail_ready).toBe(false)
    expect(completed.passkeys_ready).toBe(false)
    expect(completed.maintenance_mode).toBe(false)
    expect(completed.docs_url).toBeNull()
    expect(completed.management_url).toBeNull()
    expect(completed.data_plane_url).toBeNull()
    expect(completed.terms_url).toBeNull()
    expect(completed.privacy_url).toBeNull()
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

  it("does not invent the two fields that say what the deployment is", () => {
    // web/AGENTS.md: do not guess a deployment. Absent here means absent below,
    // rather than a deployment identity this file made up.
    const completed = normalizeBootstrap(
      omitting("deployment_type", "session_type"),
    )

    expect(completed.deployment_type).toBeUndefined()
    expect(completed.session_type).toBeUndefined()
  })
})
