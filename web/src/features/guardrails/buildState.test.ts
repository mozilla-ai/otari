import { describe, expect, it } from "vitest"

import type { OrganizationGuardrailDefinition } from "@/client"
import {
  countServedUnchecked,
  definitionHealth,
  mandateConsequence,
} from "@/features/guardrails/buildState"
import { organizationGuardrail } from "@/tests/fixtures"

function definition(
  overrides: Partial<OrganizationGuardrailDefinition> = {},
): OrganizationGuardrailDefinition {
  return {
    id: "d1",
    organization_id: "o1",
    name: "prod-lakera",
    guardrail_name: "lakera_guard",
    enabled: true,
    build_state: "built",
    create_kwargs: {},
    create_secrets: {},
    secrets_decryptable: true,
    created_at: "2026-09-22T00:00:00Z",
    updated_at: "2026-09-22T00:00:00Z",
    ...overrides,
  }
}

const failClosed = { mode: "block", on_unavailable: "block" }
const failOpen = { mode: "block", on_unavailable: "monitor" }

describe("definitionHealth", () => {
  it.each([
    ["built", "running"],
    ["pending", "starting"],
    ["failed", "not_running"],
    ["disabled", "off"],
  ] as const)("reads %s as %s", (buildState, health) => {
    expect(definitionHealth(definition({ build_state: buildState }))).toBe(
      health,
    )
  })

  it("reports unreadable credentials apart from a failed build, since the fix differs", () => {
    expect(
      definitionHealth(
        definition({ build_state: "failed", secrets_decryptable: false }),
      ),
    ).toBe("credentials_unreadable")
  })
})

describe("mandateConsequence", () => {
  const definitions = [
    definition({ id: "built", build_state: "built" }),
    definition({ id: "failed", build_state: "failed" }),
    definition({ id: "off", build_state: "disabled", enabled: false }),
    definition({ id: "pending", build_state: "pending" }),
  ]

  it("says nothing of a mandate whose definition runs", () => {
    expect(
      mandateConsequence(
        organizationGuardrail({ definition_id: "built", ...failClosed }),
        definitions,
      ),
    ).toBe("")
  })

  it("refuses requests when a blocking mandate cannot run and refuses on failure", () => {
    expect(
      mandateConsequence(
        organizationGuardrail({ definition_id: "failed", ...failClosed }),
        definitions,
      ),
    ).toBe("refused")
  })

  it("serves requests unchecked when the mandate fails open", () => {
    expect(
      mandateConsequence(
        organizationGuardrail({ definition_id: "failed", ...failOpen }),
        definitions,
      ),
    ).toBe("unchecked")
  })

  it("serves unchecked for every monitoring mandate, whatever it says on failure", () => {
    expect(
      mandateConsequence(
        organizationGuardrail({
          definition_id: "failed",
          mode: "monitor",
          on_unavailable: "block",
        }),
        definitions,
      ),
    ).toBe("unchecked")
  })

  it("treats a switched-off definition as one that cannot run", () => {
    // Switching a definition off leaves its mandates unevaluable, and a
    // fail-closed one then refuses every request it covers.
    expect(
      mandateConsequence(
        organizationGuardrail({ definition_id: "off", ...failClosed }),
        definitions,
      ),
    ).toBe("refused")
  })

  it("marks nothing while a definition is starting", () => {
    expect(
      mandateConsequence(
        organizationGuardrail({ definition_id: "pending", ...failClosed }),
        definitions,
      ),
    ).toBe("")
  })

  it("marks nothing on a remote mandate, whose reachability is per request", () => {
    expect(
      mandateConsequence(organizationGuardrail({ ...failClosed }), definitions),
    ).toBe("")
  })

  it("marks nothing on a mandate that is itself off", () => {
    expect(
      mandateConsequence(
        organizationGuardrail({
          definition_id: "failed",
          enabled: false,
          ...failClosed,
        }),
        definitions,
      ),
    ).toBe("")
  })
})

describe("countServedUnchecked", () => {
  it("counts mandates, not definitions: one broken definition can open several", () => {
    const definitions = [definition({ id: "failed", build_state: "failed" })]
    const mandates = [
      organizationGuardrail({ id: "a", definition_id: "failed", ...failOpen }),
      organizationGuardrail({ id: "b", definition_id: "failed", ...failOpen }),
      organizationGuardrail({
        id: "c",
        definition_id: "failed",
        ...failClosed,
      }),
    ]
    expect(countServedUnchecked(mandates, definitions)).toBe(2)
  })
})
