import { describe, expect, it } from "vitest"

import type { BuiltInGuardrailCatalog } from "@/client"
import {
  credentialsLabel,
  definitionChecks,
  guardrailLabel,
  ifItCantRunLabel,
  mandatesFor,
  mandatesOn,
  runsOnLabel,
} from "@/features/guardrails/guardrailRows"
import {
  organizationGuardrail,
  organizationGuardrailDefinition,
} from "@/tests/fixtures"

const CATALOG = {
  guardrails: [
    {
      guardrail_name: "lakera_guard",
      display_name: "Lakera Guard",
      vendor: "Lakera",
      categories: ["prompt_injection", "general_judge", "pii"],
    },
  ],
} as unknown as BuiltInGuardrailCatalog

const lakera = organizationGuardrailDefinition({
  id: "d1",
  name: "prod-lakera",
})

describe("guardrailLabel", () => {
  it("names the product and its maker", () => {
    expect(guardrailLabel(lakera, CATALOG)).toBe("Lakera Guard · Lakera")
  })

  it("falls back to the stored class when the catalog does not list it", () => {
    expect(guardrailLabel(lakera, { guardrails: [] })).toBe("lakera_guard")
  })
})

describe("definitionChecks", () => {
  it("lists the checks as the picker offers them", () => {
    expect(definitionChecks(lakera, CATALOG)).toBe(
      "Personally identifiable information, Prompt injection",
    )
  })
})

describe("credentialsLabel", () => {
  it("says which secrets are stored, never what they are", () => {
    expect(credentialsLabel(lakera)).toBe("Api key set")
  })

  it("says when the stored ones cannot be read", () => {
    expect(
      credentialsLabel(
        organizationGuardrailDefinition({
          create_secrets: {},
          secrets_decryptable: false,
        }),
      ),
    ).toBe("unreadable")
  })

  it("says when there are none", () => {
    expect(
      credentialsLabel(organizationGuardrailDefinition({ create_secrets: {} })),
    ).toBe("none")
  })
})

describe("runsOnLabel", () => {
  it("names the definition a mandate runs", () => {
    expect(
      runsOnLabel(organizationGuardrail({ definition_id: "d1" }), [lakera]),
    ).toBe("prod-lakera")
  })

  it("names the host of a mandate's own endpoint", () => {
    expect(
      runsOnLabel(
        organizationGuardrail({
          url: "https://guardrails.example:8443/validate",
        }),
        [],
      ),
    ).toBe("guardrails.example:8443")
  })

  it("says the deployment's service when the mandate names neither", () => {
    expect(runsOnLabel(organizationGuardrail(), [])).toBe(
      "the deployment guardrails service",
    )
  })
})

describe("ifItCantRunLabel", () => {
  it("reads a blocking mandate's stored choice", () => {
    expect(
      ifItCantRunLabel(
        organizationGuardrail({ mode: "block", on_unavailable: "block" }),
      ),
    ).toBe("Refuse the request")
  })

  it("says a monitoring mandate always serves, whatever it stored", () => {
    expect(
      ifItCantRunLabel(
        organizationGuardrail({ mode: "monitor", on_unavailable: "block" }),
      ),
    ).toBe("Serve it unchecked")
  })
})

describe("mandatesFor", () => {
  it("keeps the mandates that cover the workspace, by name or by covering all", () => {
    const mandates = [
      organizationGuardrail({
        id: "a",
        profile: "named",
        workspace_ids: ["w1"],
      }),
      organizationGuardrail({
        id: "b",
        profile: "everywhere",
        applies_to_all_workspaces: true,
      }),
      organizationGuardrail({
        id: "c",
        profile: "elsewhere",
        workspace_ids: ["w2"],
      }),
      organizationGuardrail({ id: "d", profile: "nowhere", workspace_ids: [] }),
    ]
    expect(mandatesFor("w1", mandates).map((m) => m.profile)).toEqual([
      "named",
      "everywhere",
    ])
  })
})

describe("mandatesOn", () => {
  it("finds the mandates that point at a definition", () => {
    const mandates = [
      organizationGuardrail({ id: "a", profile: "a", definition_id: "d1" }),
      organizationGuardrail({ id: "b", profile: "b" }),
    ]
    expect(mandatesOn(lakera, mandates).map((m) => m.profile)).toEqual(["a"])
  })
})
