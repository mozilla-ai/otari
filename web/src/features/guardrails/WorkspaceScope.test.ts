import { describe, expect, it } from "vitest"

import type { Workspace } from "@/client"
import { scopeLabel } from "@/features/guardrails/WorkspaceScope"
import { organizationGuardrail } from "@/tests/fixtures"

const ALPHA = "11111111-1111-1111-1111-111111111111"
const BETA = "22222222-2222-2222-2222-222222222222"
const WORKSPACES = [
  { id: ALPHA, name: "Alpha" },
  { id: BETA, name: "Beta" },
] as Workspace[]

describe("scopeLabel", () => {
  it("says a mandate on every workspace covers new ones too", () => {
    expect(
      scopeLabel(
        organizationGuardrail({ applies_to_all_workspaces: true }),
        WORKSPACES,
      ),
    ).toBe("Every workspace, including new ones")
  })

  it("says when a mandate runs nowhere yet", () => {
    expect(
      scopeLabel(organizationGuardrail({ workspace_ids: [] }), WORKSPACES),
    ).toBe("No workspaces yet")
  })

  it("names the chosen workspaces, and counts them before their names load", () => {
    const mandate = organizationGuardrail({ workspace_ids: [ALPHA, BETA] })
    expect(scopeLabel(mandate, WORKSPACES)).toBe("Alpha, Beta")
    expect(scopeLabel(mandate, [])).toBe("2 workspaces")
  })
})
