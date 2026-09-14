import { describe, expect, it } from "vitest"

import { departureSummary } from "@/features/workspaces/providerKeyDepartures"
import { workspaceProviderKeyOverride } from "@/tests/fixtures"

describe("departureSummary", () => {
  it("says nothing for a read that has not answered", () => {
    expect(departureSummary(undefined)).toBeUndefined()
  })

  it("says nothing for an organization that holds no keys", () => {
    // Distinct from full inheritance: there is nothing here to depart from.
    expect(departureSummary([])).toBeUndefined()
  })

  it("calls a workspace that changed nothing an inheritor", () => {
    expect(departureSummary([workspaceProviderKeyOverride()])).toEqual({
      text: "Inherits all",
      hasDepartures: false,
    })
  })

  it("counts the three departures separately, in one phrase", () => {
    const summary = departureSummary([
      workspaceProviderKeyOverride({ is_default: true }),
      workspaceProviderKeyOverride({ allowed_models: ["gpt-4o"] }),
      workspaceProviderKeyOverride({ allowed_models: ["claude"] }),
      workspaceProviderKeyOverride({ disabled: true }),
    ])
    expect(summary).toEqual({
      text: "1 always used, 2 narrowed, 1 never used",
      hasDepartures: true,
    })
  })

  it("counts a pinned key that is also narrowed under both", () => {
    // Two independent departures on one key, not one of them chosen over the
    // other: the pin says which key serves, the narrowing which models it does.
    expect(
      departureSummary([
        workspaceProviderKeyOverride({
          is_default: true,
          allowed_models: ["gpt-4o"],
        }),
      ]),
    ).toEqual({ text: "1 always used, 1 narrowed", hasDepartures: true })
  })
})
