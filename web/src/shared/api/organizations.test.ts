import { describe, expect, it } from "vitest"

import { deploymentOperatorAnswer } from "@/shared/api/organizations"
import { organizationContext } from "@/tests/fixtures"

describe("deploymentOperatorAnswer", () => {
  it("answers from the context once it has landed", () => {
    expect(
      deploymentOperatorAnswer({
        data: organizationContext({ deployment_operator: true }),
        isFetched: true,
      }),
    ).toBe("operator")
    expect(
      deploymentOperatorAnswer({
        data: organizationContext({ deployment_operator: false }),
        isFetched: true,
      }),
    ).toBe("not-operator")
  })

  it("tells a read still in flight from one that failed", () => {
    expect(deploymentOperatorAnswer({ isFetched: false })).toBe("pending")
    // `isFetched` stays true while a failed read retries, which is what keeps
    // a page from swinging back to its spinner on every attempt.
    expect(deploymentOperatorAnswer({ isFetched: true })).toBe("unavailable")
  })
})
