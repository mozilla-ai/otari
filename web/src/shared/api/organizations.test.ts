import { describe, expect, it } from "vitest"

import { ApiError } from "@/shared/api/client"
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

  it("reads a 404 as no organization, not as a failed read", () => {
    // The one failure that is an answer: an unavailable context fails open
    // onto the operator surface, and a caller in no organization operates
    // nothing.
    expect(
      deploymentOperatorAnswer({
        error: new ApiError(404, "Organization not found"),
        isFetched: true,
      }),
    ).toBe("no-organization")
    expect(
      deploymentOperatorAnswer({
        error: new ApiError(500, "Internal server error"),
        isFetched: true,
      }),
    ).toBe("unavailable")
  })
})
