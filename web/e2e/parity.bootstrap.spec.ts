import { expect, test } from "@playwright/test"

// The shell reads /v1/bootstrap before it renders anything, so every other spec
// here already depends on it answering: a failure paints an error banner in
// place of the whole dashboard. What is asserted here is the contract itself,
// against the real gateway. Readable without a session, because whether a
// sign-in screen is the right first paint is decided from this response.
test("the deployment bootstrap is served unauthenticated", async ({
  request,
}) => {
  const response = await request.get("/v1/bootstrap")

  expect(response.status()).toBe(200)
  expect(await response.json()).toEqual({
    deployment_type: "standalone",
    session_type: "local_operator",
    // Spelled out rather than derived: this is the list the sidebar gates on,
    // and a surface that quietly appears or disappears should fail here.
    surfaces: [
      "budgets",
      "keys",
      "models",
      "organizations",
      "providers",
      "routing",
      "settings",
      "tools",
      "usage",
      "users",
      "workspaces",
    ],
    management_url: null,
  })
})
