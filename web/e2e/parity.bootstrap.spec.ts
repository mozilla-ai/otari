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
      "pricing",
      "providers",
      "routing",
      "settings",
      "tools",
      "usage",
      "users",
      "workspaces",
    ],
    // The master key, because nothing in this environment has claimed the
    // deployment by setting an operator password. Once something does, the
    // gateway answers ["password"] instead and the master key stops being a
    // sign-in credential while staying the API one; see
    // docs/access-control.md#dashboard-sessions-and-identity.
    sign_in_methods: ["master_key"],
    management_url: null,
    // Sign-ins are open, which is the resting state: maintenance mode is a
    // stored row an operator sets to freeze them during a redeploy, and
    // nothing in this environment sets it.
    maintenance_mode: false,
    // No public_base_url in this e2e environment, so the gateway can derive no
    // WebAuthn relying-party ID and the account page offers no passkey form.
    // Distinct from `passkey` in sign_in_methods above, which additionally
    // needs a registered passkey; see docs/access-control.md.
    passkeys_ready: false,
    // No SMTP configured in this e2e environment, so invitations are
    // creatable but not emailed; see docs/configuration.md#mail.
    mail_ready: false,
  })
})
