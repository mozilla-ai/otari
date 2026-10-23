import { expect, test } from "@playwright/test"
import { API_ROOT, DASHBOARD_BUILD_PATH } from "@/shared/api/client"

// The shell reads /v1/bootstrap before it renders anything, so every other spec
// here already depends on it answering: a failure paints an error banner in
// place of the whole dashboard. What is asserted here is the contract itself,
// against the real gateway. Readable without a session, because whether a
// sign-in screen is the right first paint is decided from this response.
test("the deployment bootstrap is served unauthenticated", async ({
  request,
}) => {
  const response = await request.get(`${API_ROOT}/bootstrap`)

  expect(response.status()).toBe(200)
  expect(await response.json()).toEqual({
    // Spelled out rather than derived: this is the list the sidebar gates on,
    // and a surface that quietly appears or disappears should fail here.
    surfaces: [
      "admin",
      "budgets",
      "health_checks",
      "hosted_providers",
      "keys",
      "models",
      "organization_guardrails",
      "organization_providers",
      "organization_usage",
      "organizations",
      "playground",
      "pricing",
      "routing",
      "settings",
      "tools",
      "traces",
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
    // No docs_url in this e2e environment, so the dashboard's Documentation
    // links open docs.otari.ai; see docs/configuration.md#documentation-links.
    docs_url: null,
    // No legal pages configured here; see docs/configuration.md#legal-pages.
    terms_url: null,
    privacy_url: null,
    // No site of its own here, so the public catalog's logo links to the
    // catalog; see docs/configuration.md#the-public-site.
    site_url: null,
    // Sign-ins are open, which is the resting state: maintenance mode is a
    // stored row an operator sets to freeze them during a redeploy, and
    // nothing in this environment sets it.
    maintenance_mode: false,
    // No public_base_url in this e2e environment, so the gateway can derive no
    // WebAuthn relying-party ID and the account page offers no passkey form.
    // Distinct from `passkey` in sign_in_methods above, which additionally
    // needs a registered passkey; see docs/access-control.md.
    passkeys_enabled: true,
    passkeys_ready: false,
    oauth_providers: [],
    feedback_enabled: true,
    // On in e2e/otari.yml, for the workload identity spec.
    federation_member_rules: true,
    // No SMTP configured in this e2e environment, so invitations are
    // creatable but not emailed; see docs/configuration.md#mail.
    mail_ready: false,
    // The catalog stays behind the sign-in screen unless an operator opens it
    // with public_catalog; see docs/configuration.md#a-public-catalog.
    public_catalog: false,
    // Closed, which is the default and is what mail_ready above would force
    // anyway: signup sends a verification link, so a deployment that cannot
    // send one registers nobody.
    open_signup: false,
  })
})

// The stale-tab check, asserted against the real gateway because it is the one
// place the two halves meet: the route is mounted beside the dashboard at the
// gateway's own root, and the poll reaches it through `siteFetch` rather than
// `apiFetch` for exactly that reason. Built the same way the client builds it,
// so a caller moved back under the API root fails here rather than silently
// polling a 404, which is how this went unnoticed from #1026 until now.
test("the dashboard build id is served at the gateway's own root", async ({
  request,
}) => {
  const response = await request.get(DASHBOARD_BUILD_PATH)

  expect(response.status()).toBe(200)
  const body = await response.json()
  expect(typeof body.build).toBe("string")
  expect(body.build.length).toBeGreaterThan(0)
  expect(typeof body.version).toBe("string")
  expect(body.version.length).toBeGreaterThan(0)

  // And not under the API root, which is where it was being asked for.
  expect(
    (await request.get(`${API_ROOT}${DASHBOARD_BUILD_PATH}`)).status(),
  ).toBe(404)
})
