import { expect, test } from "@playwright/test"

// Runs in the `hybrid` project, against its own gateway (web/e2e/otari.hybrid.yml,
// booted by web/e2e/serve-hybrid.sh), not the standalone one every other spec
// here uses. That second deployment is the point: hybrid mode is a different
// deployment shape, not a different page, and the shape is decided by the
// server. See web/playwright.config.ts.
//
// A gateway attached to otari.ai is a data-plane runtime. Its organizations,
// credentials, routing, budgets and usage are owned there, so what it may show
// an operator standing at its URL is: is this gateway up, can it reach the
// control plane, and where do I go to change anything. Nothing else.
//
// Two halves of that are already covered elsewhere and are deliberately not
// repeated here: the component renders from a mocked bootstrap in
// HybridLanding.test.tsx, and every management router answering 404 with the
// "manage this on the platform" hint is asserted per path in
// tests/integration/test_hybrid_mode_surface.py. What only this suite can do is
// put the server's real answer and the real bundle in one browser, which is
// where a regression in either alone would still leave both green.
test.describe("hybrid deployment", () => {
  // The link web/e2e/otari.hybrid.yml configures, and what /v1/bootstrap
  // publishes from it.
  const MANAGEMENT_URL = "https://otari.ai"

  // Asserted over HTTP against a running gateway for the same reason the
  // standalone shape is (parity.bootstrap.spec.ts): the shell reads this before
  // it renders anything, so whether a management dashboard is even the right
  // first paint is decided from this response.
  test("the deployment bootstrap offers no surface and no session", async ({
    request,
  }) => {
    const response = await request.get("/v1/bootstrap")

    expect(response.status()).toBe(200)
    // Whole-object, as the standalone assertion is: a field quietly appearing
    // here is what would put a management surface back on this deployment.
    expect(await response.json()).toEqual({
      deployment_type: "hybrid",
      session_type: "none",
      // Empty, not a subset: this gateway mounts none of the management API, so
      // a surface named here would be a page with no server behind it.
      surfaces: [],
      // No session to mint, so no credential to offer. This is why the shell
      // renders the landing page rather than a sign-in screen.
      sign_in_methods: [],
      management_url: MANAGEMENT_URL,
      // Null, and never anything else on a hybrid gateway: this deployment *is*
      // the data plane, so the address that reached it is the address that
      // reaches its API. See otari#823.
      data_plane_url: null,
      // Deployment-wide rather than standalone-only, and unset here: a hybrid
      // gateway may carry a hosted documentation link, and this one configures
      // none, so the bundled guide stays the target.
      docs_url: null,
      // Never frozen, because the freeze is on a sign-in this deployment does
      // not serve: a hybrid gateway mints no session for maintenance mode to
      // refuse. Its control plane owns that, as it owns the sign-in itself.
      maintenance_mode: false,
      // No session of its own to protect, so no ceremony to run either.
      passkeys_ready: false,
      oauth_providers: [],
      // Its control plane sends the mail that carries links back to it.
      mail_ready: false,
    })
  })

  test("the landing page reports health, the connection, and where to manage it", async ({
    page,
    baseURL,
  }) => {
    await page.goto("/")

    await expect(
      page.getByRole("heading", { name: "Otari gateway" }),
    ).toBeVisible()

    // The two conditions an operator is here for, separate on purpose: a dead
    // gateway is this host's problem, an unreachable control plane is a network
    // question. "Connected" is a real answer rather than a fixture: this gateway
    // made an HTTP health request to the address in its config (the standalone
    // gateway next door, standing in for otari.ai) and reported what came back,
    // so the success path of that check is covered and not only its failure.
    await expect(page.getByText("Healthy", { exact: true })).toBeVisible()
    await expect(page.getByText("Connected", { exact: true })).toBeVisible()

    // What a client is pointed at, taken from the URL this page was served from
    // rather than from anything the gateway reports, so a deployment behind a
    // proxy shows the address its operator is actually looking at. That makes it
    // this project's own base URL, which is the one thing a component test
    // rendering in jsdom cannot check.
    await expect(page.getByText(`${baseURL}/v1`, { exact: true })).toBeVisible()

    const link = page.getByRole("link", {
      name: "Manage this gateway on otari.ai",
    })
    await expect(link).toHaveAttribute("href", MANAGEMENT_URL)
    // An outbound link from a page that knows which gateway this is: opening it
    // must hand the control plane neither this deployment's URL nor a handle on
    // this window.
    await expect(link).toHaveAttribute("rel", "noreferrer")
    await expect(link).toHaveAttribute("target", "_blank")
  })

  test("and nothing else: no navigation, no sign-in, no local route", async ({
    page,
  }) => {
    await page.goto("/")
    // The heading rather than a status pill: what this test is about does not
    // depend on the health check, and anchoring on its answer would report an
    // unreachable control plane as a management surface that appeared.
    await expect(
      page.getByRole("heading", { name: "Otari gateway" }),
    ).toBeVisible()

    // Not a shell with its pages hidden: there is no rail to hide them in, and
    // no sign-in screen either, since this deployment issues no session.
    await expect(page.getByRole("navigation")).toHaveCount(0)
    await expect(page.locator('input[type="password"]')).toHaveCount(0)

    // One link, and it leaves. Anything else would be a route into a dashboard
    // this gateway does not serve.
    await expect(page.locator("a")).toHaveCount(1)

    // Asking for a management route by hash lands here too: the shell decides on
    // the deployment ahead of the route table (see DeploymentRoot in
    // web/src/app/App.tsx), so a bookmarked or pasted URL cannot open a page
    // whose data lives on otari.ai. Both a management page and the invitation
    // route, which is the one URL a standalone deployment serves ahead of its
    // own auth gate and which a hybrid gateway holds no membership state to
    // honor.
    for (const route of ["#/providers", "#/accept-invitation?token=whatever"]) {
      await page.goto(`/${route}`)
      await expect(
        page.getByRole("heading", { name: "Otari gateway" }),
      ).toBeVisible()
      await expect(page.getByRole("navigation")).toHaveCount(0)
    }
  })
})
