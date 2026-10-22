import { expect, test } from "@playwright/test"
import { API_ROOT } from "@/shared/api/client"
import { authHeaders, expectOk, gotoRoute } from "./helpers"

// An organization admin, not the operator: federation_member_rules (on in
// e2e/otari.yml) is what lets an owner or admin give their own key a rule
// naming any public issuer.
test("an organization admin gives their own key a federation rule and reads its setup", async ({
  browser,
  page,
}) => {
  const email = `workload-admin-${Date.now()}@example.com`
  const password = "workload-password"

  const workspaces = await page.request.get(
    `${API_ROOT}/workspaces?skip=0&limit=1`,
    { headers: authHeaders },
  )
  await expectOk(workspaces, "list workspaces")
  const [workspace] = ((await workspaces.json()) as { data: { id: string }[] })
    .data
  const invited = await page.request.post(
    `${API_ROOT}/organizations/me/member-invitations`,
    {
      headers: authHeaders,
      data: {
        email,
        role: "admin",
        workspace_assignments: [{ workspace_id: workspace.id }],
      },
    },
  )
  await expectOk(invited, "invite the member")
  const { accept_link } = (await invited.json()) as { accept_link: string }
  const token = new URL(accept_link, "http://x").hash.split("token=")[1]
  await expectOk(
    await page.request.post(`${API_ROOT}/invitations/accept`, {
      data: { token, password },
    }),
    "accept the invitation",
  )

  const member = await browser.newContext()
  const memberPage = await member.newPage()
  await memberPage.goto("/")
  await memberPage.getByLabel("Email").fill(email)
  await memberPage.getByLabel("Password").fill(password)
  await memberPage.getByRole("button", { name: "Sign in" }).click()
  await expect(
    memberPage.getByRole("navigation", { name: "Sidebar" }),
  ).toBeVisible()
  await expectOk(
    await memberPage.request.post(`${API_ROOT}/organizations/me/keys`, {
      data: { key_name: "hackbot" },
    }),
    "create the member's key",
  )

  await gotoRoute(memberPage, "/keys")
  await memberPage.getByRole("button", { name: "Actions for hackbot" }).click()
  await memberPage.getByRole("menuitem", { name: /^Workload identity/ }).click()
  const dialog = memberPage.getByRole("dialog", { name: "Workload identity" })
  await expect(dialog.getByText("No rules yet.")).toBeVisible()

  await dialog
    .getByLabel("Service account email")
    .fill("hackbot@my-project.iam.gserviceaccount.com")
  await dialog.getByRole("button", { name: "Add rule" }).click()

  const setup = dialog.getByRole("region", { name: "Environment code" })
  await expect(setup).toContainText("ANTHROPIC_FEDERATION_RULE_ID=")
  await expect(setup).toContainText("ANTHROPIC_ORGANIZATION_ID=")
  await expect(setup).toContainText(
    `ANTHROPIC_BASE_URL="${new URL(memberPage.url()).origin}/api"`,
  )

  await dialog
    .getByRole("button", {
      name: "Delete rule for hackbot@my-project.iam.gserviceaccount.com",
    })
    .click()
  await memberPage
    .getByRole("alertdialog")
    .getByRole("button", { name: "Delete rule" })
    .click()
  await expect(dialog.getByText("No rules yet.")).toBeVisible()
  await member.close()
})
