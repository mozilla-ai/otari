import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { Outlet } from "@tanstack/react-router"
import { screen, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, describe, expect, it, vi } from "vitest"

import { WorkspaceSwitcher } from "@/app/nav/WorkspaceSwitcher"
import { SelectedWorkspaceProvider } from "@/shared/hooks/SelectedWorkspace"
import {
  BASE_CAPABILITIES,
  EntitlementProvider,
} from "@/shared/hooks/useEntitlements"
import { organizationContext, workspace } from "@/tests/fixtures"
import { renderWithRouter } from "@/tests/router"

const CREATED_ID = "77777777-7777-7777-7777-777777777777"
// The form's own hold, mirrored rather than imported: it is a private constant,
// and a test that waited on the real one would pass even if it were removed.
const ENTER_HOLD_MS = 800

function jsonResponse(body: unknown): Response {
  return new Response(JSON.stringify(body), {
    status: 200,
    headers: { "Content-Type": "application/json" },
  })
}

// The gateway as this flow meets it: one workspace to start with, and a context
// that reports the new membership only once the workspace has been created,
// which is what the real one does and what the selection has to survive.
function mockApi() {
  let created = false
  vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
    const url = String(input)
    const method = (init?.method ?? "GET").toUpperCase()
    if (url.includes("/v1/budgets")) return jsonResponse([])
    if (url.includes("/v1/workspaces") && method === "POST") {
      created = true
      return jsonResponse(workspace({ id: CREATED_ID, name: "Staging" }))
    }
    return jsonResponse(
      organizationContext({
        workspace_memberships: [
          {
            workspace_id: workspace().id,
            name: "Default Workspace",
            role: "owner",
          },
          ...(created
            ? [{ workspace_id: CREATED_ID, name: "Staging", role: "owner" }]
            : []),
        ],
      }),
    )
  })
}

// Started away from "/" so the landing is observable, and mounted in the shell
// rather than in the page: the switcher is chrome in the real tree, and one
// rendered as the page's own content would unmount the moment it navigated.
function renderSwitcher() {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  })
  return renderWithRouter(<div>USAGE PAGE</div>, {
    url: "/usage",
    routes: [{ path: "/", element: <div>OVERVIEW PAGE</div> }],
    shell: (
      <QueryClientProvider client={client}>
        <EntitlementProvider
          value={{ capabilities: BASE_CAPABILITIES, isLoading: false }}
        >
          <SelectedWorkspaceProvider>
            <WorkspaceSwitcher collapsed={false} />
            {/* Where the routes render, as they do under the real shell. */}
            <Outlet />
          </SelectedWorkspaceProvider>
        </EntitlementProvider>
      </QueryClientProvider>
    ),
  })
}

afterEach(() => {
  vi.restoreAllMocks()
})

// Open the switcher, open the create form from it, and name the workspace. The
// press is left to the test, because what each one is about is what happens
// after it.
async function fillCreateForm(user: ReturnType<typeof userEvent.setup>) {
  await user.click(
    await screen.findByRole("button", { name: /^Switch workspace/ }),
  )
  await user.click(
    within(await screen.findByRole("dialog")).getByRole("button", {
      name: "Create workspace",
    }),
  )
  const form = await screen.findByRole("dialog", { name: "Create workspace" })
  await user.type(within(form).getByLabelText(/^Name/), "Staging")
  return form
}

describe("WorkspaceSwitcher create flow", () => {
  it("enters the workspace it just created", async () => {
    mockApi()
    const user = userEvent.setup()
    await renderSwitcher()

    const form = await fillCreateForm(user)
    // Nothing in the button at rest: the spinner arrives in the label's place,
    // so a resting slot for it would only leave dead space.
    expect(
      within(form)
        .getByRole("button", { name: /Create and open/ })
        .querySelector("[data-slot='spinner']"),
    ).toBeNull()

    // The label names the navigation, which is the only warning the operator
    // gets that the page is about to change under them.
    await user.click(
      within(form).getByRole("button", { name: /Create and open/ }),
    )

    // The press is acknowledged before the page moves, rather than the create
    // landing them somewhere else with nothing in between. Asserted as "still
    // here", because the hold is what makes the spinner visible at all: without
    // it this state is too short to be seen or tested.
    expect(screen.queryByText("OVERVIEW PAGE")).toBeNull()
    // Still found by its name while pending: the label is faded rather than
    // hidden, so the button does not go unnamed for the length of the wait.
    const submit = within(form).getByRole("button", { name: /Create and open/ })
    expect(submit).toHaveAttribute("data-pending", "true")
    // The spinner takes the label's place, in the label's color rather than the
    // default accent, which on this variant is the button's own fill.
    const spinner = submit.querySelector("[data-slot='spinner']")
    expect(spinner).not.toBeNull()
    expect(spinner?.className).toContain("spinner--current")
    expect(submit.querySelector("span:not([data-slot='spinner'])")).toHaveClass(
      "opacity-0",
    )

    // Both halves of "entering" it: the shell's scope moved to the new
    // workspace, and the flow ended on the page that scope reads, rather than
    // back on the page the modal was opened from. The timeout clears the hold
    // with room to spare.
    expect(
      await screen.findByText("OVERVIEW PAGE", undefined, { timeout: 3000 }),
    ).toBeInTheDocument()
    expect(
      await screen.findByRole("button", {
        name: /^Switch workspace, currently Staging/,
      }),
    ).toBeInTheDocument()
  })

  it("does not enter a workspace the operator dismissed the form over", async () => {
    mockApi()
    const user = userEvent.setup()
    await renderSwitcher()

    const form = await fillCreateForm(user)
    await user.click(
      within(form).getByRole("button", { name: /Create and open/ }),
    )
    // Escape rather than Cancel: Cancel is disabled while the create is in
    // flight, so dismissal is what is left, and it is the path that bypasses
    // every button.
    await user.keyboard("{Escape}")

    // Past the hold, so this is the completion having been dropped rather than
    // not having run yet. The workspace was created and the switcher will list
    // it; what must not happen is being taken there after saying not to.
    await new Promise((resolve) => {
      setTimeout(resolve, ENTER_HOLD_MS * 2)
    })
    expect(screen.queryByText("OVERVIEW PAGE")).toBeNull()
    expect(screen.getByText("USAGE PAGE")).toBeInTheDocument()
    expect(
      screen.getByRole("button", { name: /^Switch workspace/ }),
    ).toHaveAccessibleName(/currently Default Workspace/)
  })
})
