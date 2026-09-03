import { readFileSync } from "node:fs"
import { join } from "node:path"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { render, screen, waitFor } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, describe, expect, it, vi } from "vitest"
import type { WorkspaceCodeExecutionPolicy } from "@/client"
import {
  MAX_EXEC_TIMEOUT_S,
  MAX_ITERATIONS,
  WorkspaceCodeExecutionPolicyCard,
} from "@/features/tools/WorkspaceCodeExecutionPolicyCard"
import { SelectedWorkspaceProvider } from "@/shared/hooks/SelectedWorkspace"
import {
  organizationContext,
  workspaceCodeExecutionPolicy,
} from "@/tests/fixtures"
import { pickOption, selectTrigger } from "@/tests/select"

const ALPHA = "11111111-1111-1111-1111-111111111111"

function mockApi({
  memberships = [{ workspace_id: ALPHA, name: "Alpha", role: "admin" }],
  policy = workspaceCodeExecutionPolicy({ workspace_id: ALPHA }),
}: {
  memberships?: { workspace_id: string; name: string; role: string }[]
  policy?: WorkspaceCodeExecutionPolicy
} = {}) {
  const calls: { url: string; method: string; body: unknown }[] = []
  vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
    const url = String(input)
    const method = init?.method ?? "GET"
    if (url.includes("/code-execution-policy")) {
      calls.push({
        url,
        method,
        body:
          typeof init?.body === "string" ? JSON.parse(init.body) : init?.body,
      })
      return Response.json(policy)
    }
    return Response.json(
      organizationContext({ workspace_memberships: memberships }),
    )
  })
  return calls
}

// The form hydrates from the policy once it arrives, so a test that types
// before then would have its input overwritten by the load. The Save button is
// disabled while the query is in flight, which is the signal to wait on.
async function renderLoaded() {
  renderCard()
  await waitFor(() =>
    expect(screen.getByRole("button", { name: "Save" })).toBeEnabled(),
  )
}

function renderCard() {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  })
  return render(
    <QueryClientProvider client={client}>
      <SelectedWorkspaceProvider>
        <WorkspaceCodeExecutionPolicyCard onSaved={() => {}} />
      </SelectedWorkspaceProvider>
    </QueryClientProvider>,
  )
}

describe("WorkspaceCodeExecutionPolicyCard", () => {
  afterEach(() => {
    vi.restoreAllMocks()
    window.localStorage.clear()
  })

  it("reads an unconfigured workspace as using the deployment default", async () => {
    mockApi()
    renderCard()

    expect(await screen.findByText("NO POLICY SET")).toBeInTheDocument()
    expect(selectTrigger("Code execution")).toHaveTextContent(
      "Deployment default",
    )
  })

  it("shows a stored policy's stance and limits", async () => {
    mockApi({
      policy: workspaceCodeExecutionPolicy({
        workspace_id: ALPHA,
        configured: true,
        enabled: false,
        max_iterations: 3,
        exec_timeout_s: 12,
      }),
    })
    await renderLoaded()

    expect(selectTrigger("Code execution")).toHaveTextContent("Blocked")
    expect(screen.getByLabelText("Max tool-loop iterations")).toHaveValue("3")
    expect(screen.getByLabelText("Execution timeout (seconds)")).toHaveValue(
      "12",
    )
    expect(screen.queryByText("No policy set")).not.toBeInTheDocument()
  })

  it("saves the stance and the limits the operator typed", async () => {
    const calls = mockApi()
    const user = userEvent.setup()
    await renderLoaded()

    await pickOption(user, "Code execution", "Allowed")
    await user.type(screen.getByLabelText("Max tool-loop iterations"), "4")
    await user.click(screen.getByRole("button", { name: "Save" }))

    const put = calls.find((call) => call.method === "PUT")
    expect(put?.body).toEqual({
      enabled: true,
      default_purpose_hint: null,
      max_iterations: 4,
      exec_timeout_s: null,
      image: null,
      tools: null,
    })
  })

  it("offers only the images the operator approved, plus the deployment default", async () => {
    mockApi({
      policy: workspaceCodeExecutionPolicy({
        workspace_id: ALPHA,
        allowed_images: ["mzdotai/otari-sandbox-container:latest"],
      }),
    })
    await renderLoaded()

    await userEvent.setup().click(selectTrigger("Sandbox image"))
    expect(
      screen.getAllByRole("option").map((option) => option.textContent),
    ).toEqual(["Deployment default", "mzdotai/otari-sandbox-container:latest"])
  })

  it("says so rather than showing a picker when the operator approved no images", async () => {
    mockApi()
    await renderLoaded()

    expect(
      screen.queryByRole("button", { name: /Sandbox image$/ }),
    ).not.toBeInTheDocument()
    expect(screen.getByText(/approved no sandbox images/i)).toBeInTheDocument()
  })

  it("saves the image the operator chose", async () => {
    const calls = mockApi({
      policy: workspaceCodeExecutionPolicy({
        workspace_id: ALPHA,
        allowed_images: ["mzdotai/otari-sandbox-container:latest"],
      }),
    })
    const user = userEvent.setup()
    await renderLoaded()

    await pickOption(user, "Code execution", "Allowed")
    await pickOption(
      user,
      "Sandbox image",
      "mzdotai/otari-sandbox-container:latest",
    )
    await user.click(screen.getByRole("button", { name: "Save" }))

    const put = calls.find((call) => call.method === "PUT")
    expect(put?.body).toMatchObject({
      image: "mzdotai/otari-sandbox-container:latest",
      tools: null,
    })
  })

  it("offers no tool checkboxes when the sandbox serves a single tool", async () => {
    // Ticking and unticking one box would both mean "narrow nothing", so the
    // card says what is served instead of rendering a control that cannot
    // express anything.
    mockApi()
    await renderLoaded()

    expect(screen.queryByRole("checkbox")).not.toBeInTheDocument()
    expect(screen.getByText(/serves code_execution/i)).toBeInTheDocument()
  })

  it("offers a checkbox per tool once the sandbox serves more than one", async () => {
    mockApi({
      policy: workspaceCodeExecutionPolicy({
        workspace_id: ALPHA,
        available_tools: ["code_execution", "bash_code_execution"],
        configured: true,
        enabled: true,
        tools: ["code_execution"],
      }),
    })
    await renderLoaded()

    expect(
      screen.getByRole("checkbox", { name: "code_execution" }),
    ).toBeChecked()
    expect(
      screen.getByRole("checkbox", { name: "bash_code_execution" }),
    ).not.toBeChecked()
  })

  it("normalizes a full tool selection back to no narrowing", async () => {
    // Every tool ticked narrows nothing, and an empty list is refused by the
    // server, so both ends save `null` rather than a list.
    const calls = mockApi({
      policy: workspaceCodeExecutionPolicy({
        workspace_id: ALPHA,
        available_tools: ["code_execution", "bash_code_execution"],
        configured: true,
        enabled: true,
        tools: ["code_execution"],
      }),
    })
    const user = userEvent.setup()
    await renderLoaded()

    await user.click(
      screen.getByRole("checkbox", { name: "bash_code_execution" }),
    )
    await user.click(screen.getByRole("button", { name: "Save" }))

    const put = calls.find((call) => call.method === "PUT")
    expect(put?.body).toMatchObject({ tools: null })
  })

  it("preserves a stored tool policy this deployment no longer serves", async () => {
    // The escalation this guards: admission refuses a policy naming only kinds
    // the sandbox no longer serves, so silently dropping it on an unrelated
    // save would turn that refusal into permission. Comparing list lengths read
    // one stale entry against one served tool as "the full set" and sent null.
    const calls = mockApi({
      policy: workspaceCodeExecutionPolicy({
        workspace_id: ALPHA,
        configured: true,
        enabled: true,
        available_tools: ["code_execution"],
        tools: ["bash_code_execution"],
      }),
    })
    const user = userEvent.setup()
    await renderLoaded()

    // The operator came here to change something else entirely.
    await user.type(screen.getByLabelText("Max tool-loop iterations"), "4")
    await user.click(screen.getByRole("button", { name: "Save" }))

    const put = calls.find((call) => call.method === "PUT")
    expect(put?.body).toMatchObject({
      max_iterations: 4,
      tools: ["bash_code_execution"],
    })
  })

  it("names a stale tool policy and offers a way out of it", async () => {
    mockApi({
      policy: workspaceCodeExecutionPolicy({
        workspace_id: ALPHA,
        configured: true,
        enabled: true,
        available_tools: ["code_execution"],
        tools: ["bash_code_execution"],
      }),
    })
    await renderLoaded()

    expect(
      screen.getByRole("checkbox", {
        name: "bash_code_execution (no longer served)",
      }),
    ).toBeChecked()
    expect(
      screen.getByRole("checkbox", { name: "code_execution" }),
    ).not.toBeChecked()
    expect(
      screen.getByText(/no longer serves, so its requests are refused/i),
    ).toBeInTheDocument()
  })

  it("clears the restriction once the stale tool is unticked and a served one is not", async () => {
    const calls = mockApi({
      policy: workspaceCodeExecutionPolicy({
        workspace_id: ALPHA,
        configured: true,
        enabled: true,
        available_tools: ["code_execution"],
        tools: ["bash_code_execution"],
      }),
    })
    const user = userEvent.setup()
    await renderLoaded()

    await user.click(screen.getByRole("checkbox", { name: "code_execution" }))
    await user.click(
      screen.getByRole("checkbox", {
        name: "bash_code_execution (no longer served)",
      }),
    )
    await user.click(screen.getByRole("button", { name: "Save" }))

    // Everything served, nothing else: narrows nothing, so no list is stored.
    const put = calls.find((call) => call.method === "PUT")
    expect(put?.body).toMatchObject({ tools: null })
  })

  it("names a withdrawn pin instead of showing it bare", async () => {
    // The scenario the server guards twice: the operator dropped the image from
    // the allow-list after the workspace pinned it. With no option matching it,
    // the control would show the image bare, with nothing saying it is refused,
    // and a save would earn a 400 over a value the screen presented as ordinary.
    mockApi({
      policy: workspaceCodeExecutionPolicy({
        workspace_id: ALPHA,
        configured: true,
        enabled: true,
        allowed_images: ["mzdotai/otari-sandbox-container:latest"],
        image: "ghcr.io/acme/withdrawn:1",
      }),
    })
    await renderLoaded()

    expect(selectTrigger("Sandbox image")).toHaveTextContent(
      "ghcr.io/acme/withdrawn:1 (no longer approved)",
    )
    expect(
      screen.getByText(/no longer approves, so its requests are refused/i),
    ).toBeInTheDocument()
  })

  it("offers the withdrawn pin even when the operator approved nothing else", async () => {
    mockApi({
      policy: workspaceCodeExecutionPolicy({
        workspace_id: ALPHA,
        configured: true,
        enabled: true,
        allowed_images: [],
        image: "ghcr.io/acme/withdrawn:1",
      }),
    })
    await renderLoaded()

    expect(selectTrigger("Sandbox image")).toHaveTextContent(
      "ghcr.io/acme/withdrawn:1 (no longer approved)",
    )
  })

  it("lets the operator move a withdrawn pin back to the deployment default", async () => {
    const calls = mockApi({
      policy: workspaceCodeExecutionPolicy({
        workspace_id: ALPHA,
        configured: true,
        enabled: true,
        allowed_images: ["mzdotai/otari-sandbox-container:latest"],
        image: "ghcr.io/acme/withdrawn:1",
      }),
    })
    const user = userEvent.setup()
    await renderLoaded()

    await pickOption(user, "Sandbox image", "Deployment default")
    await user.click(screen.getByRole("button", { name: "Save" }))

    const put = calls.find((call) => call.method === "PUT")
    expect(put?.body).toMatchObject({ image: null })
  })

  it("shows a stored image", async () => {
    mockApi({
      policy: workspaceCodeExecutionPolicy({
        workspace_id: ALPHA,
        configured: true,
        enabled: true,
        allowed_images: ["ghcr.io/acme/sandbox:2"],
        image: "ghcr.io/acme/sandbox:2",
      }),
    })
    await renderLoaded()

    expect(selectTrigger("Sandbox image")).toHaveTextContent(
      "ghcr.io/acme/sandbox:2",
    )
  })

  it("clears the policy rather than storing one when set back to the deployment default", async () => {
    const calls = mockApi({
      policy: workspaceCodeExecutionPolicy({
        workspace_id: ALPHA,
        configured: true,
        enabled: false,
      }),
    })
    const user = userEvent.setup()
    await renderLoaded()

    await pickOption(user, "Code execution", "Deployment default")
    await user.click(screen.getByRole("button", { name: "Save" }))

    expect(calls.some((call) => call.method === "DELETE")).toBe(true)
    expect(calls.some((call) => call.method === "PUT")).toBe(false)
  })

  it("refuses a limit the deployment could never honor without asking the server", async () => {
    const calls = mockApi()
    const user = userEvent.setup()
    await renderLoaded()

    await pickOption(user, "Code execution", "Allowed")
    await user.type(screen.getByLabelText("Execution timeout (seconds)"), "600")
    await user.click(screen.getByRole("button", { name: "Save" }))

    expect(await screen.findByRole("alert")).toHaveTextContent(
      "Execution timeout must be a whole number of seconds from 1 to 60.",
    )
    expect(calls.some((call) => call.method === "PUT")).toBe(false)
  })

  it("says code execution is unavailable when the deployment has no sandbox", async () => {
    mockApi({
      policy: workspaceCodeExecutionPolicy({
        workspace_id: ALPHA,
        configured: true,
        enabled: true,
        sandbox_configured: false,
      }),
    })
    renderCard()

    expect(
      await screen.findByText(/no sandbox configured/i),
    ).toBeInTheDocument()
  })

  it("does not read the policy at all for a member who cannot manage the workspace", async () => {
    // Reads take the management role server-side, so asking would earn a 403.
    // The card says who can set it instead of rendering a form over an error.
    const policyRequests: string[] = []
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input)
      if (url.includes("/code-execution-policy")) {
        policyRequests.push(url)
        return new Response("forbidden", { status: 403 })
      }
      return Response.json(
        organizationContext({
          role: "member",
          workspace_memberships: [
            { workspace_id: ALPHA, name: "Alpha", role: "member" },
          ],
        }),
      )
    })
    renderCard()

    expect(
      await screen.findByText(/set by an owner or admin/i),
    ).toBeInTheDocument()
    expect(
      screen.queryByRole("button", { name: "Save" }),
    ).not.toBeInTheDocument()
    expect(policyRequests).toEqual([])
  })

  it("keeps its ceilings equal to the ones the server enforces", () => {
    // The two limits are duplicated here because `openapi-typescript` drops
    // `maximum` when it generates `schema.ts`, so the spec is the only place
    // both sides can be compared. Without this, raising the backend cap would
    // leave the form quietly refusing values the server would take.
    const spec = JSON.parse(
      readFileSync(
        join(import.meta.dirname, "../../../../docs/public/openapi.json"),
        "utf8",
      ),
    ) as {
      components: {
        schemas: {
          WorkspaceCodeExecutionPolicyUpdate: {
            properties: Record<string, { anyOf?: { maximum?: number }[] }>
          }
        }
      }
    }
    const properties =
      spec.components.schemas.WorkspaceCodeExecutionPolicyUpdate.properties
    const ceiling = (field: string) =>
      properties[field]?.anyOf?.find((arm) => arm.maximum !== undefined)
        ?.maximum

    expect(ceiling("max_iterations")).toBe(MAX_ITERATIONS)
    expect(ceiling("exec_timeout_s")).toBe(MAX_EXEC_TIMEOUT_S)
  })
})
