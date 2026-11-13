import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { act, render, screen, waitFor } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, beforeEach, expect, it, vi } from "vitest"
import { apiFetch } from "@/shared/api/client"
import { DeploymentProvider } from "@/shared/hooks/useDeployment"
import { ThemeProvider } from "@/shared/hooks/useTheme"
import { bootstrap } from "@/tests/fixtures"
import { AppAuthorizationPage } from "./AppAuthorizationPage"

vi.mock("@/shared/api/client", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/shared/api/client")>()
  return { ...actual, apiFetch: vi.fn() }
})

const realLocation = window.location
const assign = vi.fn()
const request = new URLSearchParams({
  client_id: "thunderbolt",
  redirect_uri: "http://localhost:5199/oauth/otari",
  response_type: "code",
  state: "state-nonce",
  code_challenge: "pkce-challenge",
  code_challenge_method: "S256",
})

function renderPage(
  workspaces = [{ id: "workspace-one", name: "Personal" }],
  appError?: Error,
) {
  vi.mocked(apiFetch).mockImplementation(async (path) => {
    if (appError) throw appError
    if (path.startsWith("/oauth/apps/workspaces"))
      return { data: [...workspaces], count: workspaces.length }
    if (path === "/oauth/apps/authorize")
      return {
        redirect_uri:
          "http://localhost:5199/oauth/otari?code=grant&state=state-nonce",
      }
    return {
      name: "Thunderbolt",
      issuer: "https://auth.otari.example",
      scopes: ["inference:run"],
      redirect_uri: "http://localhost:5199/oauth/otari",
    }
  })
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  })
  const view = render(
    <QueryClientProvider client={client}>
      <DeploymentProvider value={bootstrap()}>
        <ThemeProvider>
          <AppAuthorizationPage />
        </ThemeProvider>
      </DeploymentProvider>
    </QueryClientProvider>,
  )
  return { ...view, client }
}

beforeEach(() => {
  vi.clearAllMocks()
  Object.defineProperty(window, "location", {
    configurable: true,
    value: { ...realLocation, hash: `#/oauth/authorize?${request}`, assign },
  })
})
afterEach(() => {
  Object.defineProperty(window, "location", {
    configurable: true,
    value: realLocation,
  })
  vi.restoreAllMocks()
})

it("shows the app, permission boundaries and the sole workspace before explicit approval", async () => {
  renderPage()
  expect(
    await screen.findByRole("heading", { name: "Connect to Thunderbolt" }),
  ).toBeVisible()
  expect(await screen.findByText("Personal")).toBeVisible()
  // Match the shared sign-in shell placement; jsdom cannot measure its vertical offset.
  expect(screen.getByRole("main")).toHaveClass("items-center", "md:pt-15")
  expect(screen.getByRole("main")).not.toHaveClass("justify-center")
  expect(
    screen.getByText(/Requests count toward your existing budgets/),
  ).toBeVisible()
  expect(screen.getByText(/revoke access anytime/)).toBeVisible()
  expect(assign).not.toHaveBeenCalled()
  await userEvent.click(screen.getByRole("button", { name: "Allow access" }))
  await waitFor(() =>
    expect(assign).toHaveBeenCalledWith(expect.stringContaining("?code=grant")),
  )
  const call = vi
    .mocked(apiFetch)
    .mock.calls.find(([path]) => path === "/oauth/apps/authorize")
  expect(JSON.parse(String(call?.[1]?.body))).toEqual({
    ...Object.fromEntries(request),
    workspace_id: "workspace-one",
    scopes: ["inference:run"],
  })
})

it("requires a workspace choice when more than one is available", async () => {
  renderPage([
    { id: "one", name: "Personal" },
    { id: "two", name: "Research" },
  ])
  const button = await screen.findByRole("button", { name: "Allow access" })
  expect(button).toBeDisabled()
  await userEvent.type(
    screen.getByRole("combobox", { name: "Workspace" }),
    "Research",
  )
  await userEvent.click(await screen.findByRole("option", { name: "Research" }))
  expect(button).toBeEnabled()
  await userEvent.click(button)
  await waitFor(() => expect(assign).toHaveBeenCalled())
  const call = vi
    .mocked(apiFetch)
    .mock.calls.find(([path]) => path === "/oauth/apps/authorize")
  expect(JSON.parse(String(call?.[1]?.body)).workspace_id).toBe("two")
})

it.each([
  { name: "none remain", remaining: [] },
  {
    name: "other choices remain",
    remaining: [
      { id: "three", name: "Engineering" },
      { id: "four", name: "Design" },
    ],
  },
])("clears an unavailable workspace when $name", async ({ remaining }) => {
  const workspaces = [
    { id: "one", name: "Personal" },
    { id: "two", name: "Research" },
  ]
  const { client } = renderPage(workspaces)
  const user = userEvent.setup()
  await user.type(
    await screen.findByRole("combobox", { name: "Workspace" }),
    "Research",
  )
  await user.click(await screen.findByRole("option", { name: "Research" }))
  await waitFor(() =>
    expect(screen.getByRole("button", { name: "Allow access" })).toBeEnabled(),
  )
  await user.tab()

  workspaces.splice(0, workspaces.length, ...remaining)
  await act(async () => {
    await client.invalidateQueries()
  })

  await waitFor(() =>
    expect(screen.getByRole("button", { name: "Allow access" })).toBeDisabled(),
  )
  expect(screen.queryByDisplayValue("Research")).not.toBeInTheDocument()
  expect(assign).not.toHaveBeenCalled()
})

it("replaces a removed selection with the current sole workspace", async () => {
  const workspaces = [
    { id: "one", name: "Personal" },
    { id: "two", name: "Research" },
  ]
  const { client } = renderPage(workspaces)
  const user = userEvent.setup()
  await user.type(
    await screen.findByRole("combobox", { name: "Workspace" }),
    "Research",
  )
  await user.click(await screen.findByRole("option", { name: "Research" }))
  await waitFor(() =>
    expect(screen.getByRole("button", { name: "Allow access" })).toBeEnabled(),
  )

  workspaces.splice(0, workspaces.length, { id: "one", name: "Personal" })
  await act(async () => {
    await client.invalidateQueries()
  })

  expect(await screen.findByText("Personal")).toBeVisible()
  expect(screen.queryByText("Research")).not.toBeInTheDocument()
  await user.click(screen.getByRole("button", { name: "Allow access" }))
  await waitFor(() => expect(assign).toHaveBeenCalled())
  const approval = vi
    .mocked(apiFetch)
    .mock.calls.find(([path]) => path === "/oauth/apps/authorize")
  expect(JSON.parse(String(approval?.[1]?.body)).workspace_id).toBe("one")
})

it("explains missing workspace access and prevents approval", async () => {
  renderPage([])
  expect(
    await screen.findByText(/You don’t have a workspace to connect/),
  ).toBeVisible()
  expect(screen.getByRole("button", { name: "Allow access" })).toBeDisabled()
})

it("cancels back to the verified callback without issuing a grant", async () => {
  renderPage()
  await userEvent.click(await screen.findByRole("button", { name: "Cancel" }))
  expect(assign).toHaveBeenCalledWith(
    "http://localhost:5199/oauth/otari?error=access_denied&state=state-nonce&iss=https%3A%2F%2Fauth.otari.example",
  )
  expect(
    vi
      .mocked(apiFetch)
      .mock.calls.some(([path]) => path === "/oauth/apps/authorize"),
  ).toBe(false)
})

it("never offers approval or a redirect when the app cannot be verified", async () => {
  renderPage(undefined, new Error("Unknown app"))
  expect(await screen.findByRole("alert")).toHaveTextContent("Unknown app")
  expect(
    screen.queryByRole("button", { name: "Allow access" }),
  ).not.toBeInTheDocument()
  expect(
    screen.queryByRole("button", { name: "Cancel" }),
  ).not.toBeInTheDocument()
  expect(assign).not.toHaveBeenCalled()
})

it("keeps consent open and displays an authorization failure", async () => {
  renderPage()
  const button = await screen.findByRole("button", { name: "Allow access" })
  vi.mocked(apiFetch).mockRejectedValue(new Error("Workspace access denied"))
  await userEvent.click(button)
  expect(await screen.findByRole("alert")).toHaveTextContent(
    "Workspace access denied",
  )
  expect(button).toBeEnabled()
  expect(assign).not.toHaveBeenCalled()
})

it("keeps cancellation available if loading workspaces fails", async () => {
  renderPage()
  vi.mocked(apiFetch).mockImplementation(async (path) => {
    if (path.startsWith("/oauth/apps/workspaces"))
      throw new Error("Workspaces are unavailable")
    return {
      name: "Thunderbolt",
      issuer: "https://auth.otari.example",
      scopes: ["inference:run"],
      redirect_uri: "http://localhost:5199/oauth/otari",
    }
  })
  expect(await screen.findByRole("alert")).toHaveTextContent(
    "Workspaces are unavailable",
  )
  expect(screen.getByRole("button", { name: "Allow access" })).toBeDisabled()
  expect(screen.getByRole("button", { name: "Cancel" })).toBeEnabled()
  expect(assign).not.toHaveBeenCalled()
})

it("prevents repeated approval and cancellation while authorization is pending", async () => {
  renderPage()
  const button = await screen.findByRole("button", { name: "Allow access" })
  const cancel = screen.getByRole("button", { name: "Cancel" })
  let finishAuthorization: (value: unknown) => void = () => {}
  const pending = new Promise<unknown>((resolve) => {
    finishAuthorization = resolve
  })
  vi.mocked(apiFetch).mockReturnValue(pending)
  await userEvent.click(button)
  expect(button).toBeDisabled()
  expect(cancel).toBeDisabled()
  await userEvent.click(button)
  expect(
    vi
      .mocked(apiFetch)
      .mock.calls.filter(([path]) => path === "/oauth/apps/authorize"),
  ).toHaveLength(1)
  finishAuthorization({
    redirect_uri:
      "http://localhost:5199/oauth/otari?code=one&state=state-nonce",
  })
  await waitFor(() => expect(assign).toHaveBeenCalledTimes(1))
})

it("does not redirect or offer approval after cached app validation fails", async () => {
  const { client } = renderPage()
  expect(await screen.findByRole("button", { name: "Cancel" })).toBeEnabled()
  vi.mocked(apiFetch).mockRejectedValue(new Error("App registration removed"))
  await act(async () => {
    await client.invalidateQueries()
  })
  expect(await screen.findByRole("alert")).toHaveTextContent(
    "App registration removed",
  )
  expect(
    screen.queryByRole("button", { name: "Cancel" }),
  ).not.toBeInTheDocument()
  expect(
    screen.queryByRole("button", { name: "Allow access" }),
  ).not.toBeInTheDocument()
  expect(assign).not.toHaveBeenCalled()
})

it("passes slash-containing client ids as query parameters and preserves omitted response_type", async () => {
  const params = new URLSearchParams(request)
  params.set("client_id", "team/thunderbolt")
  params.delete("response_type")
  window.location.hash = `#/oauth/authorize?${params}`
  renderPage()
  await userEvent.click(
    await screen.findByRole("button", { name: "Allow access" }),
  )
  await waitFor(() => expect(assign).toHaveBeenCalled())
  const metadata = vi
    .mocked(apiFetch)
    .mock.calls.find(([path]) => path.startsWith("/oauth/apps/clients?"))
  expect(
    new URL(String(metadata?.[0]), "https://otari.example").searchParams.get(
      "client_id",
    ),
  ).toBe("team/thunderbolt")
  const approval = vi
    .mocked(apiFetch)
    .mock.calls.find(([path]) => path === "/oauth/apps/authorize")
  const body = JSON.parse(String(approval?.[1]?.body))
  expect(body.client_id).toBe("team/thunderbolt")
  expect(body).not.toHaveProperty("response_type")
})

it("searches beyond the first workspace page and retains the chosen workspace", async () => {
  renderPage([{ id: "first", name: "First workspace" }])
  vi.mocked(apiFetch).mockImplementation(async (path) => {
    if (path.startsWith("/oauth/apps/workspaces")) {
      const search = new URL(path, "https://otari.example").searchParams.get(
        "search",
      )
      return search
        ? { data: [{ id: "last", name: "Research" }], count: 1 }
        : { data: [{ id: "first", name: "First workspace" }], count: 100001 }
    }
    if (path === "/oauth/apps/authorize")
      return { redirect_uri: "http://localhost:5199/oauth/otari?code=grant" }
    return {
      name: "Thunderbolt",
      issuer: "https://auth.otari.example",
      scopes: ["inference:run"],
      redirect_uri: "http://localhost:5199/oauth/otari",
    }
  })
  const user = userEvent.setup()
  const input = await screen.findByRole("combobox", { name: "Workspace" })
  await user.type(input, "Research")
  await user.click(await screen.findByRole("option", { name: "Research" }))
  await waitFor(() => expect(input).toHaveValue("Research"))
  const button = screen.getByRole("button", { name: "Allow access" })
  await waitFor(() => expect(button).toBeEnabled())
  await user.click(button)
  await waitFor(() => expect(assign).toHaveBeenCalled())
  const approval = vi
    .mocked(apiFetch)
    .mock.calls.find(([path]) => path === "/oauth/apps/authorize")
  expect(JSON.parse(String(approval?.[1]?.body)).workspace_id).toBe("last")
  const searches = vi
    .mocked(apiFetch)
    .mock.calls.filter(([path]) => path.startsWith("/oauth/apps/workspaces"))
  expect(
    searches.some(
      ([path]) =>
        new URL(path, "https://otari.example").searchParams.get("search") ===
        "Research",
    ),
  ).toBe(true)
})
