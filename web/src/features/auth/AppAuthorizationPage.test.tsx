import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { render, screen, waitFor } from "@testing-library/react"
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
    if (path.startsWith("/workspaces"))
      return { data: workspaces, count: workspaces.length }
    if (path === "/oauth/apps/authorize")
      return {
        redirect_uri:
          "http://localhost:5199/oauth/otari?code=grant&state=state-nonce",
      }
    return {
      name: "Thunderbolt",
      redirect_uri: "http://localhost:5199/oauth/otari",
    }
  })
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  })
  return render(
    <QueryClientProvider client={client}>
      <DeploymentProvider value={bootstrap()}>
        <ThemeProvider>
          <AppAuthorizationPage />
        </ThemeProvider>
      </DeploymentProvider>
    </QueryClientProvider>,
  )
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
  })
})

it("requires a workspace choice when more than one is available", async () => {
  renderPage([
    { id: "one", name: "Personal" },
    { id: "two", name: "Research" },
  ])
  const button = await screen.findByRole("button", { name: "Allow access" })
  expect(button).toBeDisabled()
  await userEvent.click(screen.getByRole("button", { name: /Workspace/ }))
  await userEvent.click(await screen.findByRole("option", { name: "Research" }))
  expect(button).toBeEnabled()
  await userEvent.click(button)
  await waitFor(() => expect(assign).toHaveBeenCalled())
  const call = vi
    .mocked(apiFetch)
    .mock.calls.find(([path]) => path === "/oauth/apps/authorize")
  expect(JSON.parse(String(call?.[1]?.body)).workspace_id).toBe("two")
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
    "http://localhost:5199/oauth/otari?error=access_denied&state=state-nonce",
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
    if (path.startsWith("/workspaces"))
      throw new Error("Workspaces are unavailable")
    return {
      name: "Thunderbolt",
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
