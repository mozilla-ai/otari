import { render, screen, waitFor } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, beforeEach, expect, it, vi } from "vitest"
import App from "@/app/App"
import { Provider } from "@/app/provider"
import { apiFetch } from "@/shared/api/client"
import { currentRoute } from "@/shared/helpers/appLocation"
import { bootstrap } from "@/tests/fixtures"

vi.mock("@/shared/api/client", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/shared/api/client")>()
  return { ...actual, apiFetch: vi.fn(), siteFetch: vi.fn() }
})

const realLocation = window.location
const assign = vi.fn()
const params = new URLSearchParams({
  client_id: "team/thunderbolt",
  redirect_uri: "https://app.thunderbolt.io/oauth/callback",
  response_type: "code",
  state: "original+/state=",
  code_challenge: "challenge-value",
  code_challenge_method: "S256",
})
const consentHash = `/oauth/authorize?${params}`

function renderApp() {
  return render(
    <Provider>
      <App bootstrap={bootstrap({ oauth_providers: ["google"] })} />
    </Provider>,
  )
}

beforeEach(() => {
  vi.clearAllMocks()
  window.localStorage.clear()
  window.sessionStorage.clear()
  Object.defineProperty(window, "location", {
    configurable: true,
    value: {
      ...realLocation,
      get pathname() {
        return realLocation.pathname
      },
      get search() {
        return realLocation.search
      },
      get hash() {
        return realLocation.hash
      },
      assign,
      replace: (url: string) => realLocation.replace(url),
    },
  })
  window.history.replaceState(null, "", consentHash)
  vi.mocked(apiFetch).mockImplementation(async (path) => {
    if (path.startsWith("/oauth/apps/clients?"))
      return {
        name: "Thunderbolt",
        issuer: "https://auth.otari.example",
        scopes: ["inference:run"],
        redirect_uri: params.get("redirect_uri"),
        verified: true,
        spend_limits: true,
      }
    if (path.startsWith("/oauth/apps/workspaces?"))
      return { data: [{ id: "workspace", name: "Personal" }], count: 1 }
    if (path === "/oauth/apps/authorize")
      return {
        redirect_uri: "https://app.thunderbolt.io/oauth/callback?code=grant",
      }
    throw new Error(`Unexpected request: ${path}`)
  })
  vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
    const body = String(input).endsWith("/authorize")
      ? {
          authorization_url: "https://accounts.google.com/authorize",
          state: "provider-state",
        }
      : { expires_at: "2030-01-01T00:00:00Z" }
    return new Response(JSON.stringify(body), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    })
  })
})

afterEach(() => {
  Object.defineProperty(window, "location", {
    configurable: true,
    value: realLocation,
  })
  window.history.replaceState(null, "", "/")
  window.localStorage.clear()
  window.sessionStorage.clear()
  vi.restoreAllMocks()
})

async function approveOriginalRequest() {
  expect(
    await screen.findByRole("heading", { name: "Connect to Thunderbolt" }),
  ).toBeVisible()
  expect(new URLSearchParams(currentRoute().split("?")[1]).toString()).toBe(
    params.toString(),
  )
  await userEvent.click(
    await screen.findByRole("button", { name: "Allow access" }),
  )
  await waitFor(() =>
    expect(assign).toHaveBeenCalledWith(
      "https://app.thunderbolt.io/oauth/callback?code=grant",
    ),
  )
  const request = vi
    .mocked(apiFetch)
    .mock.calls.find(([path]) => path === "/oauth/apps/authorize")
  expect(JSON.parse(String(request?.[1]?.body))).toEqual({
    ...Object.fromEntries(params),
    workspace_id: "workspace",
    scopes: ["inference:run"],
    expiration: "never",
  })
}

it("preserves the complete app request through local sign-in", async () => {
  renderApp()
  const user = userEvent.setup()
  await user.click(
    screen.getByRole("button", { name: "Sign in with master key" }),
  )
  await user.type(screen.getByLabelText("Master key"), "local-test-key")
  await user.click(screen.getByRole("button", { name: "Sign in" }))
  await approveOriginalRequest()
})

it("preserves the complete app request across provider sign-in and reload", async () => {
  const first = renderApp()
  await userEvent.click(
    screen.getByRole("button", { name: /Sign in with Google/ }),
  )
  await waitFor(() =>
    expect(assign).toHaveBeenCalledWith(
      "https://accounts.google.com/authorize",
    ),
  )
  expect(window.sessionStorage.getItem("otari.oauth.return-route")).toBe(
    consentHash,
  )
  first.unmount()
  window.history.replaceState(
    null,
    "",
    "/sign-in/google/callback?code=provider-code&state=provider-state",
  )
  renderApp()
  await approveOriginalRequest()
})
