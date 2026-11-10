import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { render, screen, waitFor } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { SetupPage } from "@/features/auth/SetupPage"
import { captureSetupKey, forgetSetupKey } from "@/features/auth/setupKey"
import {
  forgetSignInNotice,
  readSignInNotice,
} from "@/features/auth/signInNotice"
import { ApiError, apiFetch, createSession } from "@/shared/api/client"
import { DeploymentProvider, useDeployment } from "@/shared/hooks/useDeployment"
import { ThemeProvider } from "@/shared/hooks/useTheme"
import { bootstrap } from "@/tests/fixtures"

vi.mock("@/shared/api/client", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/shared/api/client")>()
  return { ...actual, apiFetch: vi.fn(), createSession: vi.fn() }
})

const auth = { isAuthenticated: false, login: vi.fn() }
vi.mock("@/features/auth/AuthContext", () => ({ useAuth: () => auth }))

const LINK = "#/setup?key=otari-mk-first-run"

function SignInMethods() {
  return <output>{useDeployment().sign_in_methods.join(",")}</output>
}

function renderPage(
  hash: string,
  signInMethods: ("master_key" | "password")[] = ["master_key"],
) {
  window.location.hash = hash
  const client = new QueryClient({
    defaultOptions: { mutations: { retry: false } },
  })
  return render(
    <DeploymentProvider value={bootstrap({ sign_in_methods: signInMethods })}>
      <ThemeProvider>
        <QueryClientProvider client={client}>
          <SetupPage hash={hash} />
          <SignInMethods />
        </QueryClientProvider>
      </ThemeProvider>
    </DeploymentProvider>,
  )
}

async function fillAndSubmit() {
  const user = userEvent.setup()
  await user.type(screen.getByLabelText("Email"), "operator@example.com")
  await user.type(screen.getByLabelText("Password"), "a-real-password")
  await user.click(screen.getByRole("button", { name: "Finish setup" }))
}

beforeEach(() => {
  vi.clearAllMocks()
  auth.isAuthenticated = false
  forgetSetupKey()
  forgetSignInNotice()
  vi.mocked(createSession).mockResolvedValue({ isOk: true })
  vi.mocked(apiFetch).mockResolvedValue({
    email: "operator@example.com",
    master_key_sign_in_retired: true,
  })
})

afterEach(() => {
  window.location.hash = ""
})

describe("captureSetupKey", () => {
  it("takes the key out of the address bar and keeps it for the page", () => {
    window.location.hash = LINK

    expect(captureSetupKey(window.location.hash)).toBe("otari-mk-first-run")
    expect(window.location.hash).toBe("#/setup")
    expect(window.location.href).not.toContain("otari-mk-first-run")
    expect(captureSetupKey(window.location.hash)).toBe("otari-mk-first-run")
  })

  it("ignores a key on any other page", () => {
    window.location.hash = "#/keys?key=otari-mk-first-run"

    expect(captureSetupKey(window.location.hash)).toBeNull()
    expect(window.location.hash).toBe("#/keys?key=otari-mk-first-run")
  })
})

describe("SetupPage", () => {
  it("strips the key from the address bar on arrival and asks only for an address and a password", () => {
    renderPage(LINK)

    expect(window.location.href).not.toContain("otari-mk-first-run")
    expect(window.location.hash).toBe("#/setup")
    expect(
      screen.getByRole("heading", { name: "Set up Otari" }),
    ).toBeInTheDocument()
    expect(screen.queryByLabelText(/master key/i)).toBeNull()
    expect(createSession).not.toHaveBeenCalled()
  })

  it("signs in with the key from the link, then claims the deployment with the password", async () => {
    renderPage(LINK)

    await fillAndSubmit()

    await waitFor(() => expect(auth.login).toHaveBeenCalledOnce())
    expect(createSession).toHaveBeenCalledWith({
      masterKey: "otari-mk-first-run",
    })
    const [path, init] = vi.mocked(apiFetch).mock.calls[0] ?? []
    expect(path).toBe("/auth/password")
    expect(init?.method).toBe("PUT")
    expect(JSON.parse(String(init?.body))).toEqual({
      email: "operator@example.com",
      new_password: "a-real-password",
    })
    // The claim retired the master key as a sign-in, in this tab too.
    expect(screen.getByRole("status")).toHaveTextContent("password")
    expect(window.location.hash).toBe("#/")
    expect(readSignInNotice()).toBeNull()
  })

  it("falls back to sign-in with an explanation on a gateway that is already claimed", async () => {
    renderPage(LINK, ["password"])

    await waitFor(() => expect(window.location.hash).toBe("#/"))
    expect(readSignInNotice()).toMatch(/already set up/)
    expect(screen.queryByRole("heading", { name: "Set up Otari" })).toBeNull()
    expect(createSession).not.toHaveBeenCalled()
    expect(window.location.href).not.toContain("otari-mk-first-run")
  })

  it("sends a signed-in caller on a claimed gateway to the dashboard with nothing to explain", async () => {
    auth.isAuthenticated = true
    renderPage(LINK, ["password"])

    await waitFor(() => expect(window.location.hash).toBe("#/"))
    expect(readSignInNotice()).toBeNull()
  })

  it("falls back to sign-in when the link carries no key", async () => {
    renderPage("#/setup")

    await waitFor(() => expect(window.location.hash).toBe("#/"))
    expect(readSignInNotice()).toMatch(/incomplete/)
  })

  it("falls back to sign-in when the gateway refuses the key", async () => {
    vi.mocked(createSession).mockResolvedValue({
      isOk: false,
      status: 401,
      message: "Invalid master key.",
    })
    renderPage(LINK)

    await fillAndSubmit()

    await waitFor(() => expect(window.location.hash).toBe("#/"))
    expect(readSignInNotice()).toMatch(/did not accept/)
    expect(apiFetch).not.toHaveBeenCalled()
    expect(auth.login).not.toHaveBeenCalled()
  })

  it("keeps the form up when the password is refused, and does not sign in twice", async () => {
    vi.mocked(apiFetch).mockRejectedValueOnce(
      new ApiError(409, "That address belongs to another account."),
    )
    renderPage(LINK)

    await fillAndSubmit()

    expect(
      await screen.findByText("That address belongs to another account."),
    ).toBeInTheDocument()
    expect(auth.login).not.toHaveBeenCalled()

    await userEvent.click(screen.getByRole("button", { name: "Finish setup" }))

    await waitFor(() => expect(auth.login).toHaveBeenCalledOnce())
    expect(createSession).toHaveBeenCalledOnce()
  })
})
