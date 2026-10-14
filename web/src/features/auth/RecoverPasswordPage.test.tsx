import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { render, screen } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { RecoverPasswordPage } from "@/features/auth/RecoverPasswordPage"
import {
  readSignInAddress,
  rememberSignInAddress,
} from "@/features/auth/signInAddress"
import { ApiError, apiFetch } from "@/shared/api/client"
import { DeploymentProvider } from "@/shared/hooks/useDeployment"
import { ThemeProvider } from "@/shared/hooks/useTheme"
import { bootstrap } from "@/tests/fixtures"

// The network boundary, not the hooks: the real hooks, their query keys, and
// the mutation state the page branches on all stay live.
vi.mock("@/shared/api/client", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/shared/api/client")>()
  return { ...actual, apiFetch: vi.fn() }
})

function renderPage() {
  const client = new QueryClient({
    defaultOptions: { mutations: { retry: false } },
  })
  return render(
    <DeploymentProvider value={bootstrap()}>
      <ThemeProvider>
        <QueryClientProvider client={client}>
          <RecoverPasswordPage />
        </QueryClientProvider>
      </ThemeProvider>
    </DeploymentProvider>,
  )
}

beforeEach(() => {
  vi.clearAllMocks()
  window.sessionStorage.clear()
  window.location.hash = ""
})

afterEach(() => {
  vi.restoreAllMocks()
  window.location.hash = ""
})

vi.mock("@/features/auth/overlayPublicAuthFields", () => ({
  PublicAuthFields: ({ page, isBusy }: { page: string; isBusy: boolean }) => (
    <p>{`fields for ${page}, ${isBusy ? "busy" : "idle"}`}</p>
  ),
}))

describe("RecoverPasswordPage", () => {
  it("renders the edition's own fields ahead of the address", () => {
    renderPage()

    expect(
      screen.getByText("fields for recover-password, idle"),
    ).toBeInTheDocument()
  })

  it("confirms in the conditional rather than reporting on the address", async () => {
    vi.mocked(apiFetch).mockResolvedValue({ message: "…" } as never)
    const user = userEvent.setup()
    renderPage()

    await user.type(screen.getByLabelText("Email"), "ada@example.com")
    await user.click(screen.getByRole("button", { name: "Send link" }))

    expect(
      await screen.findByText(
        /If that address has an account on this gateway, a link to set or reset its password/,
      ),
    ).toBeInTheDocument()
    expect(vi.mocked(apiFetch).mock.calls[0]?.[0]).toBe("/auth/password/reset")
  })

  it("starts from the address the sign-in screen handed over, and hands it back", async () => {
    rememberSignInAddress("ada@example.com")
    vi.mocked(apiFetch).mockResolvedValue({ message: "…" } as never)
    const user = userEvent.setup()
    renderPage()

    expect(screen.getByLabelText("Email")).toHaveValue("ada@example.com")
    await user.click(screen.getByRole("button", { name: "Send link" }))

    await screen.findByRole("heading", { name: "Check your email" })
    expect(
      JSON.parse(String(vi.mocked(apiFetch).mock.calls[0]?.[1]?.body)),
    ).toEqual({
      email: "ada@example.com",
    })
    expect(readSignInAddress()).toBe("ada@example.com")
  })

  it("clears a stale refusal as soon as the address is retyped", async () => {
    vi.mocked(apiFetch).mockRejectedValue(
      new ApiError(429, "Too many attempts. Try again in a minute."),
    )
    const user = userEvent.setup()
    renderPage()

    await user.type(screen.getByLabelText("Email"), "ada@example.com")
    await user.click(screen.getByRole("button", { name: "Send link" }))
    expect(
      await screen.findByText("Too many attempts. Try again in a minute."),
    ).toBeInTheDocument()

    await user.type(screen.getByLabelText("Email"), "x")

    expect(
      screen.queryByText("Too many attempts. Try again in a minute."),
    ).toBeNull()
  })
})
