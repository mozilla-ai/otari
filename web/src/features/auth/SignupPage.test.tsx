import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { render, screen } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import { SignupPage } from "@/features/auth/SignupPage"
import { ApiError, apiFetch } from "@/shared/api/client"

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
    <QueryClientProvider client={client}>
      <SignupPage />
    </QueryClientProvider>,
  )
}

beforeEach(() => {
  vi.clearAllMocks()
  window.location.hash = ""
})

afterEach(() => {
  vi.restoreAllMocks()
  window.location.hash = ""
})

describe("SignupPage", () => {
  it("claims the identity and lands on the check-email page", async () => {
    vi.mocked(apiFetch).mockResolvedValue({ message: "…" } as never)
    const user = userEvent.setup()
    renderPage()

    await user.type(screen.getByLabelText("Email"), "ada@example.com")
    await user.type(screen.getByLabelText("Password"), "correct-horse")
    await user.type(screen.getByLabelText("Confirm password"), "correct-horse")
    await user.click(screen.getByRole("button", { name: "Claim account" }))

    await vi.waitFor(() => {
      expect(window.location.hash).toBe("#/check-email?type=signup")
    })
    const [path, init] = vi.mocked(apiFetch).mock.calls[0] ?? []
    expect(path).toBe("/v1/auth/signup")
    expect(JSON.parse(String(init?.body))).toEqual({
      email: "ada@example.com",
      password: "correct-horse",
      full_name: null,
    })
  })

  it("keeps the button disabled until the two passwords agree", async () => {
    const user = userEvent.setup()
    renderPage()

    const submit = screen.getByRole("button", { name: "Claim account" })
    await user.type(screen.getByLabelText("Email"), "ada@example.com")
    await user.type(screen.getByLabelText("Password"), "correct-horse")
    await user.type(screen.getByLabelText("Confirm password"), "typo")

    expect(
      await screen.findByText("The two passwords do not match."),
    ).toBeInTheDocument()
    expect(submit).toBeDisabled()
    expect(apiFetch).not.toHaveBeenCalled()
  })

  it("refuses a password the gateway would refuse, without asking it", async () => {
    const user = userEvent.setup()
    renderPage()

    await user.type(screen.getByLabelText("Email"), "ada@example.com")
    await user.type(screen.getByLabelText("Password"), "short")
    await user.type(screen.getByLabelText("Confirm password"), "short")

    expect(
      await screen.findByText("At least 8 characters."),
    ).toBeInTheDocument()
    expect(apiFetch).not.toHaveBeenCalled()
  })

  it("shows the gateway's own refusal and stays on the form", async () => {
    vi.mocked(apiFetch).mockRejectedValue(
      new ApiError(503, "Outgoing mail is not configured on this deployment."),
    )
    const user = userEvent.setup()
    renderPage()

    await user.type(screen.getByLabelText("Email"), "ada@example.com")
    await user.type(screen.getByLabelText("Password"), "correct-horse")
    await user.type(screen.getByLabelText("Confirm password"), "correct-horse")
    await user.click(screen.getByRole("button", { name: "Claim account" }))

    expect(
      await screen.findByText(
        "Outgoing mail is not configured on this deployment.",
      ),
    ).toBeInTheDocument()
    // No navigation: the page only leaves for #/check-email on a success.
    expect(window.location.hash).toBe("")
  })
})
