import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { render, screen } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import { PublicAuthPage } from "@/features/auth/PublicAuthPage"
import type { PublicAuthPath } from "@/features/auth/publicAuthPaths"
import { ApiError, apiFetch } from "@/shared/api/client"
import { DeploymentProvider } from "@/shared/hooks/useDeployment"
import { bootstrap } from "@/tests/fixtures"

// The network boundary, not the hooks: the real hooks, query keys and the
// mutation state the pages branch on all stay live.
vi.mock("@/shared/api/client", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/shared/api/client")>()
  return { ...actual, apiFetch: vi.fn() }
})

function renderPage(
  path: PublicAuthPath,
  { hash = `#${path}`, mailReady = true } = {},
) {
  window.location.hash = hash
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  })
  return render(
    <QueryClientProvider client={client}>
      <DeploymentProvider value={bootstrap({ mail_ready: mailReady })}>
        <PublicAuthPage path={path} hash={hash} />
      </DeploymentProvider>
    </QueryClientProvider>,
  )
}

beforeEach(() => {
  vi.clearAllMocks()
})

afterEach(() => {
  vi.restoreAllMocks()
  window.location.hash = ""
})

describe("PublicAuthPage: the mail gate", () => {
  it.each([
    "/signup",
    "/check-email",
    "/resend-verification",
    "/recover-password",
  ] as const)(
    "answers %s with a panel instead of a form when this gateway cannot send mail",
    (path) => {
      renderPage(path, { mailReady: false })

      expect(
        screen.getByRole("heading", { name: "Not available on this gateway" }),
      ).toBeInTheDocument()
      expect(screen.queryByRole("button")).toBeNull()
      expect(apiFetch).not.toHaveBeenCalled()
    },
  )

  it("still opens a reset link, whose message was sent while mail worked", () => {
    renderPage("/reset-password", {
      hash: "#/reset-password?token=abc",
      mailReady: false,
    })

    expect(
      screen.getByRole("heading", { name: "Set a new password" }),
    ).toBeInTheDocument()
  })
})

describe("SignupPage", () => {
  it("claims the identity and lands on the check-email page", async () => {
    vi.mocked(apiFetch).mockResolvedValue({ message: "…" } as never)
    const user = userEvent.setup()
    renderPage("/signup")

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
    renderPage("/signup")

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
    renderPage("/signup")

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
    renderPage("/signup")

    await user.type(screen.getByLabelText("Email"), "ada@example.com")
    await user.type(screen.getByLabelText("Password"), "correct-horse")
    await user.type(screen.getByLabelText("Confirm password"), "correct-horse")
    await user.click(screen.getByRole("button", { name: "Claim account" }))

    expect(
      await screen.findByText(
        "Outgoing mail is not configured on this deployment.",
      ),
    ).toBeInTheDocument()
    expect(window.location.hash).toBe("#/signup")
  })
})

describe("ResendVerificationPage", () => {
  it("sends a fresh link and lands on the check-email page", async () => {
    vi.mocked(apiFetch).mockResolvedValue({ message: "…" } as never)
    const user = userEvent.setup()
    renderPage("/resend-verification")

    await user.type(screen.getByLabelText("Email"), "ada@example.com")
    await user.click(screen.getByRole("button", { name: "Send link" }))

    await vi.waitFor(() => {
      expect(window.location.hash).toBe("#/check-email?type=resend")
    })
    expect(vi.mocked(apiFetch).mock.calls[0]?.[0]).toBe(
      "/v1/auth/resend-verification",
    )
  })
})

describe("CheckEmailPage", () => {
  it("says what was sent without ever saying whether the address exists", () => {
    renderPage("/check-email", { hash: "#/check-email?type=signup" })

    expect(
      screen.getByText(/If that address is on this gateway's roster/),
    ).toBeInTheDocument()
  })

  it("uses the resend wording when that is what sent it", () => {
    renderPage("/check-email", { hash: "#/check-email?type=resend" })

    expect(
      screen.getByText(/If that address is registered and still unverified/),
    ).toBeInTheDocument()
  })
})

describe("RecoverPasswordPage", () => {
  it("confirms in the conditional rather than reporting on the address", async () => {
    vi.mocked(apiFetch).mockResolvedValue({ message: "…" } as never)
    const user = userEvent.setup()
    renderPage("/recover-password")

    await user.type(screen.getByLabelText("Email"), "ada@example.com")
    await user.click(screen.getByRole("button", { name: "Send reset link" }))

    expect(
      await screen.findByText(/If that address has a password on this gateway/),
    ).toBeInTheDocument()
    expect(vi.mocked(apiFetch).mock.calls[0]?.[0]).toBe(
      "/v1/auth/password/reset",
    )
  })
})
