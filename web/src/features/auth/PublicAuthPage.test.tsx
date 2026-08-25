import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { render, screen } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import { PublicAuthPage } from "@/features/auth/PublicAuthPage"
import type { PublicAuthPath } from "@/features/auth/publicAuthPaths"
import { apiFetch } from "@/shared/api/client"
import { DeploymentProvider } from "@/shared/hooks/useDeployment"
import { bootstrap } from "@/tests/fixtures"
import { AppProviders } from "@/tests/providers"

// The network boundary, not the hooks: the real hooks, query keys and the
// mutation state the pages branch on all stay live.
vi.mock("@/shared/api/client", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/shared/api/client")>()
  return { ...actual, apiFetch: vi.fn() }
})

function renderPage(
  path: PublicAuthPath,
  { hash = `#${path}`, mailReady = true, oauthProviders = [] as string[] } = {},
) {
  window.location.hash = hash
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  })
  return render(
    <QueryClientProvider client={client}>
      {/* The real provider tree as well as this file's own query client: the
          OAuth callback page signs somebody in through `useAuth`, the way
          `DeploymentRoot` renders it, so it needs the auth provider above it. */}
      <AppProviders>
        <DeploymentProvider
          value={bootstrap({
            mail_ready: mailReady,
            oauth_providers: oauthProviders,
          })}
        >
          <PublicAuthPage path={path} hash={hash} />
        </DeploymentProvider>
      </AppProviders>
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

describe("PublicAuthPage: the OAuth provider gate", () => {
  it.each(["/auth/google/callback", "/auth/github/callback"] as const)(
    "answers %s with a panel naming the provider when this gateway configures none",
    (path) => {
      renderPage(path, { oauthProviders: [] })

      expect(
        screen.getByRole("heading", { name: "Not available on this gateway" }),
      ).toBeInTheDocument()
      // The provider's name, not "this gateway sends no mail": that is the
      // wrong sentence for somebody a provider just redirected here.
      expect(screen.getByText(/sign anyone in with/)).toBeInTheDocument()
      expect(apiFetch).not.toHaveBeenCalled()
    },
  )

  it("gates per provider, not per flow", () => {
    // Google configured, GitHub not: the GitHub callback still answers with the
    // panel, or a bookmark reaches a page whose only outcome is a 503.
    renderPage("/auth/github/callback", { oauthProviders: ["google"] })

    expect(
      screen.getByRole("heading", { name: "Not available on this gateway" }),
    ).toBeInTheDocument()
    expect(
      screen.getByText(
        "This deployment is not configured to sign anyone in with GitHub.",
      ),
    ).toBeInTheDocument()
  })

  it("does not gate an OAuth callback on mail, which it never sends", () => {
    renderPage("/auth/google/callback", {
      mailReady: false,
      oauthProviders: ["google"],
    })

    // The page renders, so the gate let it past. It refuses on its own terms a
    // moment later, because this hash carries no code and no state, which is
    // the callback page's answer and not the deployment's.
    expect(
      screen.queryByRole("heading", { name: "Not available on this gateway" }),
    ).toBeNull()
    expect(
      screen.getByRole("heading", { name: "That sign-in did not complete" }),
    ).toBeInTheDocument()
  })
})
