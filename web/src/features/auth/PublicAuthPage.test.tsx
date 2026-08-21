import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { render, screen } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import { PublicAuthPage } from "@/features/auth/PublicAuthPage"
import type { PublicAuthPath } from "@/features/auth/publicAuthPaths"
import { apiFetch } from "@/shared/api/client"
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
