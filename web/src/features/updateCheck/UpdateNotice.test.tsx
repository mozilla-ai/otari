import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { render, screen, waitFor } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, describe, expect, it, vi } from "vitest"
import type { UpdateCheckStatus } from "@/client"
import { UpdateNotice } from "@/features/updateCheck/UpdateNotice"
import { API_ROOT } from "@/shared/api/client"
import { DeploymentProvider } from "@/shared/hooks/useDeployment"
import {
  bootstrap,
  organizationContext,
  updateCheckStatus,
} from "@/tests/fixtures"

function jsonResponse(body: unknown): Response {
  return new Response(JSON.stringify(body), {
    status: 200,
    headers: { "Content-Type": "application/json" },
  })
}

function renderNotice(status: UpdateCheckStatus) {
  vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
    const url = String(input)
    if (url.endsWith(`${API_ROOT}/organizations/me`)) {
      return jsonResponse(organizationContext())
    }
    return jsonResponse(url.endsWith(`${API_ROOT}/update-check`) ? status : {})
  })
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  })
  const rendered = render(
    <QueryClientProvider client={client}>
      <DeploymentProvider value={bootstrap()}>
        <UpdateNotice />
      </DeploymentProvider>
    </QueryClientProvider>,
  )
  // Resolves once the status has landed, so an absence asserted after it is not
  // the absence of a page still loading.
  const loaded = () =>
    waitFor(() =>
      expect(client.getQueryState(["update-check"])?.status).toBe("success"),
    )
  return { ...rendered, loaded }
}

const NEWER = updateCheckStatus({
  enabled: true,
  answered_by: "operator",
  latest_version: "0.31.0",
  release_url: "https://github.com/mozilla-ai/otari/releases/tag/v0.31.0",
  update_available: true,
})

const ADVISORY = {
  id: "GHSA-abcd-efgh-ijkl",
  summary: "A key could be read back",
  severity: "high" as const,
  url: "https://github.com/mozilla-ai/otari/security/advisories/GHSA-abcd-efgh-ijkl",
  fixed_in: "0.30.10",
}

describe("UpdateNotice", () => {
  afterEach(() => {
    vi.restoreAllMocks()
    window.localStorage.clear()
  })

  it("says a newer release is out and links to it", async () => {
    renderNotice(NEWER)

    expect(
      await screen.findByText(/Otari 0\.31\.0 is available/),
    ).toBeInTheDocument()
    expect(screen.getByRole("link", { name: "What’s new" })).toHaveAttribute(
      "href",
      NEWER.release_url,
    )
  })

  it("stays dismissed for that release, in this browser", async () => {
    const user = userEvent.setup()
    const { unmount } = renderNotice(NEWER)

    await user.click(await screen.findByRole("button", { name: "Dismiss" }))
    expect(screen.queryByText(/is available/)).not.toBeInTheDocument()

    unmount()
    vi.restoreAllMocks()
    const { loaded } = renderNotice(NEWER)
    await loaded()
    expect(screen.queryByText(/is available/)).not.toBeInTheDocument()
  })

  it("keeps a later release quiet for 30 days after one is dismissed", async () => {
    const day = 24 * 60 * 60 * 1000
    const later = { ...NEWER, latest_version: "0.32.0" }
    window.localStorage.setItem(
      "otari.updateNotice.dismissed",
      JSON.stringify({ version: "0.31.0", at: Date.now() - 29 * day }),
    )
    const first = renderNotice(later)
    await first.loaded()
    expect(screen.queryByText(/is available/)).not.toBeInTheDocument()

    first.unmount()
    vi.restoreAllMocks()
    window.localStorage.setItem(
      "otari.updateNotice.dismissed",
      JSON.stringify({ version: "0.31.0", at: Date.now() - 31 * day }),
    )
    renderNotice(later)
    expect(
      await screen.findByText(/Otari 0\.32\.0 is available/),
    ).toBeInTheDocument()
  })

  it("puts a security advisory ahead of the release, naming the fix", async () => {
    renderNotice({
      ...NEWER,
      advisories: [ADVISORY, { ...ADVISORY, id: "GHSA-2", fixed_in: "0.30.9" }],
    })

    const notice = await screen.findByText(
      /is affected by 2 security advisories/,
    )
    // 0.30.10 is newer than 0.30.9, which a string sort gets backwards.
    expect(notice).toHaveTextContent("Upgrade to 0.30.10 or later.")
    expect(screen.queryByText(/is available/)).not.toBeInTheDocument()
  })

  it("shows nothing while the check is off", async () => {
    const { container, loaded } = renderNotice({ ...NEWER, enabled: false })
    await loaded()
    expect(container).toBeEmptyDOMElement()
  })
})
