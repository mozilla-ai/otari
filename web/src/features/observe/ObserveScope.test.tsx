import { useLocation } from "@tanstack/react-router"
import { screen, waitFor, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import { ActivityPage } from "@/features/activity/ActivityPage"
import { OverviewIndex } from "@/features/overview/OverviewPage"
import { SessionsPage } from "@/features/traces/SessionsPage"
import { UsagePage } from "@/features/usage/UsagePage"
import { entry, mockApi, renderPage } from "@/tests/activity"
import { organizationMember } from "@/tests/fixtures"
import { ObserveFilterBar } from "./ObserveFilterBar"
import { ObserveScopeCaption, ObserveScopeSwitch } from "./ObserveScope"

const surfaces = vi.hoisted(() => ({ published: new Set<string>() }))

vi.mock("@/shared/hooks/useDeployment", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/shared/hooks/useDeployment")>()),
  useSurfaces: () => (surface: string) => surfaces.published.has(surface),
}))

beforeEach(() => {
  surfaces.published = new Set(["usage", "traces"])
})

afterEach(() => {
  vi.restoreAllMocks()
})

const ME = organizationMember({
  organization_member_id: "om-1",
  attribution_user_id: "me-id",
})
const MEMBER = {
  deploymentOperator: false,
  role: "member",
  members: [ME],
  workspace: "ws-1",
}

function UrlProbe() {
  const { searchStr } = useLocation()
  return <output aria-label="Search in the URL">{searchStr}</output>
}

const urlParams = () =>
  new URLSearchParams(
    screen.getByLabelText("Search in the URL").textContent ?? "",
  )

const paramsOf = (calls: { url: string }[], path: string) =>
  calls
    .filter((call) => call.url.includes(path))
    .map((call) => new URL(call.url, "http://localhost").searchParams)

describe("the Mine | Everyone switch", () => {
  it("writes the scope to the URL and clears the user filter for Mine", async () => {
    mockApi({ members: [ME], workspace: "ws-1" })
    renderPage(
      <>
        <ObserveScopeSwitch />
        <ObserveFilterBar view="requests" />
        <UrlProbe />
      </>,
      "/requests?user_id=alice",
    )
    const user = userEvent.setup()

    await user.click(await screen.findByRole("button", { name: "Mine" }))

    await waitFor(() => expect(urlParams().get("scope")).toBe("mine"))
    expect(urlParams().has("user_id")).toBe(false)
    expect(screen.queryByRole("combobox", { name: "User" })).toBeNull()

    await user.click(screen.getByRole("button", { name: "Everyone" }))
    await waitFor(() => expect(urlParams().get("scope")).toBe("everyone"))
    expect(screen.getByRole("combobox", { name: "User" })).toBeVisible()
  })

  it("hides the User picker and its chips for a member, who opens on Mine", async () => {
    mockApi(MEMBER)
    renderPage(
      <>
        <ObserveScopeSwitch />
        <ObserveFilterBar view="requests" />
      </>,
      "/requests",
    )

    const mine = await screen.findByRole("button", { name: "Mine" })
    await waitFor(() => expect(mine).toHaveAttribute("aria-pressed", "true"))
    expect(screen.queryByRole("combobox", { name: "User" })).toBeNull()
    expect(screen.getByRole("combobox", { name: "Model" })).toBeVisible()
  })

  it("narrows the picker options to the caller's own id under Mine", async () => {
    const { calls } = mockApi(MEMBER)
    renderPage(
      <>
        <ObserveScopeSwitch />
        <ObserveFilterBar view="requests" />
      </>,
      "/requests",
    )

    await waitFor(() =>
      expect(paramsOf(calls, "/usage/summary").length).toBeGreaterThanOrEqual(
        2,
      ),
    )
    for (const params of paramsOf(calls, "/usage/summary")) {
      expect(params.getAll("user_id")).toEqual(["me-id"])
    }
  })

  it("disables Mine, and says why in text a keyboard or touch user can reach", async () => {
    mockApi({ members: [], workspace: "ws-1" })
    renderPage(
      <>
        <ObserveScopeCaption />
        <ObserveFilterBar view="requests" />
        <UrlProbe />
      </>,
      "/requests",
    )
    const reason =
      "Your account has no usage identity in this organization yet, so Mine is off."

    const mine = await screen.findByRole("button", { name: "Mine" })
    await waitFor(() => expect(mine).toHaveAttribute("aria-disabled", "true"))
    // Focusable, not natively disabled, and described by visible text rather
    // than a hover-only `title`.
    expect(mine).not.toBeDisabled()
    expect(mine).not.toHaveAttribute("title")
    expect(screen.getByText(reason)).toBeVisible()
    expect(mine).toHaveAccessibleDescription(reason)
    expect(screen.getByRole("button", { name: "Everyone" })).toHaveAttribute(
      "aria-pressed",
      "true",
    )

    await userEvent.setup().click(mine)
    expect(urlParams().has("scope")).toBe(false)
  })

  it("says nothing about Mine where the caller has an identity", async () => {
    mockApi({ members: [ME], workspace: "ws-1" })
    renderPage(<ObserveScopeCaption />, "/requests")

    const mine = await screen.findByRole("button", { name: "Mine" })
    expect(mine).not.toHaveAttribute("aria-disabled")
    expect(screen.queryByText(/so Mine is off/)).toBeNull()
  })
})

describe("the Overview layout", () => {
  const tabs = () =>
    within(screen.getByRole("navigation", { name: "Overview views" }))

  it("carries the scope to every other view", async () => {
    mockApi({ members: [ME], workspace: "ws-1" })
    renderPage(<OverviewIndex />, "/requests?scope=mine")

    await screen.findByRole("navigation", { name: "Overview views" })
    const link = tabs().getByRole("link", { name: "Sessions" })
    const href = new URL(link.getAttribute("href") ?? "", "http://localhost")
    expect(href.searchParams.get("scope")).toBe("mine")
  })

  it("names whose requests it shows, with the shared-key hint under Mine", async () => {
    mockApi(MEMBER)
    renderPage(<OverviewIndex />, "/")

    expect(await screen.findByText("Your requests in Production")).toBeVisible()
    expect(
      screen.getByText(
        "Requests made with a shared key count toward the key's owner.",
      ),
    ).toBeVisible()
  })

  it("says Everyone in the workspace for a caller who manages it", async () => {
    mockApi({ members: [ME], workspace: "ws-1" })
    renderPage(<OverviewIndex />, "/")

    expect(await screen.findByText("Everyone in Production")).toBeVisible()
    expect(screen.queryByText(/shared key count/)).toBeNull()
  })

  it("says Everyone on this gateway when no workspace is selected", async () => {
    mockApi({ members: [ME] })
    renderPage(<OverviewIndex />, "/")

    expect(await screen.findByText("Everyone on this gateway")).toBeVisible()
  })

  it("asks for no usage before the scope is known", async () => {
    const { mock, calls } = mockApi(MEMBER)
    const answer = mock.getMockImplementation()
    let release = () => {}
    const gate = new Promise<void>((resolve) => {
      release = resolve
    })
    mock.mockImplementation(async (input, init) => {
      await gate
      return answer?.(input, init) as Promise<Response>
    })
    renderPage(<OverviewIndex />, "/requests")
    await new Promise((resolve) => setTimeout(resolve, 50))
    expect(calls.filter((call) => call.url.includes("/usage"))).toEqual([])

    release()
    await waitFor(() =>
      expect(paramsOf(calls, "/usage/summary").length).toBeGreaterThan(0),
    )
    // Every read that went out names the caller; none carries the team.
    for (const params of paramsOf(calls, "/usage/summary")) {
      expect(params.getAll("user_id")).toEqual(["me-id"])
    }
  })
})

describe("the Observe views under Mine", () => {
  it("reads the request log with the owner id", async () => {
    const { calls } = mockApi({ ...MEMBER, rows: [entry()] })
    renderPage(<ActivityPage embedded />, "/requests")

    await waitFor(() =>
      expect(paramsOf(calls, "/usage/count").length).toBeGreaterThan(0),
    )
    for (const params of paramsOf(calls, "/usage")) {
      expect(params.getAll("user_id")).toEqual(["me-id"])
    }
  })

  it("reads the sessions with the owner id", async () => {
    const { calls } = mockApi(MEMBER)
    renderPage(<SessionsPage embedded />, "/sessions")

    await waitFor(() =>
      expect(paramsOf(calls, "/traces/count").length).toBeGreaterThan(0),
    )
    expect(paramsOf(calls, "/traces/count")[0]?.getAll("user_id")).toEqual([
      "me-id",
    ])
  })

  it("reads the usage with the owner id", async () => {
    const { calls } = mockApi(MEMBER)
    renderPage(<UsagePage embedded />, "/")

    await waitFor(() =>
      expect(paramsOf(calls, "/usage/summary").length).toBeGreaterThan(0),
    )
    for (const params of paramsOf(calls, "/usage/summary")) {
      expect(params.getAll("user_id")).toEqual(["me-id"])
    }
  })

  it("leaves the organization usage page on the URL's own users", async () => {
    const { calls } = mockApi({ ...MEMBER, role: "owner" })
    renderPage(
      <UsagePage scope="organization" />,
      "/organization/usage?user_id=alice",
    )

    await waitFor(() =>
      expect(paramsOf(calls, "/usage/summary").length).toBeGreaterThan(0),
    )
    // The user picker's own suggestion read leaves its filter out; the rest
    // follow the URL, never an owner id.
    const asked = paramsOf(calls, "/usage/summary").flatMap((params) =>
      params.getAll("user_id"),
    )
    expect(asked).toContain("alice")
    expect(asked).not.toContain("me-id")
  })
})
