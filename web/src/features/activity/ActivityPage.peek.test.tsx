import { useRouter, useRouterState } from "@tanstack/react-router"
import { act, screen, waitFor, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, describe, expect, it, vi } from "vitest"

import { ActivityPage } from "@/features/activity/ActivityPage"
import { entry, mockApi, renderPage } from "@/tests/activity"

afterEach(() => {
  vi.restoreAllMocks()
})

// What the router holds, and its Back, read from inside it.
const location: {
  search: Record<string, unknown>
  back: () => void
  filter: (tool: string) => Promise<void>
} = {
  search: {},
  back: () => undefined,
  filter: async () => undefined,
}
function LocationProbe() {
  const router = useRouter()
  location.search = useRouterState({ select: (state) => state.location.search })
  location.back = () => router.history.back()
  location.filter = (tool) =>
    router.navigate({ to: ".", search: (prev) => ({ ...prev, tool }) })
  return null
}

const ROWS = [
  entry({ id: "r1", model: "gpt-4o" }),
  entry({ id: "r2", model: "claude-sonnet" }),
  entry({ id: "r3", model: "mistral-small" }),
]

function renderLog(route = "/activity", rows = ROWS, total?: number) {
  mockApi({ rows, total })
  renderPage(
    <>
      <ActivityPage />
      <LocationProbe />
    </>,
    route,
  )
}

const panel = () =>
  screen.getByRole("complementary", { name: "Request detail" })
const findPanel = () =>
  screen.findByRole("complementary", { name: "Request detail" })
const queryPanel = () =>
  screen.queryByRole("complementary", { name: "Request detail" })
const shownModel = () =>
  within(panel()).getByRole("heading", { level: 2 }).textContent
const rowOf = async (model: string) =>
  (await screen.findByText(model)).closest("tr")!

describe("ActivityPage request panel", () => {
  it("opens a clicked row in the panel, names it in the URL and marks the row", async () => {
    const user = userEvent.setup()
    renderLog()

    const row = await rowOf("claude-sonnet")
    await user.click(row)

    await findPanel()
    expect(shownModel()).toBe("claude-sonnet")
    expect(location.search.request).toBe("r2")
    // Focus lands in the panel, on the record's name.
    expect(within(panel()).getByRole("heading", { level: 2 })).toHaveFocus()
    // jsdom computes no layout, so the open row is pinned as the class the
    // stylesheet washes its cells with.
    expect(row).toHaveClass("is-peeked")
    expect(await rowOf("gpt-4o")).not.toHaveClass("is-peeked")
  })

  it("opens the request a link names", async () => {
    renderLog("/activity?request=r3")

    await findPanel()
    expect(shownModel()).toBe("mistral-small")
  })

  it("walks the page with previous and next, and with K and J", async () => {
    const user = userEvent.setup()
    renderLog("/activity?request=r1")
    await findPanel()

    expect(
      within(panel()).getByRole("button", { name: "Previous request" }),
    ).toBeDisabled()
    await user.click(
      within(panel()).getByRole("button", { name: "Next request" }),
    )
    expect(shownModel()).toBe("claude-sonnet")
    expect(location.search.request).toBe("r2")

    await user.keyboard("j")
    expect(shownModel()).toBe("mistral-small")
    expect(
      within(panel()).getByRole("button", { name: "Next request" }),
    ).toBeDisabled()

    await user.keyboard("k")
    expect(shownModel()).toBe("claude-sonnet")
  })

  it("closes from its Close control, and drops the request from the URL", async () => {
    const user = userEvent.setup()
    renderLog("/activity?request=r2")
    await findPanel()

    await user.click(within(panel()).getByRole("button", { name: "Close" }))

    await waitFor(() => expect(queryPanel()).toBeNull())
    expect(location.search.request).toBeUndefined()
  })

  it("closes on Escape", async () => {
    const user = userEvent.setup()
    renderLog("/activity?request=r2")
    await findPanel()

    await user.keyboard("{Escape}")

    await waitFor(() => expect(queryPanel()).toBeNull())
    expect(location.search.request).toBeUndefined()
  })

  it("closes on Back after a row was opened", async () => {
    const user = userEvent.setup()
    renderLog()

    await user.click(await rowOf("gpt-4o"))
    await findPanel()

    act(() => location.back())

    await waitFor(() => expect(queryPanel()).toBeNull())
    expect(location.search.request).toBeUndefined()
  })

  it("says so, and can dismiss it, when the linked request is not on the page", async () => {
    const user = userEvent.setup()
    renderLog("/activity?request=gone")
    await screen.findByText("gpt-4o")

    expect(
      await screen.findByText("That request is not in the current results."),
    ).toBeInTheDocument()
    expect(queryPanel()).toBeNull()

    await user.click(screen.getByRole("button", { name: "Dismiss" }))

    await waitFor(() =>
      expect(
        screen.queryByText("That request is not in the current results."),
      ).toBeNull(),
    )
    expect(location.search.request).toBeUndefined()
  })

  it("drops the request from the URL when the page changes", async () => {
    const user = userEvent.setup()
    renderLog(
      "/activity?request=r2",
      Array.from({ length: 50 }, (_, i) => entry({ id: `r${i + 1}` })),
      200,
    )
    await findPanel()

    await user.click(screen.getByRole("button", { name: "Next page" }))

    await waitFor(() => expect(queryPanel()).toBeNull())
    expect(location.search.request).toBeUndefined()
  })

  it("drops the request from the URL when the filters change", async () => {
    renderLog("/activity?request=r2&tool=web_search")
    await findPanel()

    // Awaited: the navigation is a promise, and an act left pending stops
    // every later test's renders from flushing.
    await act(async () => {
      await location.filter("web_fetch")
    })

    await waitFor(() => expect(queryPanel()).toBeNull())
    expect(location.search.request).toBeUndefined()
  })
})
