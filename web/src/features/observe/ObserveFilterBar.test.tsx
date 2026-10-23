import { useLocation } from "@tanstack/react-router"
import { screen, waitFor } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, describe, expect, it, vi } from "vitest"

import { API_ROOT } from "@/shared/api/client"
import {
  entry,
  jsonResponse,
  mockApi,
  operatorContext,
  renderPage,
} from "@/tests/activity"
import { pickOption, selectTrigger } from "@/tests/select"
import { ObserveFilterBar } from "./ObserveFilterBar"
import type { ObserveView } from "./observeSearch"

afterEach(() => {
  vi.restoreAllMocks()
})

function UrlProbe() {
  const { searchStr } = useLocation()
  return <output aria-label="Search in the URL">{searchStr}</output>
}

const PATHS: Record<ObserveView, string> = {
  usage: "/",
  requests: "/requests",
  sessions: "/sessions",
}

async function renderBar(view: ObserveView, search = "") {
  renderPage(
    <>
      <ObserveFilterBar view={view} />
      <UrlProbe />
    </>,
    `${PATHS[view]}${search}`,
  )
  await screen.findByRole("button", { name: "Add filter" })
}

// By element rather than role: an open picker hides everything outside its
// popover from the accessibility tree.
const urlParams = () =>
  new URLSearchParams(document.querySelector("output")?.textContent ?? "")

const summaryReads = (calls: ReturnType<typeof mockApi>["calls"]) =>
  calls
    .filter((call) => call.url.includes("/usage/summary"))
    .map((call) => new URL(call.url, "http://localhost"))

const reading = (
  calls: ReturnType<typeof mockApi>["calls"],
  dimension: string,
) => summaryReads(calls).find((url) => url.search.includes(dimension))

const HOUR_MS = 3_600_000
const startAge = (url: URL | undefined) =>
  Date.now() - Date.parse(url?.searchParams.get("start_date") ?? "")

describe("ObserveFilterBar options", () => {
  it("reads each view's own window", async () => {
    const { calls } = mockApi({ rows: [entry()] })
    await renderBar("requests", "?range=7d")

    await waitFor(() => expect(reading(calls, "user")).toBeDefined())
    expect(startAge(reading(calls, "model"))).toBeGreaterThan(167 * HOUR_MS)
    expect(startAge(reading(calls, "model"))).toBeLessThan(169 * HOUR_MS)
    expect(startAge(reading(calls, "user"))).toBeGreaterThan(167 * HOUR_MS)
  })

  it("opens Usage on its 30 days and Sessions on 24 hours", async () => {
    const charts = mockApi({ rows: [entry()] })
    await renderBar("usage")
    await waitFor(() => expect(reading(charts.calls, "user")).toBeDefined())
    expect(startAge(reading(charts.calls, "user"))).toBeGreaterThan(
      719 * HOUR_MS,
    )

    const sessions = mockApi({ rows: [entry()] })
    await renderBar("sessions")
    await waitFor(() => expect(reading(sessions.calls, "user")).toBeDefined())
    expect(startAge(reading(sessions.calls, "user"))).toBeLessThan(25 * HOUR_MS)
  })

  it("prefers explicit bounds to the range", async () => {
    const { calls } = mockApi({ rows: [entry()] })
    await renderBar(
      "requests",
      "?range=7d&start_date=2026-10-01T00:00:00.000Z&end_date=2026-10-02T00:00:00.000Z",
    )

    await waitFor(() => expect(reading(calls, "user")).toBeDefined())
    const read = reading(calls, "user")
    expect(read?.searchParams.get("start_date")).toBe(
      "2026-10-01T00:00:00.000Z",
    )
    expect(read?.searchParams.get("end_date")).toBe("2026-10-02T00:00:00.000Z")
  })

  it("reads the deployment's usage for an operator and the organization's otherwise", async () => {
    const operator = mockApi({ rows: [entry()], deploymentOperator: true })
    await renderBar("requests")
    await waitFor(() => expect(reading(operator.calls, "user")).toBeDefined())
    expect(reading(operator.calls, "user")?.pathname).toBe(
      `${API_ROOT}/usage/summary`,
    )

    const member = mockApi({ rows: [entry()], deploymentOperator: false })
    await renderBar("requests")
    await waitFor(() => expect(reading(member.calls, "user")).toBeDefined())
    expect(reading(member.calls, "user")?.pathname).toBe(
      `${API_ROOT}/organizations/me/usage/summary`,
    )
  })

  it("scopes the reads to the selected workspace", async () => {
    const { calls, mock } = mockApi({ rows: [entry()] })
    const answer = mock.getMockImplementation() as typeof fetch
    mock.mockImplementation(async (input, init) =>
      String(input).endsWith(`${API_ROOT}/organizations/me`)
        ? jsonResponse({
            ...(await operatorContext().json()),
            workspace_memberships: [
              { workspace_id: "ws-1", name: "Platform team", role: "owner" },
            ],
          })
        : answer(input, init),
    )
    await renderBar("requests")

    await waitFor(() =>
      expect(
        summaryReads(calls).some(
          (url) => url.searchParams.get("workspace_id") === "ws-1",
        ),
      ).toBe(true),
    )
  })

  it("leaves each picker's own filter out of its list", async () => {
    // The model list stays narrowed by the user and key, and the user and key
    // lists by the model, but none by itself, or a picked value would be the
    // only one left on offer.
    const { calls } = mockApi({ rows: [entry()] })
    await renderBar("requests", "?model=gpt-4o&user_id=alice&api_key_id=key-1")

    await waitFor(() => expect(reading(calls, "user")).toBeDefined())
    const model = reading(calls, "dimensions=model")
    expect(model?.searchParams.getAll("user_id")).toEqual(["alice"])
    expect(model?.searchParams.getAll("api_key_id")).toEqual(["key-1"])
    expect(model?.searchParams.has("model")).toBe(false)

    const entities = reading(calls, "dimensions=user")
    expect(entities?.searchParams.getAll("model")).toEqual(["gpt-4o"])
    expect(entities?.searchParams.has("user_id")).toBe(false)
    expect(entities?.searchParams.has("api_key_id")).toBe(false)
  })

  it("lists the models and users of the window", async () => {
    const user = userEvent.setup()
    mockApi({ rows: [entry({ model: "claude-x" })] })
    await renderBar("requests")

    await user.click(screen.getByRole("combobox", { name: "Model" }))
    expect(
      await screen.findByRole("option", { name: "claude-x" }),
    ).toBeInTheDocument()
  })
})

describe("ObserveFilterBar writing the URL", () => {
  it("adds a picked value and returns to the first page", async () => {
    const user = userEvent.setup()
    mockApi({ rows: [entry()] })
    await renderBar("requests", "?user_id=alice&page=3")

    await user.click(screen.getByRole("combobox", { name: "User" }))
    await user.click(await screen.findByRole("option", { name: /bob/ }))

    await waitFor(() =>
      expect(urlParams().getAll("user_id")).toEqual(["alice", "bob"]),
    )
    expect(urlParams().has("page")).toBe(false)
  })

  it("takes a free-text model on Enter", async () => {
    const user = userEvent.setup()
    mockApi({ rows: [entry()] })
    await renderBar("requests")

    const input = screen.getByRole("combobox", { name: "Model" })
    await user.click(input)
    await user.type(input, "some-unlisted-model")
    await user.keyboard("{Enter}")

    await waitFor(() =>
      expect(urlParams().getAll("model")).toEqual(["some-unlisted-model"]),
    )
  })

  it("writes the status and returns to the first page", async () => {
    const user = userEvent.setup()
    mockApi({ rows: [entry()] })
    await renderBar("requests", "?page=2")

    await pickOption(user, "Status", "Error")

    await waitFor(() => expect(urlParams().get("status")).toBe("error"))
    expect(urlParams().has("page")).toBe(false)
  })

  it("names a status the URL invented rather than react-aria's placeholder", async () => {
    mockApi({ rows: [entry()] })
    await renderBar("requests", "?status=bogus")

    await waitFor(() =>
      expect(selectTrigger("Status")).toHaveTextContent("bogus"),
    )
    expect(selectTrigger("Status")).not.toHaveTextContent("Select an item")
  })
})

describe("ObserveFilterBar chips", () => {
  it("shows a chip for every applied filter, each clearing only itself", async () => {
    const user = userEvent.setup()
    mockApi({ rows: [entry()] })
    await renderBar(
      "requests",
      "?user_id=alice&user_id=bob&model=gpt-4o&api_key_id=key-1&status=error&page=2",
    )

    for (const name of [
      "Remove User filter alice",
      "Remove User filter bob",
      "Remove Model filter gpt-4o",
      "Remove API key filter key-1",
      "Remove Status filter",
    ]) {
      expect(screen.getByRole("button", { name })).toBeInTheDocument()
    }

    await user.click(
      screen.getByRole("button", { name: "Remove User filter alice" }),
    )

    await waitFor(() => expect(urlParams().getAll("user_id")).toEqual(["bob"]))
    expect(urlParams().getAll("model")).toEqual(["gpt-4o"])
    expect(urlParams().get("status")).toBe("error")
    expect(urlParams().has("page")).toBe(false)
  })

  it("clears every shared filter at once and leaves the window alone", async () => {
    const user = userEvent.setup()
    mockApi({ rows: [entry()] })
    await renderBar(
      "requests",
      "?range=7d&user_id=alice&model=gpt-4o&api_key_id=key-1&status=error",
    )

    await user.click(screen.getByRole("button", { name: "Clear all" }))

    await waitFor(() => expect(urlParams().has("model")).toBe(false))
    expect(urlParams().has("user_id")).toBe(false)
    expect(urlParams().has("api_key_id")).toBe(false)
    expect(urlParams().has("status")).toBe(false)
    expect(urlParams().get("range")).toBe("7d")
  })

  it("offers no Clear all while nothing is applied", async () => {
    mockApi({ rows: [entry()] })
    await renderBar("requests")

    expect(
      await screen.findByRole("button", { name: "Add filter" }),
    ).toBeInTheDocument()
    expect(
      screen.queryByRole("button", { name: "Clear all" }),
    ).not.toBeInTheDocument()
  })
})

describe("ObserveFilterBar status by view", () => {
  it("applies every status to Requests", async () => {
    const { calls } = mockApi({ rows: [entry()] })
    await renderBar("requests", "?status=absorbed")

    expect(
      screen.getByRole("button", { name: "Remove Status filter" }),
    ).toBeInTheDocument()
    await waitFor(() => expect(reading(calls, "user")).toBeDefined())
    expect(reading(calls, "user")?.searchParams.get("status")).toBe("absorbed")
  })

  it("applies an outcome to Sessions", async () => {
    const { calls } = mockApi({ rows: [entry()] })
    await renderBar("sessions", "?status=error")

    await waitFor(() => expect(reading(calls, "user")).toBeDefined())
    expect(reading(calls, "user")?.searchParams.get("status")).toBe("error")
    expect(screen.queryByText(/not applied/)).not.toBeInTheDocument()
  })

  it("reads a status with no session meaning as not applied, and still removable", async () => {
    const user = userEvent.setup()
    const { calls } = mockApi({ rows: [entry()] })
    await renderBar("sessions", "?status=absorbed")

    expect(
      await screen.findByText(/absorbed \(not applied to sessions\)/),
    ).toBeInTheDocument()
    await waitFor(() => expect(reading(calls, "user")).toBeDefined())
    expect(reading(calls, "user")?.searchParams.has("status")).toBe(false)

    await user.click(
      screen.getByRole("button", { name: "Remove Status filter" }),
    )
    await waitFor(() => expect(urlParams().has("status")).toBe(false))
  })

  it("offers Usage no status control, and reads one it was handed as not applied", async () => {
    mockApi({ rows: [entry()] })
    await renderBar("usage", "?status=error")

    expect(
      await screen.findByText(/error \(not applied to usage\)/),
    ).toBeInTheDocument()
    expect(
      screen.queryByRole("button", { name: /Status$/ }),
    ).not.toBeInTheDocument()
  })
})
