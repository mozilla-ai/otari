import { Outlet } from "@tanstack/react-router"
import { screen } from "@testing-library/react"
import { afterEach, describe, expect, it, vi } from "vitest"
import type { TraceSummary } from "@/client"
import { API_ROOT } from "@/shared/api/client"
import { useSessionsSince } from "@/shared/api/traces"
import { AppProviders as Provider } from "@/tests/providers"
import { renderWithRouter } from "@/tests/router"
import { traceSummary } from "@/tests/traceFixtures"

/** The hook's rows, as the session ids it returned, written into the page. */
function Ids() {
  "use no memo"
  const { data } = useSessionsSince("2026-11-01T00:00:00Z")
  return data ? (
    <output>{data.sessions.map((one) => one.trace_id).join(",")}</output>
  ) : null
}

async function readIds() {
  await renderWithRouter(<Ids />, {
    url: "/sessions",
    shell: (
      <Provider>
        <Outlet />
      </Provider>
    ),
  })
  const text = (await screen.findByRole("status")).textContent ?? ""
  return text.split(",")
}

/**
 * A session list ordered by last activity, newest first, where the session at
 * `touched` is used again after the page at `after` has been served: it moves
 * to the front, as a live session does while the list is being read.
 */
function liveList(total: number, touched: number, after: number) {
  const order = Array.from({ length: total }, (_, index) => `s-${index}`)
  const urls: string[] = []
  let served = 0
  vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
    const url = String(input).slice(API_ROOT.length)
    urls.push(url)
    const query = new URLSearchParams(url.split("?")[1])
    if (url.startsWith("/organizations/me/traces/count")) {
      return Response.json({ count: total })
    }
    const skip = Number(query.get("skip"))
    const limit = Number(query.get("limit"))
    const items: TraceSummary[] = order
      .slice(skip, skip + limit)
      .map((id) => traceSummary({ trace_id: id }))
    served += 1
    if (served === after) {
      const [id] = order.splice(touched, 1)
      order.unshift(id)
    }
    return Response.json({ items, total })
  })
  return urls
}

describe("useSessionsSince, over a list that moves while it is read", () => {
  afterEach(() => {
    vi.restoreAllMocks()
  })

  it("counts every session once", async () => {
    // 250 sessions; the 150th (on the second page) is used again after the
    // first page was served, so every row after it moves down one place.
    liveList(250, 150, 1)
    const ids = await readIds()
    expect(ids).toHaveLength(250)
    expect(new Set(ids).size).toBe(250)
  })

  it("reads a list that fits one page once", async () => {
    const urls = liveList(40, 5, 99)
    expect(await readIds()).toHaveLength(40)
    expect(urls.filter((url) => url.includes("/traces?"))).toHaveLength(1)
  })
})
