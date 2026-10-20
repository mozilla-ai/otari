import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { type AnyRouter, useRouter } from "@tanstack/react-router"
import { act, render, screen, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import { AutoUpdate, reloadOnStaleChunk } from "@/app/AutoUpdate"
import { flushRouter, withRouter } from "@/tests/router"

// The build the fake gateway is currently serving. Tests flip it to stand in
// for a deploy landing under an already-open tab.
let servedBuild = "build-a"

function mockGateway() {
  return vi.spyOn(globalThis, "fetch").mockImplementation(async () => {
    return new Response(
      JSON.stringify({ build: servedBuild, version: "1.0.0" }),
      {
        status: 200,
        headers: { "Content-Type": "application/json" },
      },
    )
  })
}

// Object.defineProperty on window.location is not undone by restoreAllMocks, so
// capture and restore the original descriptor to keep the mock from leaking.
let originalLocation: PropertyDescriptor | undefined

// jsdom's location.reload cannot be spied on directly.
function mockReload() {
  const reload = vi.fn()
  originalLocation = Object.getOwnPropertyDescriptor(window, "location")
  Object.defineProperty(window, "location", {
    configurable: true,
    value: { ...window.location, reload },
  })
  return reload
}

async function renderShell() {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  })
  const captured: { router: AnyRouter | null } = { router: null }
  function CaptureRouter() {
    captured.router = useRouter()
    return null
  }
  const Router = withRouter({
    url: "/keys",
    routes: [{ path: "/models", element: <p>Models page</p> }],
  })
  render(
    <QueryClientProvider client={client}>
      <Router>
        <AutoUpdate />
        <CaptureRouter />
      </Router>
    </QueryClientProvider>,
  )
  await flushRouter()
  const router = captured.router
  if (!router) {
    throw new Error("router did not mount")
  }
  return { client, router }
}

// Resolves once the tab has actually observed a build, which is what fixes the
// version it considers itself to be running. Waiting on fetch being *called* is
// not enough: the response may not have landed yet, and the tab would then
// adopt whatever build arrives first, including a newer one.
async function waitForLoadedBuild(client: QueryClient, build: string) {
  await waitFor(() =>
    expect(client.getQueryData(["build"])).toMatchObject({ build }),
  )
}

async function deploy(client: QueryClient) {
  servedBuild = "build-b"
  await act(() => client.refetchQueries({ queryKey: ["build"] }))
  // React Query tells its observers a tick after the cache changes, so wait for
  // the component to have seen the new build.
  await act(() => new Promise((resolve) => setTimeout(resolve, 0)))
}

async function navigate(router: AnyRouter, href: string) {
  await act(async () => {
    router.history.push(href)
  })
}

describe("AutoUpdate", () => {
  beforeEach(() => {
    servedBuild = "build-a"
  })
  afterEach(() => {
    vi.restoreAllMocks()
    if (originalLocation) {
      Object.defineProperty(window, "location", originalLocation)
      originalLocation = undefined
    }
  })

  it("reloads on the next page change once a new build is served", async () => {
    const reload = mockReload()
    mockGateway()
    const { client, router } = await renderShell()
    await waitForLoadedBuild(client, "build-a")

    await deploy(client)
    expect(reload).not.toHaveBeenCalled()

    await navigate(router, "/models")
    expect(reload).toHaveBeenCalledTimes(1)
  })

  it("navigates client-side while the build is unchanged", async () => {
    const reload = mockReload()
    mockGateway()
    const { client, router } = await renderShell()
    await waitForLoadedBuild(client, "build-a")
    await act(() => client.refetchQueries({ queryKey: ["build"] }))

    await navigate(router, "/models")

    expect(await screen.findByText("Models page")).toBeInTheDocument()
    expect(reload).not.toHaveBeenCalled()
  })

  it("keeps a filter change on the same page client-side", async () => {
    // A search-param change is a navigation too, and reloading on one would
    // throw away the page the operator is still on.
    const reload = mockReload()
    mockGateway()
    const { client, router } = await renderShell()
    await waitForLoadedBuild(client, "build-a")

    await deploy(client)
    await navigate(router, "/keys?status=active")

    expect(reload).not.toHaveBeenCalled()
  })

  it("stays put when the build check fails", async () => {
    const reload = mockReload()
    vi.spyOn(globalThis, "fetch").mockRejectedValue(new Error("offline"))
    const { router } = await renderShell()
    await waitFor(() => expect(globalThis.fetch).toHaveBeenCalled())

    await navigate(router, "/models")

    expect(await screen.findByText("Models page")).toBeInTheDocument()
    expect(reload).not.toHaveBeenCalled()
  })
})

describe("reloadOnStaleChunk", () => {
  let uninstall: (() => void) | undefined

  beforeEach(() => {
    sessionStorage.clear()
  })
  afterEach(() => {
    uninstall?.()
    uninstall = undefined
    vi.restoreAllMocks()
    if (originalLocation) {
      Object.defineProperty(window, "location", originalLocation)
      originalLocation = undefined
    }
  })

  it("reloads when a chunk fails to load", () => {
    const reload = mockReload()
    uninstall = reloadOnStaleChunk()

    window.dispatchEvent(new Event("vite:preloadError"))

    expect(reload).toHaveBeenCalledTimes(1)
  })

  it("does not reload twice in a row, so a broken chunk cannot loop", () => {
    const reload = mockReload()
    uninstall = reloadOnStaleChunk()

    window.dispatchEvent(new Event("vite:preloadError"))
    window.dispatchEvent(new Event("vite:preloadError"))

    expect(reload).toHaveBeenCalledTimes(1)
  })
})
