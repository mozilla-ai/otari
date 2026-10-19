import {
  QueryClient,
  QueryClientProvider,
  useQuery,
} from "@tanstack/react-query"
import { act, renderHook, waitFor } from "@testing-library/react"
import { createElement, type ReactNode } from "react"
import { afterEach, describe, expect, it, vi } from "vitest"

import { ApiError } from "@/shared/api/client"
import {
  deploymentOperatorAnswer,
  useKeepOrganizationContext,
  useOrganizationContext,
  useSwitchOrganization,
} from "@/shared/api/organizations"
import { ORGANIZATION_CONTEXT, workspaceMeta } from "@/shared/api/queryKeys"
import { organizationContext } from "@/tests/fixtures"

describe("deploymentOperatorAnswer", () => {
  it("answers from the context once it has landed", () => {
    expect(
      deploymentOperatorAnswer({
        data: organizationContext({ deployment_operator: true }),
        isFetched: true,
      }),
    ).toBe("operator")
    expect(
      deploymentOperatorAnswer({
        data: organizationContext({ deployment_operator: false }),
        isFetched: true,
      }),
    ).toBe("not-operator")
  })

  it("tells a read still in flight from one that failed", () => {
    expect(deploymentOperatorAnswer({ isFetched: false })).toBe("pending")
    // `isFetched` stays true while a failed read retries, which is what keeps
    // a page from swinging back to its spinner on every attempt.
    expect(deploymentOperatorAnswer({ isFetched: true })).toBe("unavailable")
  })

  it("reads a 404 as no organization, not as a failed read", () => {
    // The one failure that is an answer: an unavailable context fails open
    // onto the operator surface, and a caller in no organization operates
    // nothing.
    expect(
      deploymentOperatorAnswer({
        error: new ApiError(404, "Organization not found"),
        isFetched: true,
      }),
    ).toBe("no-organization")
    expect(
      deploymentOperatorAnswer({
        error: new ApiError(500, "Internal server error"),
        isFetched: true,
      }),
    ).toBe("unavailable")
  })
})

describe("useSwitchOrganization", () => {
  const LEFT = "11111111-1111-1111-1111-111111111111"
  const JOINED = "22222222-2222-2222-2222-222222222222"

  afterEach(() => {
    vi.restoreAllMocks()
  })

  it("holds back only the queries that declare a workspace it left", async () => {
    const client = new QueryClient({
      defaultOptions: { queries: { retry: false } },
    })
    client.setQueryData(
      ORGANIZATION_CONTEXT,
      organizationContext({
        workspace_memberships: [
          { workspace_id: LEFT, name: "Left", role: "member" },
        ],
      }),
    )
    vi.spyOn(globalThis, "fetch").mockImplementation(async () =>
      Response.json(organizationContext()),
    )
    const reads = { left: 0, trace: 0 }
    const wrapper = ({ children }: { children: ReactNode }) =>
      createElement(QueryClientProvider, { client }, children)
    const { result } = renderHook(
      () => {
        useQuery({
          queryKey: ["left", LEFT],
          meta: workspaceMeta(LEFT),
          queryFn: () => ++reads.left,
        })
        // A trace whose own id happens to equal the left workspace's, read in
        // the workspace just joined: it is refetched like any other query.
        useQuery({
          queryKey: ["trace", LEFT, JOINED],
          meta: workspaceMeta(JOINED),
          queryFn: () => ++reads.trace,
        })
        return useSwitchOrganization()
      },
      { wrapper },
    )
    await waitFor(() => expect(reads).toEqual({ left: 1, trace: 1 }))

    await act(() => result.current.mutateAsync("org-there"))

    await waitFor(() => expect(reads.trace).toBe(2))
    expect(reads.left).toBe(1)
    expect(client.getQueryState(["left", LEFT])?.isInvalidated).toBe(true)
  })
})

describe("the kept organization context", () => {
  const KEPT = "otari.dashboard.organizationContext"
  const kept = organizationContext({
    deployment_operator: false,
    organization: { ...organizationContext().organization, name: "Kept" },
  })
  const current = organizationContext({
    organization: { ...organizationContext().organization, name: "Current" },
  })

  function createWrapper(client: QueryClient) {
    return ({ children }: { children: ReactNode }) =>
      createElement(QueryClientProvider, { client }, children)
  }

  function createClient() {
    return new QueryClient({ defaultOptions: { queries: { retry: false } } })
  }

  afterEach(() => {
    vi.restoreAllMocks()
    window.localStorage.clear()
  })

  it("stands in while the current context loads, then gives way to it", async () => {
    window.localStorage.setItem(
      KEPT,
      JSON.stringify({ origin: "", context: kept }),
    )
    let answer: (response: Response) => void = () => {}
    vi.spyOn(globalThis, "fetch").mockImplementation(
      () => new Promise((resolve) => (answer = resolve)),
    )

    const { result } = renderHook(() => useOrganizationContext(), {
      wrapper: createWrapper(createClient()),
    })

    expect(result.current.data?.organization.name).toBe("Kept")
    expect(result.current.isPlaceholderData).toBe(true)
    // Settled at once, so nothing that waits on the answer waits a round trip.
    expect(deploymentOperatorAnswer(result.current)).toBe("not-operator")

    act(() => answer(Response.json(current)))
    await waitFor(() =>
      expect(result.current.data?.organization.name).toBe("Current"),
    )
    expect(result.current.isPlaceholderData).toBe(false)
  })

  it("is not offered to another API origin, or to a read that is disabled", () => {
    window.localStorage.setItem(
      KEPT,
      JSON.stringify({ origin: "https://eu.example.com", context: kept }),
    )
    vi.spyOn(globalThis, "fetch").mockImplementation(
      () => new Promise(() => {}),
    )
    const elsewhere = renderHook(() => useOrganizationContext(), {
      wrapper: createWrapper(createClient()),
    })
    expect(elsewhere.result.current.data).toBeUndefined()

    window.localStorage.setItem(
      KEPT,
      JSON.stringify({ origin: "", context: kept }),
    )
    const disabled = renderHook(() => useOrganizationContext(false), {
      wrapper: createWrapper(createClient()),
    })
    expect(disabled.result.current.data).toBeUndefined()
  })

  it("offers nothing for an entry that no longer parses, however often it is read", () => {
    // In particular not whichever entry was read before it.
    window.localStorage.setItem(
      KEPT,
      JSON.stringify({ origin: "", context: kept }),
    )
    vi.spyOn(globalThis, "fetch").mockImplementation(
      () => new Promise(() => {}),
    )
    const readable = renderHook(() => useOrganizationContext(), {
      wrapper: createWrapper(createClient()),
    })
    expect(readable.result.current.data?.organization.name).toBe("Kept")
    window.localStorage.setItem(KEPT, "{not json")
    for (let read = 0; read < 2; read++) {
      const unreadable = renderHook(() => useOrganizationContext(), {
        wrapper: createWrapper(createClient()),
      })
      expect(unreadable.result.current.data).toBeUndefined()
    }
  })

  it("reports a failed read instead of the kept context, and forgets it", async () => {
    // A caller removed from their organization since the last load: the
    // kept answer must not survive the 404, nor come back on a refetch.
    window.localStorage.setItem(
      KEPT,
      JSON.stringify({ origin: "", context: kept }),
    )
    const fetchMock = vi
      .spyOn(globalThis, "fetch")
      .mockImplementation(async () =>
        Response.json({ detail: "No active membership" }, { status: 404 }),
      )

    const { result } = renderHook(() => useOrganizationContext(), {
      wrapper: createWrapper(createClient()),
    })

    await waitFor(() => expect(result.current.isError).toBe(true))
    expect(result.current.data).toBeUndefined()
    expect(deploymentOperatorAnswer(result.current)).toBe("no-organization")
    expect(window.localStorage.getItem(KEPT)).toBeNull()

    fetchMock.mockImplementation(() => new Promise(() => {}))
    act(() => {
      void result.current.refetch()
    })
    await waitFor(() => expect(result.current.isFetching).toBe(true))
    expect(result.current.data).toBeUndefined()
  })

  it("keeps a hosted deployment's answer, including one a switch wrote", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation(async () =>
      Response.json(current),
    )
    const client = createClient()

    renderHook(() => useKeepOrganizationContext(true), {
      wrapper: createWrapper(client),
    })

    await waitFor(() =>
      expect(JSON.parse(window.localStorage.getItem(KEPT) ?? "null")).toEqual({
        origin: "",
        context: current,
      }),
    )

    act(() => client.setQueryData(ORGANIZATION_CONTEXT, kept))
    await waitFor(() =>
      expect(
        JSON.parse(window.localStorage.getItem(KEPT) ?? "null")?.context
          .organization.name,
      ).toBe("Kept"),
    )
  })

  it("keeps nothing for a deployment that is not hosted", async () => {
    window.localStorage.setItem(
      KEPT,
      JSON.stringify({ origin: "", context: kept }),
    )
    vi.spyOn(globalThis, "fetch").mockImplementation(async () =>
      Response.json(current),
    )

    const { result } = renderHook(
      () => {
        useKeepOrganizationContext(false)
        return useOrganizationContext()
      },
      { wrapper: createWrapper(createClient()) },
    )

    await waitFor(() =>
      expect(result.current.data?.organization.name).toBe("Current"),
    )
    expect(window.localStorage.getItem(KEPT)).toBeNull()
  })
})
