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
  isNoOrganizationError,
  useDeploymentOperator,
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

describe("useOrganizationContext for a caller with no organization", () => {
  afterEach(() => {
    vi.restoreAllMocks()
  })

  // A new account that has not accepted its invitation yet. Anything that
  // rereads the context (otari.ai's wallet chip does on every 403) used to put
  // the query back to pending, where the shell hides its rows and then fails
  // open onto the operator ones: the sidebar and top bar flickered with it.
  it("keeps answering no organization while the context is read again", async () => {
    const client = new QueryClient({
      defaultOptions: { queries: { retry: false } },
    })
    let reads = 0
    let release = () => {}
    vi.spyOn(globalThis, "fetch").mockImplementation(async () => {
      reads += 1
      if (reads > 1) {
        await new Promise<void>((resolve) => {
          release = resolve
        })
      }
      return Response.json(
        { detail: "Organization not found" },
        { status: 404 },
      )
    })
    const wrapper = ({ children }: { children: ReactNode }) =>
      createElement(QueryClientProvider, { client }, children)
    const { result } = renderHook(
      () => ({
        context: useOrganizationContext(),
        operator: useDeploymentOperator(),
      }),
      { wrapper },
    )
    await waitFor(() =>
      expect(result.current.operator.answer).toBe("no-organization"),
    )
    expect(isNoOrganizationError(result.current.context.error)).toBe(true)
    expect(result.current.context.data).toBeUndefined()

    act(() => {
      void client.invalidateQueries({ queryKey: ORGANIZATION_CONTEXT })
    })
    await waitFor(() => expect(reads).toBe(2))

    expect(result.current.context.isPending).toBe(false)
    expect(isNoOrganizationError(result.current.context.error)).toBe(true)
    expect(result.current.operator).toEqual({
      answer: "no-organization",
      isOperator: false,
      isSettled: true,
    })
    await act(async () => release())
  })

  it("still reports any other failure as one", async () => {
    const client = new QueryClient({
      defaultOptions: { queries: { retry: false } },
    })
    vi.spyOn(globalThis, "fetch").mockImplementation(async () =>
      Response.json({ detail: "Internal server error" }, { status: 500 }),
    )
    const wrapper = ({ children }: { children: ReactNode }) =>
      createElement(QueryClientProvider, { client }, children)
    const { result } = renderHook(() => useDeploymentOperator(), { wrapper })
    await waitFor(() => expect(result.current.answer).toBe("unavailable"))
  })

  it("reports a failed reread as a failure, not as the 404 before it", async () => {
    const client = new QueryClient({
      defaultOptions: { queries: { retry: false } },
    })
    const statuses = [404, 500]
    vi.spyOn(globalThis, "fetch").mockImplementation(async () =>
      Response.json({ detail: "x" }, { status: statuses.shift() ?? 500 }),
    )
    const wrapper = ({ children }: { children: ReactNode }) =>
      createElement(QueryClientProvider, { client }, children)
    const { result } = renderHook(
      () => ({
        context: useOrganizationContext(),
        operator: useDeploymentOperator(),
      }),
      { wrapper },
    )
    await waitFor(() =>
      expect(result.current.operator.answer).toBe("no-organization"),
    )

    await act(() => client.refetchQueries({ queryKey: ORGANIZATION_CONTEXT }))

    await waitFor(() =>
      expect(result.current.operator.answer).toBe("unavailable"),
    )
    expect(result.current.context.data).toBeUndefined()
    expect(result.current.context.error).toBeInstanceOf(ApiError)
    expect((result.current.context.error as ApiError).status).toBe(500)
  })

  it("answers a refetch the caller awaits the same way", async () => {
    const client = new QueryClient({
      defaultOptions: { queries: { retry: false } },
    })
    vi.spyOn(globalThis, "fetch").mockImplementation(async () =>
      Response.json({ detail: "Organization not found" }, { status: 404 }),
    )
    const wrapper = ({ children }: { children: ReactNode }) =>
      createElement(QueryClientProvider, { client }, children)
    const { result } = renderHook(() => useOrganizationContext(), { wrapper })
    await waitFor(() => expect(result.current.isFetched).toBe(true))

    const refetched = await act(() => result.current.refetch())

    expect(refetched.data).toBeUndefined()
    expect(isNoOrganizationError(refetched.error)).toBe(true)
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
