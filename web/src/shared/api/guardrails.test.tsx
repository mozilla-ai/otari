import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { renderHook, waitFor } from "@testing-library/react"
import type { ReactNode } from "react"
import { afterEach, describe, expect, it, vi } from "vitest"

import {
  useBuiltInGuardrailCatalog,
  useCreateOrganizationGuardrailDefinition,
  useDeleteOrganizationGuardrailDefinition,
  useOrganizationGuardrailDefinitions,
  useUpdateOrganizationGuardrailDefinition,
} from "@/shared/api/guardrails"
import {
  BUILTIN_GUARDRAIL_CATALOG,
  GUARDRAIL_PROFILES,
  ORGANIZATION_GUARDRAIL_DEFINITIONS,
  ORGANIZATION_GUARDRAILS,
} from "@/shared/api/queryKeys"
import { useUpdateToolSettings } from "@/shared/api/tools"

function respond(body: unknown, status = 200) {
  return vi.spyOn(globalThis, "fetch").mockResolvedValue(
    new Response(JSON.stringify(body), {
      status,
      headers: { "Content-Type": "application/json" },
    }),
  )
}

function harness() {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  })
  const invalidated = vi.spyOn(client, "invalidateQueries")
  const wrapper = ({ children }: { children: ReactNode }) => (
    <QueryClientProvider client={client}>{children}</QueryClientProvider>
  )
  const keys = () =>
    invalidated.mock.calls.map(([options]) => options?.queryKey)
  return { wrapper, keys }
}

function requested(fetch: ReturnType<typeof respond>) {
  const [input, init] = fetch.mock.calls[0] ?? []
  return { url: String(input), method: init?.method ?? "GET" }
}

afterEach(() => {
  vi.restoreAllMocks()
})

describe("useBuiltInGuardrailCatalog", () => {
  it("reads the built-in catalog, not the remote profile list", async () => {
    const fetch = respond({ guardrails: [] })
    const { wrapper } = harness()

    const { result } = renderHook(() => useBuiltInGuardrailCatalog(), {
      wrapper,
    })

    await waitFor(() => expect(result.current.isSuccess).toBe(true))
    expect(requested(fetch).url).toMatch(
      /\/tool-settings\/guardrails\/catalog$/,
    )
  })

  it("survives a tool-settings save that re-dials the remote profiles", async () => {
    // Pointing `guardrails_url` at another service changes the remote answer
    // and not the installed library, so only one of the two catalogs goes.
    respond({ fields: [] })
    const { wrapper, keys } = harness()

    const { result } = renderHook(() => useUpdateToolSettings(), { wrapper })
    result.current.mutate({ guardrails_url: "http://elsewhere" } as never)

    await waitFor(() => expect(result.current.isSuccess).toBe(true))
    expect(keys()).toContainEqual([GUARDRAIL_PROFILES])
    expect(keys()).not.toContainEqual([BUILTIN_GUARDRAIL_CATALOG])
  })
})

describe("useOrganizationGuardrailDefinitions", () => {
  it("lists the organization's definitions", async () => {
    const fetch = respond({ data: [], count: 0 })
    const { wrapper } = harness()

    const { result } = renderHook(() => useOrganizationGuardrailDefinitions(), {
      wrapper,
    })

    await waitFor(() => expect(result.current.isSuccess).toBe(true))
    expect(result.current.data).toEqual([])
    expect(requested(fetch).url).toMatch(
      /\/organizations\/me\/guardrail-definitions$/,
    )
  })
})

describe("definition writes", () => {
  it("create posts the body and refreshes only the definitions", async () => {
    const fetch = respond({ id: "d1" }, 201)
    const { wrapper, keys } = harness()

    const { result } = renderHook(
      () => useCreateOrganizationGuardrailDefinition(),
      { wrapper },
    )
    result.current.mutate({
      name: "prod-lakera",
      guardrail_name: "lakera_guard",
    })

    await waitFor(() => expect(result.current.isSuccess).toBe(true))
    expect(requested(fetch).method).toBe("POST")
    expect(keys()).toContainEqual([ORGANIZATION_GUARDRAIL_DEFINITIONS])
    // A definition write changes no mandate field.
    expect(keys()).not.toContainEqual([ORGANIZATION_GUARDRAILS])
  })

  it("update patches the named definition", async () => {
    const fetch = respond({ id: "d 1" })
    const { wrapper, keys } = harness()

    const { result } = renderHook(
      () => useUpdateOrganizationGuardrailDefinition(),
      { wrapper },
    )
    result.current.mutate({ definitionId: "d 1", body: { enabled: false } })

    await waitFor(() => expect(result.current.isSuccess).toBe(true))
    expect(requested(fetch)).toEqual({
      url: expect.stringMatching(/\/guardrail-definitions\/d%201$/),
      method: "PATCH",
    })
    expect(keys()).toContainEqual([ORGANIZATION_GUARDRAIL_DEFINITIONS])
  })

  it("delete removes the named definition", async () => {
    const fetch = respond({ message: "deleted" })
    const { wrapper, keys } = harness()

    const { result } = renderHook(
      () => useDeleteOrganizationGuardrailDefinition(),
      { wrapper },
    )
    result.current.mutate("d1")

    await waitFor(() => expect(result.current.isSuccess).toBe(true))
    expect(requested(fetch).method).toBe("DELETE")
    expect(keys()).toContainEqual([ORGANIZATION_GUARDRAIL_DEFINITIONS])
  })
})
