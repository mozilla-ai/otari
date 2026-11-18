import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { act, renderHook } from "@testing-library/react"
import { createElement, type ReactNode } from "react"
import { afterEach, describe, expect, it, vi } from "vitest"

import { useDeletePricing, useSetPricing } from "@/shared/api/pricing"
import {
  HOSTED_PROVIDERS,
  ORGANIZATION_PROVIDER_MODELS,
} from "@/shared/api/queryKeys"

afterEach(() => {
  vi.restoreAllMocks()
})

function setup() {
  const client = new QueryClient()
  const invalidate = vi.spyOn(client, "invalidateQueries")
  vi.spyOn(globalThis, "fetch").mockImplementation(async () =>
    Response.json({}),
  )
  const wrapper = ({ children }: { children: ReactNode }) =>
    createElement(QueryClientProvider, { client }, children)
  const keys = () => invalidate.mock.calls.map(([filters]) => filters?.queryKey)
  return { wrapper, keys }
}

describe("deployment price writes", () => {
  it("refresh both provider model lists, which show the deployment's price", async () => {
    const { wrapper, keys } = setup()
    const { result } = renderHook(() => useSetPricing(), { wrapper })

    await act(() =>
      result.current.mutateAsync({
        model_key: "openai:gpt-4o",
        input_price_per_million: 1,
        output_price_per_million: 2,
      }),
    )

    expect(keys()).toEqual(
      expect.arrayContaining([
        [ORGANIZATION_PROVIDER_MODELS],
        [HOSTED_PROVIDERS, "models"],
      ]),
    )
  })

  it("refresh them when a price is deleted too", async () => {
    const { wrapper, keys } = setup()
    const { result } = renderHook(() => useDeletePricing(), { wrapper })

    await act(() =>
      result.current.mutateAsync({ kind: "model", modelKey: "openai:gpt-4o" }),
    )

    expect(keys()).toEqual(
      expect.arrayContaining([
        [ORGANIZATION_PROVIDER_MODELS],
        [HOSTED_PROVIDERS, "models"],
      ]),
    )
  })
})
