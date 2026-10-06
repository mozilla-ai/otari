import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { act, renderHook, waitFor } from "@testing-library/react"
import type { ReactNode } from "react"
import { afterEach, expect, it, vi } from "vitest"
import { useSendFeedback } from "./feedback"

afterEach(() => vi.restoreAllMocks())

it("does not retry delivery even if the application configures mutation retries", async () => {
  const fetch = vi
    .spyOn(globalThis, "fetch")
    .mockRejectedValue(new TypeError("Offline"))
  const client = new QueryClient({
    defaultOptions: { mutations: { retry: 3 } },
  })
  const wrapper = ({ children }: { children: ReactNode }) => (
    <QueryClientProvider client={client}>{children}</QueryClientProvider>
  )
  const { result } = renderHook(() => useSendFeedback(), { wrapper })
  act(() => result.current.mutate({ message: "Idea" }))
  await waitFor(() => expect(result.current.isError).toBe(true))
  expect(fetch).toHaveBeenCalledTimes(1)
  expect(fetch.mock.calls[0][1]?.signal).toBeInstanceOf(AbortSignal)
})
