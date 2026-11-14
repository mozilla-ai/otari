import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { act, renderHook } from "@testing-library/react"
import type { ReactNode } from "react"
import { afterEach, expect, it, vi } from "vitest"

import { useMcpSignIn } from "@/features/tools/useMcpSignIn"

afterEach(() => {
  vi.restoreAllMocks()
})

it("closes the blank popup when the page goes before the sign-in starts", () => {
  // The start never answers, as when the person leaves while it is on its way.
  vi.spyOn(globalThis, "fetch").mockImplementation(
    () => new Promise<Response>(() => {}),
  )
  const popup = { closed: false, close: vi.fn() }
  const client = new QueryClient()
  const { result, unmount } = renderHook(
    () =>
      useMcpSignIn({
        workspaceId: "w-1",
        returnTo: "/tools/mcp-servers/s-1",
        openWindow: () => popup as unknown as Window,
      }),
    {
      wrapper: ({ children }: { children: ReactNode }) => (
        <QueryClientProvider client={client}>{children}</QueryClientProvider>
      ),
    },
  )

  act(() => result.current.start("s-1", "connect"))
  expect(popup.close).not.toHaveBeenCalled()
  unmount()

  expect(popup.close).toHaveBeenCalledOnce()
})
