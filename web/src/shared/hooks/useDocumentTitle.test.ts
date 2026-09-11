import { renderHook } from "@testing-library/react"
import { afterEach, expect, it } from "vitest"
import { useDocumentTitle } from "./useDocumentTitle"

const originalTitle = document.title
afterEach(() => {
  document.title = originalTitle
})

it("updates the page title and restores the brand on unmount", () => {
  const { rerender, unmount } = renderHook(
    ({ page }: { page?: string }) => useDocumentTitle(page),
    { initialProps: { page: "API Keys" } },
  )
  expect(document.title).toBe("API Keys · Otari")
  rerender({ page: "Usage" })
  expect(document.title).toBe("Usage · Otari")
  unmount()
  expect(document.title).toBe("Otari")
})

it("uses the brand when no page is named", () => {
  renderHook(() => useDocumentTitle())
  expect(document.title).toBe("Otari")
})
