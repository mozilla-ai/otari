import { useLocation } from "@tanstack/react-router"
import { act, renderHook } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import { flushRouter, withRouter } from "@/tests/router"

import {
  matchesSearch,
  SEARCH_MAX_LENGTH,
  useListSearch,
} from "./useListSearch"

async function searchAt(
  url: string,
  options?: Parameters<typeof useListSearch>[0],
) {
  const { result } = renderHook(
    () => ({
      search: useListSearch(options),
      searchStr: useLocation({ select: (location) => location.searchStr }),
    }),
    { wrapper: withRouter({ url }) },
  )
  await flushRouter()
  return result
}

describe("useListSearch", () => {
  it("reads the term from the URL, so a shared link opens narrowed", async () => {
    const result = await searchAt("/?q=%20gpt%20")
    expect(result.current.search.value).toBe(" gpt ")
    expect(result.current.search.term).toBe("gpt")
  })

  it("writes the term to the URL and goes back to the first page", async () => {
    const result = await searchAt("/?page=3&size=50")
    await act(async () => {
      result.current.search.onChange("claude")
    })
    const params = new URLSearchParams(result.current.searchStr)
    expect(params.get("q")).toBe("claude")
    expect(params.has("page")).toBe(false)
    expect(params.get("size")).toBe("50")
  })

  it("keeps a second list's search and pager apart from the page's own", async () => {
    const result = await searchAt("/?q=keys&page=2&models_page=4", {
      param: "models_q",
      pageParam: "models_page",
    })
    await act(async () => {
      result.current.search.onChange("gpt")
    })
    const params = new URLSearchParams(result.current.searchStr)
    expect(params.get("models_q")).toBe("gpt")
    expect(params.has("models_page")).toBe(false)
    expect(params.get("q")).toBe("keys")
    expect(params.get("page")).toBe("2")
  })

  it("drops the param once the field is cleared", async () => {
    const result = await searchAt("/?q=claude")
    await act(async () => {
      result.current.search.onChange("")
    })
    expect(result.current.searchStr).toBe("")
  })

  it("caps the term at what the gateway accepts", async () => {
    const result = await searchAt(`/?q=${"x".repeat(SEARCH_MAX_LENGTH + 5)}`)
    expect(result.current.search.term).toHaveLength(SEARCH_MAX_LENGTH)
  })

  describe("while typing", () => {
    beforeEach(() => {
      vi.useFakeTimers({ shouldAdvanceTime: true })
    })
    afterEach(() => {
      vi.useRealTimers()
    })

    it("settles the term after a pause rather than on every keystroke", async () => {
      const result = await searchAt("/")
      await act(async () => {
        result.current.search.onChange("g")
      })
      await act(async () => {
        result.current.search.onChange("gp")
      })
      expect(result.current.search.value).toBe("gp")
      expect(result.current.search.term).toBe("")
      await act(async () => {
        vi.advanceTimersByTime(250)
      })
      expect(result.current.search.term).toBe("gp")
    })
  })
})

describe("matchesSearch", () => {
  it("matches a substring of any field, ignoring case", () => {
    expect(matchesSearch("PROD", "staging", "my-prod-key")).toBe(true)
    expect(matchesSearch("prod", "staging", "dev")).toBe(false)
  })

  it("matches everything when there is no term", () => {
    expect(matchesSearch("", "anything")).toBe(true)
  })

  it("skips a field that is absent", () => {
    expect(matchesSearch("ada", undefined, "Ada Lovelace")).toBe(true)
    expect(matchesSearch("ada", undefined)).toBe(false)
    expect(matchesSearch("ada", null, "Ada Lovelace")).toBe(true)
  })

  it("takes the LIKE wildcards literally, as the gateway does", () => {
    expect(matchesSearch("o3_", "o3-mini")).toBe(false)
    expect(matchesSearch("100%", "100% off")).toBe(true)
  })
})
