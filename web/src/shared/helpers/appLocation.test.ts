import { afterEach, describe, expect, it, vi } from "vitest"

import {
  currentRoute,
  followInAppLink,
  inAppRoute,
  navigateTo,
  routeParam,
  routePath,
  routeSearchParams,
  routeWithoutFragment,
} from "@/shared/helpers/appLocation"

describe("currentRoute", () => {
  it("is the path, query and fragment of a location at the root base", () => {
    expect(
      currentRoute({ pathname: "/verify-email", search: "", hash: "#token=t" }),
    ).toBe("/verify-email#token=t")
    expect(
      currentRoute({ pathname: "/models", search: "?provider=x", hash: "" }),
    ).toBe("/models?provider=x")
  })
})

describe("routePath", () => {
  it("drops the query and the fragment", () => {
    expect(routePath("/verify-email#token=t")).toBe("/verify-email")
    expect(routePath("/check-email?type=signup")).toBe("/check-email")
    expect(routePath("")).toBe("/")
  })
})

describe("routeParam", () => {
  it("reads the fragment ahead of the query", () => {
    expect(routeParam("/x?token=q#token=f", "token")).toBe("f")
    expect(routeParam("/x?token=q", "token")).toBe("q")
    expect(routeParam("/x#token=", "token")).toBeNull()
  })
})

describe("routeSearchParams", () => {
  it("reads the query alone, never the fragment", () => {
    expect(routeSearchParams("/x?a=1#b=2").get("a")).toBe("1")
    expect(routeSearchParams("/x?a=1#b=2").get("b")).toBeNull()
  })
})

describe("inAppRoute", () => {
  it("answers a root-relative href and refuses one that leaves the page", () => {
    expect(inAppRoute("/models")).toBe("/models")
    expect(inAppRoute("https://otari.example.com/models")).toBeNull()
    expect(inAppRoute("//elsewhere.example.com")).toBeNull()
  })
})

describe("navigateTo and followInAppLink", () => {
  afterEach(() => {
    window.history.replaceState(null, "", "/")
  })

  it("moves the address without a page load", () => {
    navigateTo("/signup")
    expect(window.location.pathname).toBe("/signup")
    navigateTo("/recover-password", { replace: true })
    expect(window.location.pathname).toBe("/recover-password")
  })

  function click(href: string, init: Partial<MouseEvent> = {}) {
    const preventDefault = vi.fn()
    followInAppLink({
      button: 0,
      metaKey: false,
      ctrlKey: false,
      shiftKey: false,
      altKey: false,
      defaultPrevented: false,
      ...init,
      preventDefault,
      currentTarget: {
        getAttribute: (name: string) => (name === "href" ? href : null),
      },
    })
    return preventDefault
  }

  it("follows a plain click on an in-app link in place", () => {
    expect(click("/providers")).toHaveBeenCalled()
    expect(window.location.pathname).toBe("/providers")
  })

  it("leaves a modified click and an outside link to the browser", () => {
    expect(click("/providers", { metaKey: true })).not.toHaveBeenCalled()
    expect(click("https://example.com/")).not.toHaveBeenCalled()
    expect(window.location.pathname).toBe("/")
  })
})

describe("routeWithoutFragment", () => {
  it("keeps the path and query and drops the fragment", () => {
    expect(routeWithoutFragment("/usage?range=7d#token=t")).toBe(
      "/usage?range=7d",
    )
    expect(routeWithoutFragment("#key=k")).toBe("/")
  })
})
