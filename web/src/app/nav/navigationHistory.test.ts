import { afterEach, describe, expect, it } from "vitest"
import {
  lastLocation,
  lastRailContext,
  rememberLocation,
} from "./navigationHistory"
import type { NavItem } from "./types"

const WORKSPACE_KEY = "otari.dashboard.lastWorkspaceLocation"
const ORGANIZATION_KEY = "otari.dashboard.lastOrganizationLocation"

// What the shell's own predicate answers for a deployment that hosts every
// surface, which is the ordinary standalone gateway.
const showsEverything = () => true
// And what it answers for one that does not. The predicate is given the entry,
// so a surface is all a test needs to withhold to hide a destination.
const withoutSurface =
  (surface: string) =>
  (item: NavItem): boolean =>
    item.surface !== surface

describe("rail location memory", () => {
  afterEach(() => {
    window.localStorage.clear()
  })

  it("files a destination under the rail that declares it", () => {
    rememberLocation("/usage", showsEverything)
    rememberLocation("/budgets", showsEverything)

    // Two rails, two memories: recording an organization page must not overwrite
    // where you were on the workspace one, which is the whole point.
    expect(lastLocation("workspace", showsEverything)).toEqual({ to: "/usage" })
    expect(lastLocation("organization", showsEverything)).toEqual({
      to: "/budgets",
    })
  })

  it("remembers a nested destination, not just a top-level one", () => {
    // "Including its subpage" is the part the shell used to lose: it sent you to
    // the rail's landing page however deep you had been.
    rememberLocation("/tools/web-search", showsEverything)

    expect(lastLocation("workspace", showsEverything)).toEqual({
      to: "/tools/web-search",
    })
  })

  it("ignores a path the registry does not declare", () => {
    rememberLocation("/usage", showsEverything)
    rememberLocation("/docs", showsEverything)

    // The guide and the 404 splat are not places to return to, so the last real
    // destination stands rather than being replaced by one.
    expect(lastLocation("workspace", showsEverything)).toEqual({ to: "/usage" })
  })

  it("ignores a destination this deployment gates off", () => {
    rememberLocation("/budgets", showsEverything)
    // Registered, and reachable by URL, but the shell answers it with the "not
    // available here" panel. The page someone was just told they cannot have is
    // not the page to send them back to.
    rememberLocation("/settings", withoutSurface("settings"))

    expect(lastLocation("organization", showsEverything)).toEqual({
      to: "/budgets",
    })
  })

  it("ignores a nested destination whose own surface is gated off", () => {
    rememberLocation("/routing", showsEverything)
    // Guardrails is grouped under Routing and served by the tools surface, so
    // the child's surface is the one that has to be consulted, not its parent's.
    rememberLocation("/tools/guardrails", withoutSurface("tools"))

    expect(lastLocation("workspace", showsEverything)).toEqual({
      to: "/routing",
    })
  })

  it("drops a stored value that is no longer a destination", () => {
    // A stale entry from an older build, or a hand-edited one. Returning it would
    // send someone to the shell's "not available here" panel.
    window.localStorage.setItem(WORKSPACE_KEY, "/retired-page")

    expect(lastLocation("workspace", showsEverything)).toBeUndefined()
  })

  it("drops a stored value the deployment has since gated off", () => {
    // The half a write-time check cannot cover: the entry was visible when it was
    // stored, and a gateway restarted against a config reporting fewer surfaces
    // changed the answer underneath it.
    //
    // A destination that is still gated, which is the whole point of the case:
    // an unregistered path is dropped by the test above for a different reason,
    // so using one here would keep passing while testing nothing.
    window.localStorage.setItem(ORGANIZATION_KEY, "/organization/usage")

    expect(
      lastLocation("organization", withoutSurface("organization_usage")),
    ).toBeUndefined()
    expect(lastLocation("organization", showsEverything)).toEqual({
      to: "/organization/usage",
    })
  })

  it("remembers a deployment page under its own rail, and only there", () => {
    rememberLocation("/settings", showsEverything)

    expect(lastLocation("deployment", showsEverything)).toEqual({
      to: "/settings",
    })
    expect(lastLocation("workspace", showsEverything)).toBeUndefined()
    expect(lastLocation("organization", showsEverything)).toBeUndefined()
  })

  it("does not let a deployment page become the rail it returns to", () => {
    // The guard that matters, and the one an obvious implementation gets wrong:
    // written on every navigation, landing on a deployment page would record
    // "deployment" as the place to come back to, which is the answer the back
    // link is about to read. A state that destroys its own input.
    rememberLocation("/organization/members", showsEverything)
    rememberLocation("/settings", showsEverything)

    expect(lastRailContext(showsEverything)).toBe("organization")
  })

  it("returns to the workspace rail when nothing has been recorded", () => {
    // A bookmark straight to a deployment page in a fresh browser. Real, not a
    // theoretical branch.
    expect(lastRailContext(showsEverything)).toBe("workspace")
  })

  it("returns to the workspace rail when the remembered one has been gated off", () => {
    // Asked of `lastLocation` rather than re-checked here, so the two answers
    // cannot disagree. Back into a rail someone can no longer use is worse than
    // back somewhere merely unexpected.
    rememberLocation("/organization/members", showsEverything)

    expect(lastRailContext(showsEverything)).toBe("organization")
    expect(lastRailContext(withoutSurface("organizations"))).toBe("workspace")
  })

  it("ignores a remembered context that is not one", () => {
    // A key left by an older build, or edited by hand, would otherwise arrive as
    // a NavContext the compiler believes.
    window.localStorage.setItem("otari.dashboard.lastRailContext", "billing")

    expect(lastRailContext(showsEverything)).toBe("workspace")
  })

  it("drops a stored value that belongs to the other rail", () => {
    // Both keys are read through the same registry check, so a destination that
    // has since moved rails cannot land you on the wrong one.
    window.localStorage.setItem(ORGANIZATION_KEY, "/usage")

    expect(lastLocation("organization", showsEverything)).toBeUndefined()
  })

  it("answers with nothing before anything has been recorded", () => {
    expect(lastLocation("workspace", showsEverything)).toBeUndefined()
    expect(lastLocation("organization", showsEverything)).toBeUndefined()
  })
})
