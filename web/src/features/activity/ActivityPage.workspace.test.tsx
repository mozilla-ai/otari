import { useRouterState } from "@tanstack/react-router"
import { act, screen, within } from "@testing-library/react"
import { useSyncExternalStore } from "react"
import { afterEach, describe, expect, it, vi } from "vitest"

import type { CallerWorkspaceMembership } from "@/client"
import { ActivityPage } from "@/features/activity/ActivityPage"
import { entry, mockApi, renderPage } from "@/tests/activity"

// The workspace switcher as a store the test moves by hand, so it can trail the
// organization context the page reads, as it can in a browser on a cold load.
const switcher = vi.hoisted(() => {
  let value = {
    memberships: [] as readonly CallerWorkspaceMembership[],
    selected: null as CallerWorkspaceMembership | null,
    select: () => {},
    isLoading: true,
    isSettled: false,
  }
  const listeners = new Set<() => void>()
  return {
    get: () => value,
    set: (next: Partial<typeof value>) => {
      value = { ...value, ...next }
      for (const listener of listeners) listener()
    },
    subscribe: (listener: () => void) => {
      listeners.add(listener)
      return () => listeners.delete(listener)
    },
  }
})

vi.mock("@/shared/hooks/SelectedWorkspace", async (importOriginal) => ({
  ...(await importOriginal<object>()),
  useSelectedWorkspace: () =>
    useSyncExternalStore(switcher.subscribe, switcher.get),
}))

afterEach(() => {
  vi.restoreAllMocks()
})

const location: { search: Record<string, unknown> } = { search: {} }
function LocationProbe() {
  location.search = useRouterState({ select: (state) => state.location.search })
  return null
}

describe("ActivityPage while the workspace switcher settles", () => {
  it("keeps a linked request open once the switcher catches up", async () => {
    mockApi({ rows: [entry({ id: "r1", model: "gpt-4o" })] })
    renderPage(
      <>
        <ActivityPage />
        <LocationProbe />
      </>,
      "/activity?request=r1",
    )
    // The organization context has answered, and the switcher has not yet.
    await screen.findByRole("status")
    expect(
      screen.queryByRole("complementary", { name: "Request detail" }),
    ).toBeNull()

    const workspace = {
      workspace_id: "ws-1",
      name: "Default workspace",
      role: "admin",
    } as CallerWorkspaceMembership
    act(() => {
      switcher.set({
        memberships: [workspace],
        selected: workspace,
        isLoading: false,
        isSettled: true,
      })
    })

    const panel = await screen.findByRole("complementary", {
      name: "Request detail",
    })
    expect(within(panel).getByRole("heading", { level: 2 })).toHaveTextContent(
      "gpt-4o",
    )
    expect(location.search.request).toBe("r1")
  })
})
