import { describe, expect, it } from "vitest"

import type { McpServerWorkspaceSettings } from "@/client"
import {
  asWorkspaceUses,
  availabilityOverride,
  hasWorkspaceSettings,
  organizationLimitLabel,
  organizationOffReason,
  overrideBody,
  workspaceAvailability,
} from "@/features/tools/mcpWorkspaceView"
import { workspaceMcpServer } from "@/tests/fixtures"

function organizationServer(own: Partial<McpServerWorkspaceSettings> = {}) {
  return workspaceMcpServer({
    scope: "organization",
    workspace_id: null,
    everyone_tools: ["*"],
    attach_by_default: false,
    in_workspace: {
      reached: true,
      enabled: true,
      everyone_tools: null,
      attach_by_default: null,
      organization_tools: null,
      ...own,
    },
  })
}

describe("asWorkspaceUses", () => {
  it("leaves a workspace's own server as it is", () => {
    const server = workspaceMcpServer()
    expect(asWorkspaceUses(server)).toBe(server)
  })

  it("takes the organization's settings where the workspace set none", () => {
    const view = asWorkspaceUses(organizationServer())
    expect(view.enabled).toBe(true)
    expect(view.attach_by_default).toBe(false)
    expect(view.everyone_tools).toEqual(["*"])
  })

  it("takes the workspace's own settings, and off where it is not reached", () => {
    const view = asWorkspaceUses(
      organizationServer({
        reached: false,
        everyone_tools: ["@read"],
        attach_by_default: true,
      }),
    )
    expect(view.enabled).toBe(false)
    expect(view.attach_by_default).toBe(true)
    expect(view.everyone_tools).toEqual(["@read"])
  })
})

describe("workspace settings", () => {
  it("reads the workspace's own choice apart from what the organization lets through", () => {
    const server = organizationServer({ reached: false, enabled: true })
    expect(workspaceAvailability(server)).toBe("on_request")
    expect(organizationOffReason(server)).toBe(
      "Your organization turned this server off for this workspace.",
    )
    expect(
      organizationOffReason(
        organizationServer({ reached: false, enabled: false }),
      ),
    ).toBe("")
  })

  it("sends every field, keeping what the change leaves alone", () => {
    const server = organizationServer({ everyone_tools: [] })
    expect(overrideBody(server, { attach_by_default: true })).toEqual({
      enabled: true,
      everyone_tools: [],
      attach_by_default: true,
    })
    expect(availabilityOverride(server, "off")).toEqual({
      enabled: false,
      everyone_tools: [],
      attach_by_default: null,
    })
    expect(availabilityOverride(server, "every_request")).toEqual({
      enabled: true,
      everyone_tools: [],
      attach_by_default: true,
    })
  })

  it("knows whether there is anything to clear", () => {
    expect(hasWorkspaceSettings(organizationServer())).toBe(false)
    expect(hasWorkspaceSettings(organizationServer({ enabled: false }))).toBe(
      true,
    )
    expect(
      hasWorkspaceSettings(organizationServer({ everyone_tools: ["x"] })),
    ).toBe(true)
  })

  it("says the organization's limit in words", () => {
    expect(organizationLimitLabel(null)).toBe("")
    expect(organizationLimitLabel([])).toBe("No tools")
    expect(organizationLimitLabel(["*"])).toBe("Any tool")
    expect(organizationLimitLabel(["@read"])).toBe("Read tools")
    expect(organizationLimitLabel(["get_issue"])).toBe("get_issue")
  })
})
