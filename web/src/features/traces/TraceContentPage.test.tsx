import { render, screen } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { describe, expect, it, vi } from "vitest"
import { TraceContentControls } from "@/features/traces/TraceContentPage"
import { ApiError } from "@/shared/api/client"

const SETTINGS = {
  content_capture: "off",
  effective: "off",
  ceiling: "tool_io",
  admin_content_access: false,
} as const

function renderControls(
  overrides: Partial<Parameters<typeof TraceContentControls>[0]> = {},
) {
  const props = {
    workspace: "Research",
    settings: SETTINGS,
    onChange: vi.fn(),
    onAdminAccessChange: vi.fn(),
    onPurge: vi.fn(),
    isSaving: false,
    isSavingAdminAccess: false,
    isPurging: false,
    saveError: null,
    adminAccessError: null,
    purgeError: null,
    ...overrides,
  }
  render(<TraceContentControls {...props} />)
  return props
}

describe("TraceContentControls", () => {
  it("offers only the levels the deployment permits", () => {
    renderControls()

    expect(
      screen.getByRole("radio", { name: "Tool calls" }),
    ).toBeInTheDocument()
    expect(
      screen.queryByRole("radio", { name: "Everything" }),
    ).not.toBeInTheDocument()
  })

  it("changes the level only after the admin confirms", async () => {
    const { onChange } = renderControls()

    await userEvent.click(screen.getByRole("radio", { name: "Tool calls" }))
    expect(onChange).not.toHaveBeenCalled()
    expect(
      await screen.findByRole("heading", {
        name: "Capture tool calls in Research?",
      }),
    ).toBeInTheDocument()
    await userEvent.click(
      screen.getByRole("button", { name: "Capture tool calls" }),
    )

    expect(onChange).toHaveBeenCalledWith("tool_io", expect.any(Function))
  })

  it("leaves the level alone when the admin cancels", async () => {
    const { onChange } = renderControls()

    await userEvent.click(screen.getByRole("radio", { name: "Tool calls" }))
    await userEvent.click(await screen.findByRole("button", { name: "Cancel" }))

    expect(onChange).not.toHaveBeenCalled()
  })

  it("purges only after the admin confirms", async () => {
    const { onPurge } = renderControls()

    await userEvent.click(
      screen.getByRole("button", { name: "Purge stored content" }),
    )
    expect(onPurge).not.toHaveBeenCalled()
    await userEvent.click(
      await screen.findByRole("button", { name: "Purge content" }),
    )

    expect(onPurge).toHaveBeenCalledOnce()
  })

  it("says turning capture off takes effect within 30 seconds", async () => {
    renderControls({
      settings: {
        content_capture: "tool_io",
        effective: "tool_io",
        ceiling: "full",
        admin_content_access: false,
      },
    })

    await userEvent.click(screen.getByRole("radio", { name: "Off" }))

    expect(
      await screen.findByText(/Takes effect within 30 seconds/),
    ).toBeInTheDocument()
  })

  it("names who may read the content and what full capture keeps", async () => {
    renderControls({ settings: { ...SETTINGS, ceiling: "full" } })

    await userEvent.click(screen.getByRole("radio", { name: "Everything" }))

    const dialog = await screen.findByRole("alertdialog")
    expect(dialog).toHaveTextContent(
      /the model's output for it, and every tool call's arguments and result/,
    )
    expect(dialog).toHaveTextContent(
      /readable by the person who ran the session/,
    )
    expect(dialog).toHaveTextContent(
      /by organization admins only if you allow it below/,
    )
    expect(dialog).toHaveTextContent(
      /platform operators only through a recorded break-glass with a stated reason/,
    )
  })

  it("is honest that a purge leaves database backups until they expire", async () => {
    renderControls()

    await userEvent.click(
      screen.getByRole("button", { name: "Purge stored content" }),
    )

    const dialog = await screen.findByRole("alertdialog")
    expect(dialog).toHaveTextContent(/deleted from the database/)
    expect(dialog).toHaveTextContent(/each session's key is destroyed/)
    expect(dialog).toHaveTextContent(
      /Copies in database backups remain until those backups expire/,
    )
    expect(dialog).not.toHaveTextContent(/can be read again/)
  })

  it("shows the server's refusal when capture cannot be turned on", async () => {
    renderControls({
      saveError: new ApiError(
        409,
        "Content encryption is not configured on this deployment",
      ),
    })

    await userEvent.click(screen.getByRole("radio", { name: "Tool calls" }))

    expect(await screen.findByRole("alert")).toHaveTextContent(
      "Content encryption is not configured on this deployment",
    )
  })

  it("explains a deployment that permits no capture", () => {
    renderControls({ settings: { ...SETTINGS, ceiling: "off" } })

    expect(
      screen.getByText(/This deployment does not permit content capture/),
    ).toBeInTheDocument()
  })

  it("lets organization admins read content only after the admin confirms", async () => {
    const { onAdminAccessChange } = renderControls()

    await userEvent.click(
      screen.getByRole("switch", {
        name: "Let organization admins read content in Research",
      }),
    )
    expect(onAdminAccessChange).not.toHaveBeenCalled()
    const dialog = await screen.findByRole("alertdialog")
    expect(dialog).toHaveTextContent(/Every read they make is recorded/)
    await userEvent.click(
      screen.getByRole("button", { name: "Let admins read" }),
    )

    expect(onAdminAccessChange).toHaveBeenCalledWith(true, expect.any(Function))
  })

  it("confirms turning admin reads off too", async () => {
    const { onAdminAccessChange } = renderControls({
      settings: { ...SETTINGS, admin_content_access: true },
    })

    await userEvent.click(
      screen.getByRole("switch", {
        name: "Let organization admins read content in Research",
      }),
    )
    expect(
      await screen.findByRole("heading", {
        name: "Stop organization admins reading content in Research?",
      }),
    ).toBeInTheDocument()
    expect(onAdminAccessChange).not.toHaveBeenCalled()
    await userEvent.click(
      screen.getByRole("button", { name: "Stop admin reads" }),
    )

    expect(onAdminAccessChange).toHaveBeenCalledWith(
      false,
      expect.any(Function),
    )
  })

  it("leaves admin reads alone when the admin cancels", async () => {
    const { onAdminAccessChange } = renderControls()

    await userEvent.click(
      screen.getByRole("switch", {
        name: "Let organization admins read content in Research",
      }),
    )
    await userEvent.click(await screen.findByRole("button", { name: "Cancel" }))

    expect(onAdminAccessChange).not.toHaveBeenCalled()
  })
})
