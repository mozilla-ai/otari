import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { act, render, screen } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import type { ReactNode } from "react"
import { expect, it, vi } from "vitest"

import type { WorkspaceMcpServerLogo } from "@/client"
import { McpServerLogoButton } from "@/features/tools/McpServerLogoDialog"
import { type PreparedLogo, prepareLogo } from "@/features/tools/mcpLogoImage"

vi.mock("@/features/tools/mcpLogoImage", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/features/tools/mcpLogoImage")>()),
  prepareLogo: vi.fn(),
}))

const LOGO: WorkspaceMcpServerLogo = {
  server_id: "s-1",
  logo_url: "/logos/s-1.png",
  logo_tone: "tile",
  logo_tile: "always",
  has_logo_override: true,
}

function withQueries(children: ReactNode) {
  return (
    <QueryClientProvider client={new QueryClient()}>
      {children}
    </QueryClientProvider>
  )
}

it("opens on the saved tile setting when the logo loaded after the page", async () => {
  const user = userEvent.setup()
  const button = (logo: WorkspaceMcpServerLogo | undefined) =>
    withQueries(
      <McpServerLogoButton
        workspaceId="w-1"
        serverId="s-1"
        name="github"
        logo={logo}
      />,
    )
  const { rerender } = render(button(undefined))
  rerender(button(LOGO))

  await user.click(
    screen.getByRole("button", { name: "Change the logo of github" }),
  )

  expect(await screen.findByRole("radio", { name: "Always" })).toBeChecked()
})

it("keeps Save off while a newly picked image is still being prepared", async () => {
  const user = userEvent.setup()
  const prepared = (name: string): PreparedLogo => ({
    png: new Blob([name], { type: "image/png" }),
    previewUrl: `blob:${name}`,
    luminance: 0.5,
  })
  let finishSecond: (logo: PreparedLogo) => void = () => {}
  vi.mocked(prepareLogo)
    .mockResolvedValueOnce(prepared("first"))
    .mockReturnValueOnce(
      new Promise((resolve) => {
        finishSecond = resolve
      }),
    )
  render(
    withQueries(
      <McpServerLogoButton
        workspaceId="w-1"
        serverId="s-1"
        name="github"
        logo={LOGO}
      />,
    ),
  )
  await user.click(
    screen.getByRole("button", { name: "Change the logo of github" }),
  )
  const input = document.querySelector<HTMLInputElement>('input[type="file"]')
  if (!input) throw new Error("no file input")
  const save = await screen.findByRole("button", { name: "Save" })

  await user.upload(input, new File(["a"], "a.png", { type: "image/png" }))
  await vi.waitFor(() => expect(save).toBeEnabled())
  await user.upload(input, new File(["b"], "b.png", { type: "image/png" }))
  expect(save).toBeDisabled()

  await act(async () => finishSecond(prepared("second")))
  expect(save).toBeEnabled()
})

function deferred() {
  let resolve: (logo: PreparedLogo) => void = () => {}
  const promise = new Promise<PreparedLogo>((done) => {
    resolve = done
  })
  return { promise, resolve }
}

const preparedAs = (name: string): PreparedLogo => ({
  png: new Blob([name], { type: "image/png" }),
  previewUrl: `blob:${name}`,
  luminance: 0.5,
})

async function openDialog() {
  const user = userEvent.setup()
  render(
    withQueries(
      <McpServerLogoButton
        workspaceId="w-1"
        serverId="s-1"
        name="github"
        logo={LOGO}
      />,
    ),
  )
  await user.click(
    screen.getByRole("button", { name: "Change the logo of github" }),
  )
  const input = document.querySelector<HTMLInputElement>('input[type="file"]')
  if (!input) throw new Error("no file input")
  return { user, input }
}

const previews = (url: string) =>
  document.querySelectorAll(`img[src="${url}"]`).length

it("keeps the latest image when an earlier one finishes preparing after it", async () => {
  const revoke = vi.spyOn(URL, "revokeObjectURL")
  const first = deferred()
  const second = deferred()
  vi.mocked(prepareLogo)
    .mockReturnValueOnce(first.promise)
    .mockReturnValueOnce(second.promise)
  const { user, input } = await openDialog()
  const save = screen.getByRole("button", { name: "Save" })

  await user.upload(input, new File(["a"], "a.png", { type: "image/png" }))
  await user.upload(input, new File(["b"], "b.png", { type: "image/png" }))
  await act(async () => second.resolve(preparedAs("second")))
  await act(async () => first.resolve(preparedAs("first")))

  expect(previews("blob:second")).toBe(2)
  expect(previews("blob:first")).toBe(0)
  expect(revoke).toHaveBeenCalledWith("blob:first")
  expect(save).toBeEnabled()
  revoke.mockRestore()
})

it("keeps Save off until the latest image is ready, whichever finishes first", async () => {
  const first = deferred()
  const second = deferred()
  vi.mocked(prepareLogo)
    .mockReturnValueOnce(first.promise)
    .mockReturnValueOnce(second.promise)
  const { user, input } = await openDialog()
  const save = screen.getByRole("button", { name: "Save" })

  await user.upload(input, new File(["a"], "a.png", { type: "image/png" }))
  await user.upload(input, new File(["b"], "b.png", { type: "image/png" }))
  await act(async () => first.resolve(preparedAs("first")))

  expect(save).toBeDisabled()
  expect(previews("blob:first")).toBe(0)
})

it("frees an image that finishes preparing after the dialog closed", async () => {
  const revoke = vi.spyOn(URL, "revokeObjectURL")
  const pending = deferred()
  vi.mocked(prepareLogo).mockReturnValueOnce(pending.promise)
  const { user, input } = await openDialog()

  await user.upload(input, new File(["a"], "a.png", { type: "image/png" }))
  await user.click(screen.getByRole("button", { name: "Cancel" }))
  expect(screen.queryByRole("button", { name: "Save" })).toBeNull()
  await act(async () => pending.resolve(preparedAs("late")))

  expect(revoke).toHaveBeenCalledWith("blob:late")
  revoke.mockRestore()
})
