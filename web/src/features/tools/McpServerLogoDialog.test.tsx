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
