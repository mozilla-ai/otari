import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { render, screen } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import type { ReactNode } from "react"
import { expect, it } from "vitest"

import type { WorkspaceMcpServerLogo } from "@/client"
import { McpServerLogoButton } from "@/features/tools/McpServerLogoDialog"

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
