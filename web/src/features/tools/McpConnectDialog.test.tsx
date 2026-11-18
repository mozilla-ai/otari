import { render, screen } from "@testing-library/react"
import { expect, it, vi } from "vitest"

import { McpConnectDialog } from "@/features/tools/McpConnectDialog"
import { workspaceMcpServer } from "@/tests/fixtures"

it("holds the other tiles while one server's sign-in is running", () => {
  const drive = workspaceMcpServer({
    id: "66666666-6666-6666-6666-666666666666",
    name: "drive",
    credential_mode: "per_user",
  })
  const notion = workspaceMcpServer({
    id: "77777777-7777-7777-7777-777777777777",
    name: "notion",
    credential_mode: "per_user",
  })
  render(
    <McpConnectDialog
      isOpen
      onOpenChange={vi.fn()}
      servers={[drive, notion]}
      logos={new Map()}
      pendingId={drive.id}
      onPick={vi.fn()}
    />,
  )

  expect(screen.getByRole("button", { name: "Connect notion" })).toBeDisabled()
})
