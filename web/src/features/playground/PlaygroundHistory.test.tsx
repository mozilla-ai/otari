import { render, screen, waitFor, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { useState } from "react"
import { expect, it, vi } from "vitest"
import type { PlaygroundComparison, PlaygroundConversation } from "@/client"
import { PlaygroundHistory } from "./PlaygroundHistory"

const conversations: PlaygroundConversation[] = Array.from(
  { length: 6 },
  (_, i) => ({
    id: `chat-${i}`,
    title: `Saved prompt ${i}`,
    model: "openai:gpt-4o",
    workspace_id: "workspace",
    message_count: 2,
    created_at: `2026-09-${String(16 - i).padStart(2, "0")}T10:00:00Z`,
  }),
)
const comparisons: PlaygroundComparison[] = [
  {
    id: "comparison-1",
    workspace_id: "workspace",
    user_question: "Compare retry strategies",
    model_a: "openai:gpt-4o",
    model_b: "anthropic:claude-sonnet-4",
    preference: "tie",
    created_at: "2026-09-16T11:00:00Z",
  },
]

function renderHistory(
  onDeleteConversation = vi.fn().mockResolvedValue(undefined),
) {
  const onLoad = vi.fn().mockResolvedValue(undefined)
  function History() {
    const [isOpen, setIsOpen] = useState(false)
    return (
      <PlaygroundHistory
        conversations={conversations}
        comparisons={comparisons}
        isOpen={isOpen}
        onOpenChange={setIsOpen}
        onLoad={onLoad}
        onDeleteConversation={onDeleteConversation}
        onDeleteComparison={vi.fn().mockResolvedValue(undefined)}
        isSaveDisabled={false}
        isSavePending={false}
        isLoading={false}
      />
    )
  }
  render(<History />)
  return { onLoad }
}

it("limits recent entries, then searches and filters the full history", async () => {
  const user = userEvent.setup()
  renderHistory()
  await user.click(screen.getByRole("button", { name: "History" }))
  expect(screen.queryByText("Saved prompt 5")).not.toBeInTheDocument()
  await user.click(screen.getByRole("button", { name: "View all" }))
  const dialog = await screen.findByRole("dialog", { name: "History" })
  expect(within(dialog).getByText("Saved prompt 5")).toBeInTheDocument()
  await user.click(within(dialog).getByRole("radio", { name: "Comparisons" }))
  expect(within(dialog).queryByText("Saved prompt 0")).not.toBeInTheDocument()
  expect(
    within(dialog).getByText("gpt-4o vs claude-sonnet-4"),
  ).toBeInTheDocument()
  await user.type(within(dialog).getByRole("searchbox"), "missing")
  expect(within(dialog).getByText("No matching history.")).toBeInTheDocument()
  await user.clear(within(dialog).getByRole("searchbox"))
  await user.type(within(dialog).getByRole("searchbox"), "retry")
  expect(
    within(dialog).getByText("gpt-4o vs claude-sonnet-4"),
  ).toBeInTheDocument()
})

it("loads an older chat from View all and closes the dialog", async () => {
  const user = userEvent.setup()
  const { onLoad } = renderHistory()
  await user.click(screen.getByRole("button", { name: "History" }))
  await user.click(screen.getByRole("button", { name: "View all" }))
  await user.click(screen.getByRole("button", { name: /^Saved prompt 5/ }))
  expect(onLoad).toHaveBeenCalledWith("chat-5")
  await waitFor(() =>
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument(),
  )
})

it("requires confirmation and preserves a failed deletion for retry", async () => {
  const user = userEvent.setup()
  const remove = vi.fn().mockRejectedValue(new Error("Could not delete chat"))
  renderHistory(remove)
  await user.click(screen.getByRole("button", { name: "History" }))
  await user.click(
    screen.getByRole("button", { name: 'Delete chat "Saved prompt 0"' }),
  )
  expect(remove).not.toHaveBeenCalled()
  await user.click(screen.getByRole("button", { name: "Delete permanently" }))
  expect(await screen.findByText("Could not delete chat")).toBeInTheDocument()
  expect(remove).toHaveBeenCalledWith("chat-0")
  expect(
    screen.getByRole("button", { name: "Delete permanently" }),
  ).toBeEnabled()
})
