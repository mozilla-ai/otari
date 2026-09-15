import { render, screen } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { expect, it } from "vitest"
import { ThinkingBlock } from "./ThinkingBlock"

it("lets the keyboard reveal and collapse provider reasoning", async () => {
  const user = userEvent.setup()
  render(<ThinkingBlock content="Reasoning returned by the provider." />)
  const trigger = screen.getByRole("button", { name: "Reasoning" })
  expect(trigger).toHaveAttribute("aria-expanded", "false")
  await user.tab()
  await user.keyboard("{Enter}")
  expect(trigger).toHaveAttribute("aria-expanded", "true")
  expect(
    await screen.findByText("Reasoning returned by the provider."),
  ).toBeVisible()
  await user.keyboard("{Enter}")
  expect(trigger).toHaveAttribute("aria-expanded", "false")
})
