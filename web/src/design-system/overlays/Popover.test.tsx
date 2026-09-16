import { render, screen, waitFor } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { expect, it } from "vitest"
import { Button } from "../actions/Button"
import { Popover } from "./Popover"

it("has one trigger and returns keyboard focus after Escape", async () => {
  const user = userEvent.setup()
  render(
    <Popover label="Models" trigger={<Button>Choose model</Button>}>
      <Button>Model option</Button>
    </Popover>,
  )
  expect(screen.getAllByRole("button", { name: "Choose model" })).toHaveLength(
    1,
  )
  await user.tab()
  const trigger = screen.getByRole("button", { name: "Choose model" })
  expect(trigger).toHaveFocus()
  await user.keyboard("{Enter}")
  expect(await screen.findByRole("dialog")).toBeInTheDocument()
  await user.keyboard("{Escape}")
  await waitFor(() => expect(trigger).toHaveFocus())
})
