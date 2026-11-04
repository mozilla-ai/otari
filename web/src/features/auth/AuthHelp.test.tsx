import { render, screen } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import type { ReactElement } from "react"
import { expect, it } from "vitest"
import { AuthHelp } from "./AuthHelp"

async function open(ui: ReactElement) {
  render(ui)
  await userEvent.setup().click(screen.getByRole("button", { name: "Help" }))
  expect(await screen.findByRole("dialog", { name: "Help" })).toBeVisible()
}

it("offers password recovery beside the master-key box, where no password field carries it", async () => {
  await open(<AuthHelp offersRecovery credential="master-key" />)

  expect(
    screen.getByRole("link", { name: /Forgot your password/ }),
  ).toHaveAttribute("href", "#/recover-password")
})

// `Login` renders recovery next to the password field itself, so a second copy
// in here would be the same link twice on one screen.
it("leaves password recovery to the form while the password box is showing", async () => {
  await open(<AuthHelp offersRecovery credential="password" />)

  expect(
    screen.queryByRole("link", { name: /Forgot your password/ }),
  ).toBeNull()
  expect(
    screen.getByRole("link", { name: /verification link/ }),
  ).toBeInTheDocument()
})

// Hidden rather than offered and then refused: every recovery flow begins by
// sending a message, so a gateway that cannot send mail offers neither.
it("offers no recovery at all where the deployment cannot send mail", async () => {
  await open(<AuthHelp offersRecovery={false} credential="master-key" />)

  expect(
    screen.queryByRole("link", { name: /Forgot your password/ }),
  ).toBeNull()
  expect(screen.queryByRole("link", { name: /verification link/ })).toBeNull()
  expect(
    screen.getByRole("link", { name: /welcome guide/ }),
  ).toBeInTheDocument()
})

it("names the credential the form beside it actually took", async () => {
  await open(<AuthHelp offersRecovery credential="password" />)

  expect(
    screen.getByText(/Your password is sent once and exchanged/),
  ).toBeInTheDocument()
  expect(screen.queryByText(/master key/)).toBeNull()
})

it("names the master key instead where that is the box", async () => {
  await open(<AuthHelp offersRecovery credential="master-key" />)

  expect(screen.getByText(/master key/, { selector: "a" })).toBeInTheDocument()
  expect(screen.queryByText(/^Your password is sent once/)).toBeNull()
})

// `SignupPage` is setting a credential rather than taking one, so a note about
// what becomes of the one just typed would describe nothing on that page.
it("leaves the credential note off a page that takes no credential", async () => {
  await open(<AuthHelp offersRecovery />)

  expect(screen.queryByText(/is sent once and exchanged/)).toBeNull()
  expect(
    screen.queryByRole("link", { name: /Forgot your password/ }),
  ).toBeNull()
  expect(
    screen.getByRole("link", { name: /verification link/ }),
  ).toBeInTheDocument()
})
