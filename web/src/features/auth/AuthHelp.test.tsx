import { render, screen } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import type { ReactElement } from "react"
import { expect, it } from "vitest"
import type { DeploymentType } from "@/client"
import { DeploymentProvider } from "@/shared/hooks/useDeployment"
import { bootstrap } from "@/tests/fixtures"
import { AuthHelp } from "./AuthHelp"

// The popover reads the bootstrap to decide whether this deployment serves the
// welcome guide, so every render here goes through a provider. Standalone by
// default, matching the fixture and the deployment most of these cases describe.
async function open(
  ui: ReactElement,
  deployment_type: DeploymentType = "standalone",
) {
  render(
    <DeploymentProvider value={bootstrap({ deployment_type })}>
      {ui}
    </DeploymentProvider>,
  )
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

// otari.ai serves the bundle from its own edge and `/welcome` from nowhere, so
// the row would be a link out of the app to a 404. See `welcomeGuideHref`.
it("offers no welcome guide on a hosted deployment, which serves none", async () => {
  await open(<AuthHelp offersRecovery credential="master-key" />, "hosted")

  expect(screen.queryByRole("link", { name: /welcome guide/ })).toBeNull()
  // The note still has to name the credential the form took; only its link goes.
  expect(screen.getByText(/master key/)).toBeInTheDocument()
  expect(screen.queryByText(/master key/, { selector: "a" })).toBeNull()
})

// Recovery is a different axis, and hosted deployments run those flows.
it("keeps the recovery links a hosted deployment does serve", async () => {
  await open(<AuthHelp offersRecovery credential="master-key" />, "hosted")

  expect(
    screen.getByRole("link", { name: /Forgot your password/ }),
  ).toBeInTheDocument()
  expect(
    screen.getByRole("link", { name: /verification link/ }),
  ).toBeInTheDocument()
})
