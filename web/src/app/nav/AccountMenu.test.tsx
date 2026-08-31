import { screen } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, describe, expect, it, vi } from "vitest"

import { AccountMenu } from "@/app/nav/AccountMenu"
import type { DeploymentBootstrap, OrganizationContext } from "@/client"
import { useOrganizationContext } from "@/shared/api/hooks"
import { DeploymentProvider } from "@/shared/hooks/useDeployment"
import {
  bootstrap,
  HOSTED_SURFACES,
  organizationContext,
} from "@/tests/fixtures"
import { AppProviders } from "@/tests/providers"
import { renderWithRouter } from "@/tests/router"

// The trigger names the person, and the person is a field on the membership
// context, so it is stubbed at fetch like the sidebar's own read of it. Every
// case installs one, including the ones about the menu's rows: the component
// makes that request either way, and an unstubbed `fetch` would leave it
// failing in the background of a test that is not about it.
function mockCaller(caller: OrganizationContext["caller"]) {
  vi.spyOn(globalThis, "fetch").mockImplementation(async () =>
    Response.json(organizationContext({ caller })),
  )
}

// The identity a standalone first boot leaves behind, which the fixture already
// describes: a name and no address.
const OPERATOR = organizationContext().caller

// The other thing that read can do. The trigger names nobody rather than
// guessing, for the same reason it does before the answer lands.
function mockCallerUnavailable() {
  vi.spyOn(globalThis, "fetch").mockImplementation(async () =>
    Response.json({ detail: "boom" }, { status: 403 }),
  )
}

// The trigger reads "Signed in" both before the caller lands and after a read
// that could not name them, so a case about the second has to observe the
// settled read rather than the first paint. This probe, mounted beside the menu
// on the same query, is what those cases wait on.
function CallerProbe() {
  const { isSuccess, isError } = useOrganizationContext()
  return <span>{isSuccess || isError ? "standing settled" : "asking"}</span>
}

// The menu holds a router Link, so it needs a real router; `renderWithRouter`
// mounts it at "/" and resolves the first location before the assertions run.
async function renderMenu(overrides: Partial<DeploymentBootstrap> = {}) {
  await renderWithRouter(
    <AppProviders>
      <DeploymentProvider value={bootstrap(overrides)}>
        <AccountMenu collapsed={false} />
        <CallerProbe />
      </DeploymentProvider>
    </AppProviders>,
  )
}

/** Wait for the caller to have been answered, one way or the other. */
function settled(): Promise<HTMLElement> {
  return screen.findByText("standing settled")
}

async function openMenu(overrides: Partial<DeploymentBootstrap> = {}) {
  await renderMenu(overrides)
  // The trigger's accessible name carries who is signed in, so it is matched on
  // its prefix rather than in full.
  await userEvent
    .setup()
    .click(screen.getByRole("button", { name: /^Account:/ }))
}

afterEach(() => {
  vi.restoreAllMocks()
})

describe("AccountMenu", () => {
  it("opens the account page, rather than naming a destination it cannot reach", async () => {
    mockCaller(OPERATOR)
    await openMenu()

    const link = await screen.findByRole("link", { name: "Account settings" })
    expect(link).toHaveAttribute("href", "/account")
  })

  it("keeps the bundled guide reachable on a phone when no docs site is configured", async () => {
    mockCaller(OPERATOR)
    await openMenu()

    const link = await screen.findByRole("link", { name: "Documentation" })
    expect(link).toHaveAttribute("href", "/docs")
    expect(link).not.toHaveAttribute("target")
  })

  it("names no legal page a deployment has not published", async () => {
    mockCaller(OPERATOR)
    await openMenu()

    expect(screen.queryByRole("link", { name: "Terms of service" })).toBeNull()
    const privacy = await screen.findByRole("button", {
      name: /^Data & Privacy \(/,
    })
    expect(privacy).toBeDisabled()
  })

  it("links both legal rows at the pages a hosted deployment published", async () => {
    mockCaller(OPERATOR)
    await openMenu({
      terms_url: "https://otari.ai/terms",
      privacy_url: "https://otari.ai/privacy",
    })

    const terms = await screen.findByRole("link", { name: "Terms of service" })
    expect(terms).toHaveAttribute("href", "https://otari.ai/terms")
    expect(terms).toHaveAttribute("target", "_blank")

    const privacy = await screen.findByRole("link", { name: "Data & Privacy" })
    expect(privacy).toHaveAttribute("href", "https://otari.ai/privacy")
    expect(privacy).toHaveAttribute("target", "_blank")
  })

  it("keeps the two legal rows independent of each other", async () => {
    mockCaller(OPERATOR)
    await openMenu({ privacy_url: "https://otari.ai/privacy" })

    expect(
      await screen.findByRole("link", { name: "Data & Privacy" }),
    ).toHaveAttribute("href", "https://otari.ai/privacy")
    expect(screen.queryByRole("link", { name: "Terms of service" })).toBeNull()
  })

  it("leaves Data & Privacy disabled for a deployment that published terms alone", async () => {
    mockCaller(OPERATOR)
    await openMenu({ terms_url: "https://otari.ai/terms" })

    expect(
      await screen.findByRole("link", { name: "Terms of service" }),
    ).toHaveAttribute("href", "https://otari.ai/terms")
    expect(
      screen.getByRole("button", { name: /^Data & Privacy \(/ }),
    ).toBeDisabled()
  })

  it("retargets the phone's Documentation row at the deployment's own docs site", async () => {
    mockCaller(OPERATOR)
    await openMenu({ docs_url: "https://docs.otari.ai/en/" })

    const link = await screen.findByRole("link", { name: "Documentation" })
    expect(link).toHaveAttribute("href", "https://docs.otari.ai/en/")
    expect(link).toHaveAttribute("target", "_blank")
    // Still the row the top bar hands off to below `md`, not a second entry
    // point: retargeting it must not make it visible where the cluster already
    // draws one.
    expect(link).toHaveClass("md:hidden")
  })

  // The bug in #832: the trigger named a standing rather than a person, and the
  // standing it named came from the bootstrap, which says what kind of session
  // the deployment issues and not who holds this one. So every signed-in member
  // of a hosted deployment was told they were the operator.
  it("names the person signed in to a hosted deployment, not a role", async () => {
    mockCaller({
      user_id: "44444444-4444-4444-4444-444444444444",
      email: "ada@example.com",
      full_name: "Ada Lovelace",
    })
    await renderMenu({ deployment_type: "hosted", surfaces: HOSTED_SURFACES })

    expect(await screen.findByText("Ada Lovelace")).toBeInTheDocument()
    expect(screen.getByText("AL")).toBeInTheDocument()
    expect(screen.queryByText("Operator")).not.toBeInTheDocument()
  })

  // Unchanged for the deployment that has one identity, and for a reason worth
  // pinning: "Operator" is what first boot *names* that identity, so the word
  // survives as a name where it used to stand in for one.
  it("still names the operator of a standalone gateway, whose name that is", async () => {
    mockCaller(OPERATOR)
    await renderMenu()

    expect(await screen.findByText("Operator")).toBeInTheDocument()
    expect(screen.getByText("OP")).toBeInTheDocument()
  })

  // A member added to the roster by address has no name until they claim the
  // identity, so the address is what identifies them.
  it("falls back to the address of a member who has not claimed a name", async () => {
    mockCaller({
      user_id: "55555555-5555-5555-5555-555555555555",
      email: "ada.lovelace@example.com",
      full_name: null,
    })
    await renderMenu()

    expect(
      await screen.findByText("ada.lovelace@example.com"),
    ).toBeInTheDocument()
    // Off the local part, and split on its punctuation: the host names the
    // deployment rather than the person.
    expect(screen.getByText("AL")).toBeInTheDocument()
  })

  // A name is split on spaces alone. The address path below splits punctuation
  // too, and one rule for both would initial a hyphenated surname off its
  // second half.
  it("initials a hyphenated surname off the name it belongs to", async () => {
    mockCaller({
      user_id: "77777777-7777-7777-7777-777777777777",
      email: null,
      full_name: "Ada Lovelace-Byron",
    })
    await renderMenu()

    expect(await screen.findByText("Ada Lovelace-Byron")).toBeInTheDocument()
    expect(screen.getByText("AL")).toBeInTheDocument()
  })

  // The label is the visible half of #832's fix; this is the half a screen
  // reader gets, and on the collapsed rail it is the only half there is.
  it("names the person in the trigger's accessible name too", async () => {
    mockCaller({
      user_id: "88888888-8888-8888-8888-888888888888",
      email: "ada@example.com",
      full_name: "Ada Lovelace",
    })
    await renderMenu()

    expect(
      await screen.findByRole("button", { name: "Account: Ada Lovelace" }),
    ).toBeInTheDocument()
  })

  // Initials are two characters, not two UTF-16 code units: a name outside the
  // basic plane is two units per character, so indexing one would put half a
  // surrogate pair in the avatar and draw the replacement mark.
  it("takes initials by character, so a name outside the basic plane survives", async () => {
    mockCaller({
      user_id: "66666666-6666-6666-6666-666666666666",
      email: null,
      full_name: "\u{20BB7}\u7530 \u592a\u90ce",
    })
    await renderMenu()

    expect(
      await screen.findByText("\u{20BB7}\u7530 \u592a\u90ce"),
    ).toBeInTheDocument()
    expect(screen.getByText("\u{20BB7}\u592a")).toBeInTheDocument()
  })

  it("names nobody when the caller cannot be read", async () => {
    mockCallerUnavailable()
    await renderMenu()
    await settled()

    expect(screen.getByText("Signed in")).toBeInTheDocument()
    expect(screen.getByText("··")).toBeInTheDocument()
  })

  it("names nobody when the deployment reports no identity at all", async () => {
    mockCaller(undefined)
    await renderMenu()
    await settled()

    expect(screen.getByText("Signed in")).toBeInTheDocument()
  })
})
