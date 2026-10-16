import { render, screen } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import type { ReactNode } from "react"
import { afterEach, describe, expect, it, vi } from "vitest"

import { Login } from "@/features/auth/Login"
import {
  readSignInAddress,
  rememberSignInAddress,
} from "@/features/auth/signInAddress"
import { DeploymentProvider } from "@/shared/hooks/useDeployment"
import { bootstrap } from "@/tests/fixtures"
import { AppProviders } from "@/tests/providers"

/**
 * What the sign-in screen does before anything is typed: which field holds
 * focus, what it already holds, and where the tab key goes from there.
 */

vi.mock("@/shared/telemetry/overlayTelemetry", async () => {
  const { telemetrySpy } = await import("@/tests/telemetry")
  return { useTelemetry: vi.fn(() => telemetrySpy) }
})

function Mounted({
  children,
  oauthProviders = [],
}: {
  children: ReactNode
  oauthProviders?: string[]
}) {
  return (
    <AppProviders>
      <DeploymentProvider
        value={bootstrap({
          sign_in_methods: ["password"],
          mail_ready: true,
          oauth_providers: oauthProviders,
        })}
      >
        {children}
      </DeploymentProvider>
    </AppProviders>
  )
}

/** A desk: a mouse or a trackpad, not a finger. */
function stubPointer(fine: boolean) {
  vi.stubGlobal(
    "matchMedia",
    vi.fn((query: string) => ({
      matches: query === "(pointer: fine)" ? fine : false,
      media: query,
      addEventListener: () => {},
      removeEventListener: () => {},
      addListener: () => {},
      removeListener: () => {},
    })),
  )
}

describe("the sign-in screen on arrival", () => {
  afterEach(() => {
    vi.restoreAllMocks()
    vi.unstubAllGlobals()
    window.localStorage.clear()
    window.sessionStorage.clear()
  })

  it("focuses the address on a desk", () => {
    stubPointer(true)
    render(
      <Mounted>
        <Login />
      </Mounted>,
    )

    expect(screen.getByLabelText("Email")).toHaveFocus()
  })

  it("leaves focus alone on a touch screen, where it raises the keyboard", () => {
    stubPointer(false)
    render(
      <Mounted>
        <Login />
      </Mounted>,
    )

    expect(screen.getByLabelText("Email")).not.toHaveFocus()
  })

  it("tabs from the address to the password, then to the recovery link", async () => {
    stubPointer(true)
    const user = userEvent.setup()
    render(
      <Mounted>
        <Login />
      </Mounted>,
    )

    expect(screen.getByLabelText("Email")).toHaveFocus()
    await user.tab()
    expect(screen.getByLabelText("Password")).toHaveFocus()
    await user.tab()
    expect(
      screen.getByRole("link", { name: "Forgot your password?" }),
    ).toHaveFocus()
  })

  it("keeps the password's label row content-sized, so a refusal sits beside the label", () => {
    // jsdom lays nothing out, so the placement class is what is pinned: in the
    // grid that shares its line with the recovery link, a stretched row would
    // push the refusal against the link instead.
    stubPointer(true)
    render(
      <Mounted>
        <Login />
      </Mounted>,
    )

    const labelRow = screen.getByText("Password").closest("div")
    expect(labelRow?.parentElement).toHaveClass("grid")
    expect(labelRow).toHaveClass("justify-self-start")
  })

  it("fills in the address an invitation link handed over, and asks for the password", () => {
    stubPointer(true)
    rememberSignInAddress("ada@example.com")
    render(
      // A provider folds the form away by default; a handed-over address is
      // a request to type the password, so the form opens.
      <Mounted oauthProviders={["google"]}>
        <Login />
      </Mounted>,
    )

    expect(screen.getByLabelText("Email")).toHaveValue("ada@example.com")
    expect(screen.getByLabelText("Password")).toHaveFocus()
    // Used once: a later visit to the sign-in screen starts empty.
    expect(readSignInAddress()).toBe("")
  })

  it("hands the typed address to the recovery page", async () => {
    stubPointer(true)
    const user = userEvent.setup()
    render(
      <Mounted>
        <Login />
      </Mounted>,
    )

    await user.type(screen.getByLabelText("Email"), "ada@example.com")
    await user.click(
      screen.getByRole("link", { name: "Forgot your password?" }),
    )

    expect(readSignInAddress()).toBe("ada@example.com")
  })
})
