import { render, screen } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { useState } from "react"
import { describe, expect, it } from "vitest"

import { AuthEmailField } from "@/features/auth/AuthFields"

function EmailForm() {
  const [email, setEmail] = useState("")
  return (
    <form onSubmit={(event) => event.preventDefault()}>
      <AuthEmailField value={email} onChange={setEmail} />
      <button type="submit">Send</button>
    </form>
  )
}

describe("AuthEmailField", () => {
  it("says why an address it marks invalid cannot be used", async () => {
    const user = userEvent.setup()
    render(<EmailForm />)

    await user.type(screen.getByLabelText(/Email/), "not-an-address")
    await user.click(screen.getByRole("button", { name: "Send" }))

    const field = screen.getByLabelText(/Email/)
    expect(field).toHaveAttribute("aria-invalid", "true")
    expect(
      await screen.findByText("Enter a valid email address."),
    ).toBeInTheDocument()
  })

  it("asks for an address when the field is left empty", async () => {
    const user = userEvent.setup()
    render(<EmailForm />)

    await user.click(screen.getByRole("button", { name: "Send" }))

    expect(await screen.findByText("Enter your email.")).toBeInTheDocument()
  })
})
