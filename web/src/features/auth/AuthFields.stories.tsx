import { Button } from "@heroui/react"
import type { Meta, StoryObj } from "@storybook/react-vite"
import { useState } from "react"

import { AuthEmailField, AuthPasswordField, AuthTextField } from "./AuthFields"

/**
 * The three pre-session fields, in one file because they are only ever used
 * together: every public auth page composes some subset of them.
 *
 * They are separate from `shared/components/Field` because a pre-session form has
 * different obligations, correct `autoComplete` values so a password manager
 * fills and saves the right thing, and no dependency on anything a session owns.
 */
const meta = {
  title: "Dashboard/Auth/AuthFields",
  component: AuthEmailField,
  args: { value: "", onChange: () => {} },
} satisfies Meta<typeof AuthEmailField>

export default meta

type Story = StoryObj<typeof meta>

export const Email: Story = {
  render: () => {
    const [email, setEmail] = useState("")
    return (
      <div className="w-[22rem]">
        <AuthEmailField value={email} onChange={setEmail} />
      </div>
    )
  },
}

export const EmailWithDescription: Story = {
  render: () => (
    <div className="w-[22rem]">
      <AuthEmailField
        value="ops@example.com"
        onChange={() => {}}
        description="The address your invitation was sent to."
      />
    </div>
  ),
}

/**
 * `autoComplete` is required, not optional: a password manager needs to know
 * whether it is filling an existing credential or saving a new one, and the two
 * forms look identical without it.
 */
export const Passwords: Story = {
  render: () => {
    const [current, setCurrent] = useState("")
    const [next, setNext] = useState("")
    return (
      <div className="flex w-[22rem] flex-col gap-4">
        <AuthPasswordField
          label="Password"
          value={current}
          onChange={setCurrent}
          autoComplete="current-password"
        />
        <AuthPasswordField
          label="New password"
          value={next}
          onChange={setNext}
          autoComplete="new-password"
          description="At least 12 characters."
        />
      </div>
    )
  },
}

/** The generic one, for the master key on a gateway that has not been claimed. */
export const Text: Story = {
  render: () => {
    const [key, setKey] = useState("")
    return (
      <div className="w-[22rem]">
        <AuthTextField
          label="Master key"
          value={key}
          onChange={setKey}
          autoComplete="off"
          description="From your gateway's config.yml, or the one-time key it printed on first boot."
        />
      </div>
    )
  },
}

/** A whole sign-in form, which is what they add up to. */
export const SignInForm: Story = {
  render: () => {
    const [email, setEmail] = useState("")
    const [password, setPassword] = useState("")
    return (
      <form
        className="flex w-[22rem] flex-col gap-4"
        onSubmit={(event) => event.preventDefault()}
      >
        <AuthEmailField value={email} onChange={setEmail} />
        <AuthPasswordField
          label="Password"
          value={password}
          onChange={setPassword}
          autoComplete="current-password"
        />
        <Button type="submit" variant="primary">
          Sign in
        </Button>
      </form>
    )
  },
}
