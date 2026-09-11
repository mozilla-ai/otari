import { Button } from "@heroui/react"
import type { Meta, StoryObj } from "@storybook/react-vite"
import { useState } from "react"

import { AuthEmailField, AuthPasswordField } from "./AuthFields"
import { PublicAuthLayout, PublicAuthLink } from "./PublicAuthLayout"

/** Public auth pages share the same background, card, and content slots. */
const meta = {
  title: "Dashboard/Auth/PublicAuthLayout",
  component: PublicAuthLayout,
  args: { title: "Create your account", children: null },
  parameters: { layout: "fullscreen" },
} satisfies Meta<typeof PublicAuthLayout>

export default meta

type Story = StoryObj<typeof meta>

export const SignUp: Story = {
  render: () => {
    const [email, setEmail] = useState("")
    const [password, setPassword] = useState("")
    return (
      <PublicAuthLayout
        title="Create your account"
        description="You were invited to the Otari gateway at gateway.example.com."
        footer={
          <>
            Already have an account?{" "}
            <PublicAuthLink to="/">Sign in</PublicAuthLink>
          </>
        }
      >
        <form
          className="flex flex-col gap-4"
          onSubmit={(event) => event.preventDefault()}
        >
          <AuthEmailField value={email} onChange={setEmail} />
          <AuthPasswordField
            label="Password"
            value={password}
            onChange={setPassword}
            autoComplete="new-password"
            description="At least 12 characters."
          />
          <Button type="submit" variant="primary">
            Create account
          </Button>
        </form>
      </PublicAuthLayout>
    )
  },
}

/** Title and copy only, which is what a terminal state looks like. */
export const CheckYourEmail: Story = {
  render: () => (
    <PublicAuthLayout
      title="Check your email"
      description="We sent a verification link to ops@example.com. It expires in an hour."
      footer={<PublicAuthLink to="/">Back to sign in</PublicAuthLink>}
    >
      <p className="text-caption">
        No email? Check spam, or ask your gateway operator whether mail delivery
        is configured.
      </p>
    </PublicAuthLayout>
  ),
}

/** `description` is a node, so it can carry an inline link or a code sample. */
export const RichDescription: Story = {
  render: () => (
    <PublicAuthLayout
      title="Reset your password"
      description={
        <>
          Enter the address you signed up with. If it has an account, we will
          send a reset link.{" "}
          <PublicAuthLink to="/">Sign in instead</PublicAuthLink>
        </>
      }
    >
      <form
        className="flex flex-col gap-4"
        onSubmit={(event) => event.preventDefault()}
      >
        <AuthEmailField value="" onChange={() => {}} />
        <Button type="submit" variant="primary">
          Send reset link
        </Button>
      </form>
    </PublicAuthLayout>
  ),
}

/** No footer and no description: the narrowest the card gets. */
export const Minimal: Story = {
  render: () => (
    <PublicAuthLayout title="Signing you in…">
      <p className="text-caption">Verifying your link.</p>
    </PublicAuthLayout>
  ),
}
