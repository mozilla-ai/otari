import type { Meta, StoryObj } from "@storybook/react-vite"

import { Button, type ButtonVariant } from "./Button"

/**
 * Three variants and three sizes, which is the whole surface.
 *
 * The narrowed union is the point of the wrapper: `variant="outline"` is a
 * compile error here where it is a silently unstyled button against HeroUI's
 * own `Button`. There is no story for a retired variant because there is no
 * way to write one.
 */
const meta = {
  title: "Design system/Actions/Button",
  component: Button,
  args: { children: "Create key" },
  parameters: { layout: "padded" },
} satisfies Meta<typeof Button>

export default meta

type Story = StoryObj<typeof meta>

const VARIANTS: ButtonVariant[] = ["primary", "ghost", "danger"]

/** `ghost` is the default: the answer for everything a band is not for. */
export const Default: Story = {}

/**
 * `primary` is the one thing a band exists to do, filled teal under white ink.
 * `ghost` is everything else. `danger` is unfilled on purpose: a filled
 * destructive button reads as a second primary, which is why it is not one.
 */
export const Variants: Story = {
  render: () => (
    <div className="flex items-center gap-3">
      {VARIANTS.map((variant) => (
        <Button key={variant} variant={variant}>
          {variant}
        </Button>
      ))}
    </div>
  ),
}

/** Every variant on the dark artboard, where the danger edge is a separate value. */
export const VariantsDark: Story = {
  render: () => (
    <div className="flex items-center gap-3">
      {VARIANTS.map((variant) => (
        <Button key={variant} variant={variant}>
          {variant}
        </Button>
      ))}
    </div>
  ),
  globals: { theme: "dark" },
}

/** 32, 36 and 40px. Press is `scale(0.98)` at every rung, not a ladder. */
export const Sizes: Story = {
  render: () => (
    <div className="flex items-center gap-3">
      <Button size="sm">Small</Button>
      <Button size="md">Medium</Button>
      <Button size="lg">Large</Button>
    </div>
  ),
}

/**
 * `isPending` disables the button and shows its own spinner, so it must not be
 * paired with `isDisabled` for the same condition. `isDisabled` alone is for a
 * control refused for a reason that is stated nearby.
 */
export const States: Story = {
  render: () => (
    <div className="flex items-center gap-3">
      <Button variant="primary" isPending>
        Saving
      </Button>
      <Button variant="primary" isDisabled>
        Disabled
      </Button>
      <Button variant="danger" isPending>
        Revoking
      </Button>
    </div>
  ),
}

/**
 * A full-width button does not press at all: a wide element displaces too far
 * for the same scale, which is why the transform is dropped rather than tuned.
 */
export const FullWidth: Story = {
  render: () => (
    <div className="flex w-80 flex-col gap-2">
      <Button variant="primary" fullWidth>
        Sign in
      </Button>
      <Button fullWidth>Use a passkey instead</Button>
    </div>
  ),
}
