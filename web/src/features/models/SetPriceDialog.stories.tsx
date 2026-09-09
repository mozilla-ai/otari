import { Button } from "@heroui/react"
import type { Meta, StoryObj } from "@storybook/react-vite"
import { useState } from "react"

import { ApiError } from "@/shared/api/client"

import { SetPriceDialog } from "./SetPriceDialog"

/**
 * Manual per-million rates for a model the pricing catalog does not cover.
 *
 * Fully controlled and entirely offline: it takes `isPending` and `error` as props
 * and reports a submit, so the page owns the mutation. That is what makes it the
 * easiest dialog in the tree to put in a catalog.
 */
const meta = {
  title: "Dashboard/Models/SetPriceDialog",
  component: SetPriceDialog,
  args: {
    isOpen: true,
    onOpenChange: () => {},
    isPending: false,
    error: null,
    onSubmit: () => {},
  },
} satisfies Meta<typeof SetPriceDialog>

export default meta

type Story = StoryObj<typeof meta>

/** Pricing one model, reached from its row in the catalog. */
export const SingleModel: Story = {
  args: { targetCount: 1 },
}

/** Pricing a bulk selection: the copy is a function of the count. */
export const BulkSelection: Story = {
  args: { targetCount: 23 },
}

/**
 * `collectModelKey` adds the model-key field, for pricing a model that is not in
 * the catalog at all rather than one selected from it.
 */
export const CollectsModelKey: Story = {
  args: { collectModelKey: true },
}

/** Prefilled, which is how "edit this price" opens. */
export const WithInitialModelKey: Story = {
  args: { collectModelKey: true, initialModelKey: "openai:gpt-4o-mini" },
}

/** Custom heading and copy, which the org-level override flow passes. */
export const CustomCopy: Story = {
  args: {
    targetCount: 4,
    title: "Override rates",
    description: (count) =>
      `These rates replace the catalog's for ${count} model${count === 1 ? "" : "s"}, for this organization only.`,
  },
}

export const Pending: Story = {
  args: { targetCount: 1, isPending: true },
}

/** A refusal: the dialog stays open holding the gateway's own message. */
export const WithError: Story = {
  args: {
    targetCount: 1,
    error: new ApiError(422, "Output price must be greater than zero."),
  },
}

/** Driven from a trigger, so the fields and validation can actually be exercised. */
export const FromTrigger: Story = {
  render: (args) => {
    const [open, setOpen] = useState(false)
    const [saved, setSaved] = useState<string | null>(null)
    return (
      <div className="flex flex-col items-start gap-3">
        <Button variant="primary" onPress={() => setOpen(true)}>
          Set price
        </Button>
        {saved ? <p className="text-caption">Submitted: {saved}</p> : null}
        <SetPriceDialog
          {...args}
          collectModelKey
          isOpen={open}
          onOpenChange={setOpen}
          onSubmit={(rates, modelKey) => {
            setSaved(
              `${modelKey || "(selected models)"} in $${rates.input_price_per_million}/M, out $${rates.output_price_per_million}/M`,
            )
            setOpen(false)
          }}
        />
      </div>
    )
  },
}
