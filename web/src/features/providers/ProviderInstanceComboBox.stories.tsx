import type { Meta, StoryObj } from "@storybook/react-vite"
import { useState } from "react"

import { API_ROOT } from "@/shared/api/client"

import { ProviderInstanceComboBox } from "./ProviderInstanceComboBox"

/**
 * The provider instances a caller can route to, for a form that refers to one
 * rather than creating one (a spend ceiling narrowed to a single provider).
 *
 * Read off the model catalog, whose ids are `instance:model`, because
 * `/v1/providers` is operator-only and the one form asking this question is
 * shown to an organization admin who is not an operator. An alias or a routing
 * policy carries no instance and contributes nothing.
 */
function catalog(ids: string[]) {
  return {
    object: "list",
    data: ids.map((id) => ({
      id,
      object: "model",
      created: 0,
      owned_by: id.split(":")[0],
      pricing_source: "none",
    })),
  }
}

const meta = {
  title: "Dashboard/Providers/ProviderInstanceComboBox",
  component: ProviderInstanceComboBox,
  args: {
    label: "Provider instance",
    value: "",
    onChange: () => {},
    placeholder: "openai-eu",
    description:
      "Optional. Narrows the cap to one provider; leave blank to cap spend across every provider.",
  },
  parameters: {
    api: {
      [`${API_ROOT}/models`]: catalog([
        "openai-eu:gpt-4o",
        "openai-eu:gpt-4o-mini",
        "anthropic:claude-sonnet-4-5",
        "vllm-lab:mistral-small",
        "fast",
      ]),
    },
  },
} satisfies Meta<typeof ProviderInstanceComboBox>

export default meta

type Story = StoryObj<typeof meta>

export const Default: Story = {
  render: (args) => {
    const [value, setValue] = useState("")
    return (
      <div className="w-[24rem]">
        <ProviderInstanceComboBox {...args} value={value} onChange={setValue} />
      </div>
    )
  },
}

/**
 * Nothing configured yet, which is the ordinary state of a fresh deployment
 * rather than an edge case: the popover says what would fill it, and the field
 * still takes a name typed by hand.
 */
export const NothingConfigured: Story = {
  parameters: { api: { [`${API_ROOT}/models`]: catalog([]) } },
  render: (args) => (
    <div className="w-[24rem]">
      <ProviderInstanceComboBox {...args} />
    </div>
  ),
}
