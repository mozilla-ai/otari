import type { Meta, StoryObj } from "@storybook/react-vite"
import { useState } from "react"

import { Button } from "../actions/Button"
import { CopyField } from "../actions/CopyField"
import { Checkbox } from "../forms/Checkbox"
import { Field } from "../forms/Field"
import { SecretField } from "../forms/SecretField"
import { Select } from "../forms/Select"
import { Tab, TabRow } from "../navigation/TabRow"
import { FormDialog } from "./FormDialog"

const meta = {
  title: "Design system/Feedback/FormDialog",
  component: FormDialog,
  parameters: { layout: "fullscreen" },
  args: {
    isOpen: true,
    onOpenChange: () => {},
    title: "New key",
    description: "The secret is shown once, right after you create it.",
    submitLabel: "Create key",
    onSubmit: () => {},
    isPending: false,
    children: null,
  },
} satisfies Meta<typeof FormDialog>

export default meta

type Story = StoryObj<typeof meta>

/** The two fields every create form has at least one of. */
function KeyFields() {
  const [name, setName] = useState("")
  const [scope, setScope] = useState("workspace")
  return (
    <>
      <Field
        label="Key name"
        value={name}
        onChange={setName}
        placeholder="checkout-service"
        description="Lowercase, hyphens, no spaces."
        autoFocus
        reserveMessage
      />
      <Select
        label="Scope"
        value={scope}
        onChange={setScope}
        options={[
          { value: "workspace", label: "This workspace" },
          { value: "organization", label: "Whole organization" },
        ]}
        reserveMessage={false}
      />
    </>
  )
}

/**
 * `md`, the default, and the size all but three call sites want: a form of two
 * to five fields with nothing to choose between.
 */
export const Medium: Story = {
  render: (args) => (
    <FormDialog {...args}>
      <KeyFields />
    </FormDialog>
  ),
}

/** `sm`, for one or two fields. Narrower than HeroUI's own floor; see the CSS. */
export const Small: Story = {
  args: {
    size: "sm",
    title: "Claim domain",
    description: undefined,
    submitLabel: "Claim domain",
  },
  render: (args) => (
    <FormDialog {...args}>
      <DomainField />
    </FormDialog>
  ),
}

function DomainField() {
  const [domain, setDomain] = useState("mozilla.ai")
  return (
    <Field
      label="Domain"
      value={domain}
      onChange={setDomain}
      description="You will verify it with a DNS record next."
      autoFocus
      reserveMessage
    />
  )
}

/** `lg`, the size a form earns by having two shapes or six or more fields. */
export const LargeWithTabs: Story = {
  args: {
    size: "lg",
    title: "Add provider",
    description:
      "Credentials are encrypted with the server secret key and never leave this gateway.",
    submitLabel: "Add provider",
    footerStart: (
      <p className="text-caption">A test request runs before saving.</p>
    ),
  },
  render: (args) => <ProviderForm {...args} />,
}

function ProviderForm(args: React.ComponentProps<typeof FormDialog>) {
  const [tab, setTab] = useState("known")
  const [secret, setSecret] = useState("")
  const [isDefault, setIsDefault] = useState(false)
  return (
    <FormDialog
      {...args}
      tabs={
        <TabRow>
          <Tab isActive={tab === "known"} onPress={() => setTab("known")}>
            Known provider
          </Tab>
          <Tab isActive={tab === "custom"} onPress={() => setTab("custom")}>
            Custom endpoint
          </Tab>
        </TabRow>
      }
    >
      <SecretField
        label="API key"
        value={secret}
        onChange={setSecret}
        description="Stored encrypted. Autofill is off for this field."
        reserveMessage
      />
      <Checkbox isSelected={isDefault} onChange={setIsDefault}>
        Make this the default route for Anthropic models
      </Checkbox>
    </FormDialog>
  )
}

/**
 * The mutation is in flight. The primary keeps its resting width, because the
 * spinner replaces the label in place; Cancel and the close control are
 * disabled for the same duration, so the footer's height never moves.
 */
export const Submitting: Story = {
  args: { isPending: true },
  render: (args) => (
    <FormDialog {...args}>
      <KeyFields />
    </FormDialog>
  ),
}

/**
 * The gateway refused. The dialog stays open holding its own fields, and the
 * banner mounts at the top of the body, where `ConfirmDialog` already puts it.
 */
export const RequestFailed: Story = {
  args: {
    error: new Error(
      "Could not claim mozilla.ai. Another organization already verified it.",
    ),
  },
  render: (args) => (
    <FormDialog {...args}>
      <KeyFields />
    </FormDialog>
  ),
}

/**
 * `isDirty`, after Escape or a click outside. The guard is in the footer rather
 * than in a second dialog: a dialog never opens a dialog.
 */
export const DirtyGuard: Story = {
  args: { isDirty: true },
  render: (args) => (
    <FormDialog {...args}>
      <KeyFields />
    </FormDialog>
  ),
}

/**
 * The success step is not a prop. The caller swaps the children and the submit
 * label once the mutation resolves; the frame stays mounted, so the secret
 * appears where the form was rather than in a second surface.
 */
export const SuccessStep: Story = {
  args: {
    title: "Key created",
    description: "Copy it now. Otari stores a hash and cannot show it again.",
    submitLabel: "Done",
    footerStart: <Button>Create another</Button>,
  },
  render: (args) => (
    <FormDialog {...args}>
      <CopyField
        label="checkout-service"
        value="otari-sk-9f3a1c77b0e244d1e8a972fc0a3bc5d86e10f4b2"
      />
      <p className="text-caption">
        Owner Léa Fontaine-Whitaker · $500.00 per month · never expires
      </p>
    </FormDialog>
  ),
}

/**
 * A body taller than the height budget. The header and the footer stay put and
 * the body scrolls between them, and the header gains its rule only once
 * content is under it.
 */
export const ScrollingBody: Story = {
  args: {
    size: "lg",
    title: "Add MCP server",
    description: "Otari calls it on the operator's behalf.",
    submitLabel: "Add server",
  },
  render: (args) => (
    <FormDialog {...args}>
      <ManyFields />
    </FormDialog>
  ),
}

function ManyFields() {
  const [values, setValues] = useState<string[]>(() => Array(8).fill(""))
  return (
    <>
      {values.map((value, index) => (
        <Field
          // The list is fixed-length and positional, so the index is the
          // identity: there is nothing to reorder and nothing to key on.
          key={index}
          label={`Header ${index + 1}`}
          value={value}
          onChange={(next) =>
            setValues((prev) =>
              prev.map((item, at) => (at === index ? next : item)),
            )
          }
          // No description, so no reserved line: the line exists for an error
          // to replace a description in, and eight of them reserved against
          // nothing added eight empty rungs to a body that already scrolls.
          //
          // Spelled `={false}` rather than omitted, which reads as redundant
          // and is not: `FieldMessages` defaults `reserve` to true and `Field`
          // forwards the prop undefined, so a field that says nothing reserves
          // a line anyway. Measured, omitting it leaves the row at 83px where
          // this brings it to 60.
          reserveMessage={false}
        />
      ))}
    </>
  )
}

/**
 * Below 640px every size is a full-screen sheet: the dialog fills the frame,
 * the inputs are 16px so iOS does not zoom, and the footer clears the home
 * indicator.
 *
 * `mobile2` is Storybook's own 414px preset, the nearest one to the 390px frame
 * this was drawn at. The width is what matters: the sheet is a media query on
 * the preview frame, so a wrapper `<div>` would not produce it. The dialog
 * portals to the frame's `<body>` and never sees a wrapper at all.
 */
/**
 * The same sheet, landscape. `isRotated` puts the phone at 844x390, which is
 * wider than 640 and shorter than it: the sheet's geometry keys on either
 * dimension, so a viewport that cannot afford the 120px gap vertically gets the
 * sheet rather than a third layout. Without that this was a 150px dialog with
 * 137px of header and footer in it.
 *
 * The footer stays a row here, unlike the portrait story below: stacking is
 * keyed on width alone, because 844px affords a row and `(height <= 639px)`
 * also matches an unmaximized desktop window.
 */
export const PhoneSheetLandscape: Story = {
  globals: { viewport: { value: "mobile2", isRotated: true } },
  args: {
    footerStart: (
      <p className="text-caption">In effect for new requests within 30s.</p>
    ),
  },
  render: (args) => (
    <FormDialog {...args}>
      <KeyFields />
    </FormDialog>
  ),
}

export const PhoneSheet: Story = {
  globals: { viewport: { value: "mobile2", isRotated: false } },
  // With a `footerStart` caption, which is the case that does not fit a row:
  // beside two buttons in a 390px sheet this wrapped to three lines, so below
  // 640px the footer stacks and each control takes the full width.
  args: {
    footerStart: (
      <p className="text-caption">In effect for new requests within 30s.</p>
    ),
  },
  render: (args) => (
    <FormDialog {...args}>
      <KeyFields />
    </FormDialog>
  ),
}

/**
 * A submit the form is not ready for. Shown disabled rather than allowed to
 * fail, with the reason beside the control that is missing: a press that does
 * nothing teaches nothing.
 */
export const SubmitBlocked: Story = {
  args: { isSubmitDisabled: true },
  render: (args) => (
    <FormDialog {...args}>
      <Field
        label="Key name"
        value=""
        onChange={() => {}}
        description="Required. Lowercase, hyphens, no spaces."
        isInvalid
        errorMessage="Give the key a name."
        reserveMessage
      />
    </FormDialog>
  ),
}

/**
 * Content that cannot be recovered once the frame closes, which is the one case
 * for taking the dismiss away: a key's plaintext secret is shown once. There is
 * no close control and no Cancel, because both are dismissals; the footer's one
 * action is the acknowledgement and the only way out.
 */
export const NotDismissable: Story = {
  args: {
    isDismissable: false,
    title: "Key created",
    description: "Copy it now. Otari stores a hash and cannot show it again.",
    submitLabel: "I\u2019ve saved this key",
    footerStart: <Button>Create another</Button>,
  },
  render: (args) => (
    <FormDialog {...args}>
      <CopyField
        label="Secret key"
        value="otari-sk-9f3a1c77b0e244d1e8a972fc0a3bc5d86e10f4b2"
      />
    </FormDialog>
  ),
}

/** Driven from a trigger, which is how a heading row actually opens it. */
export const FromTrigger: Story = {
  render: (args) => {
    const [open, setOpen] = useState(false)
    return (
      <div className="p-6">
        <Button variant="primary" onPress={() => setOpen(true)}>
          Create key
        </Button>
        <FormDialog
          {...args}
          isOpen={open}
          onOpenChange={setOpen}
          onSubmit={() => setOpen(false)}
        >
          <KeyFields />
        </FormDialog>
      </div>
    )
  },
}
