import type { Meta, StoryObj } from "@storybook/react-vite"
import { useState } from "react"
import { FiTrash2 } from "react-icons/fi"

import { Button } from "../actions/Button"
import { IconButton } from "../actions/IconButton"
import { ListDetail, ListDetailRow } from "./ListDetail"

/**
 * A list of records beside the one that is open, at 1:1.75.
 *
 * Documented as a pair, like `TabRow` and `Tab`: the frame owns the two columns
 * and the partition between them, the row owns one record and the lane its
 * actions sit in, and neither is any use alone.
 *
 * Nothing here holds state. Which record is open is the page's, which is what
 * lets one selection drive the URL, the detail column, and (below `md`) which
 * of the two columns is the one on screen. An editor is not one of the two
 * columns: a record is created and changed in a `FormDialog` over the page.
 *
 * **Narrow the canvas to see the other half of the layout.** From `md` up both
 * columns are on screen and `isDetailShown` does nothing; below it exactly one
 * is, and the Back control is the way between them.
 */
const meta = {
  title: "Design system/Layout/ListDetail",
  component: ListDetail,
  args: {
    listLabel: "Policies",
    list: null,
    detail: null,
    isDetailShown: false,
    onShowList: () => {},
  },
  parameters: { layout: "padded" },
} satisfies Meta<typeof ListDetail>

export default meta

type Story = StoryObj<typeof meta>

const POLICIES = [
  { name: "fast", serves: "openai:gpt-5-mini  +1 on failure" },
  { name: "cheap", serves: "Learned · 2 candidates, openai:gpt-5 by default" },
  { name: "balanced", serves: "Weighted · 70% / 30% across 2 models" },
  { name: "strict", serves: "anthropic:claude-sonnet-4-5" },
]

function Facts({ serves }: { serves: string }) {
  return (
    <div className="grid gap-4 sm:grid-cols-3">
      {[
        ["Serves", serves],
        ["Applies to", "Every caller"],
        ["Source", "STORED"],
      ].map(([label, value]) => (
        <div key={label} className="flex min-w-0 flex-col gap-1">
          <span className="text-overline">{label}</span>
          <span className="text-body">{value}</span>
        </div>
      ))}
    </div>
  )
}

/** A record open, which is the state the shape exists for. */
export const Default: Story = {
  render: () => {
    const [open, setOpen] = useState<string | undefined>("fast")
    const selected = POLICIES.find((policy) => policy.name === open)
    return (
      <ListDetail
        listLabel="Policies"
        backLabel="All policies"
        detailLabel="Policy detail"
        listAction={
          <Button size="sm" variant="primary" onPress={() => {}}>
            Create policy
          </Button>
        }
        isDetailShown={selected !== undefined}
        onShowList={() => setOpen(undefined)}
        list={POLICIES.map((policy) => (
          <ListDetailRow
            key={policy.name}
            label={policy.name}
            isSelected={policy.name === open}
            onSelect={() => setOpen(policy.name)}
            actions={
              <IconButton label={`Delete ${policy.name}`} onPress={() => {}}>
                <FiTrash2 aria-hidden="true" className="size-4" />
              </IconButton>
            }
          >
            {policy.serves}
          </ListDetailRow>
        ))}
        detail={
          selected === undefined ? (
            <div className="px-4 py-10 text-center text-caption">
              Pick a policy to see what it serves.
            </div>
          ) : (
            <div className="flex flex-col">
              <div className="flex flex-wrap items-center justify-between gap-3 border-b border-border px-4 py-3">
                <h2 className="text-title">{selected.name}</h2>
                <div className="flex gap-2">
                  <Button size="sm" onPress={() => {}}>
                    Edit
                  </Button>
                  <Button size="sm" onPress={() => {}}>
                    Delete
                  </Button>
                </div>
              </div>
              <div className="px-4 py-4">
                <Facts serves={selected.serves} />
              </div>
            </div>
          )
        }
      />
    )
  },
}

/**
 * Nothing open, which is where a wide viewport lands before the first click.
 * The detail column holds what to do rather than a blank half-page.
 */
export const NothingOpen: Story = {
  render: () => (
    <ListDetail
      listLabel="Policies"
      isDetailShown={false}
      onShowList={() => {}}
      list={POLICIES.map((policy) => (
        <ListDetailRow
          key={policy.name}
          label={policy.name}
          isSelected={false}
          onSelect={() => {}}
        >
          {policy.serves}
        </ListDetailRow>
      ))}
      detail={
        <div className="px-4 py-10 text-center text-caption">
          Pick a policy to see what it serves.
        </div>
      }
    />
  ),
}

/**
 * An empty list, where the only thing to do with the column is start the first
 * record, so the column is the button. It is a real one: a whole-column press
 * target that a keyboard cannot reach is a control that is not there.
 */
export const Empty: Story = {
  render: () => (
    <ListDetail
      listLabel="Policies"
      listAction={
        <Button size="sm" variant="primary" onPress={() => {}}>
          Create policy
        </Button>
      }
      isDetailShown={false}
      onShowList={() => {}}
      list={null}
      isEmpty
      empty="Create your first policy"
      emptyAction={
        <Button size="sm" variant="primary" onPress={() => {}}>
          Create policy
        </Button>
      }
      detail={
        <div className="flex flex-col gap-2 px-4 py-5">
          <h2 className="text-title">No policies yet</h2>
          <p className="text-body">
            A policy is a name your callers send as their model, and what serves
            it.
          </p>
        </div>
      }
    />
  ),
}

/**
 * The same empty column for a reader who cannot write: the sentence without
 * the invitation, since pressing it would only be refused.
 */
export const EmptyReadOnly: Story = {
  render: () => (
    <ListDetail
      listLabel="Policies"
      isDetailShown={false}
      onShowList={() => {}}
      list={null}
      empty="No policies yet."
      detail={
        <div className="px-4 py-10 text-center text-caption">
          Your organization&apos;s admins define these.
        </div>
      }
    />
  ),
}

/**
 * Below `md`, where the detail column is the one on screen and the Back control
 * is the way out of it. The frame is drawn here as it is at every width, so
 * narrow the canvas to see the list disappear behind it.
 */
export const DetailShown: Story = {
  render: () => (
    <ListDetail
      listLabel="Policies"
      backLabel="All policies"
      detailLabel="Policy detail"
      isDetailShown
      onShowList={() => {}}
      list={POLICIES.map((policy) => (
        <ListDetailRow
          key={policy.name}
          label={policy.name}
          isSelected={policy.name === "cheap"}
          onSelect={() => {}}
        >
          {policy.serves}
        </ListDetailRow>
      ))}
      detail={
        <div className="flex flex-col gap-4 px-4 py-5">
          <h2 className="text-title">cheap</h2>
          <Facts serves="Learned · 2 candidates, openai:gpt-5 by default" />
        </div>
      }
    />
  ),
}

/** The open record on the dark artboard, where the selection tint is its own value. */
export const Dark: Story = {
  render: () => (
    <ListDetail
      listLabel="Policies"
      isDetailShown={false}
      onShowList={() => {}}
      list={POLICIES.map((policy) => (
        <ListDetailRow
          key={policy.name}
          label={policy.name}
          isSelected={policy.name === "cheap"}
          onSelect={() => {}}
        >
          {policy.serves}
        </ListDetailRow>
      ))}
      detail={
        <div className="flex flex-col gap-4 px-4 py-5">
          <h2 className="text-title">cheap</h2>
          <Facts serves="Learned · 2 candidates, openai:gpt-5 by default" />
        </div>
      }
    />
  ),
  globals: { theme: "dark" },
}
