import { useState } from "react"

import type { BuiltInGuardrailSpec } from "@/client"
import { InfoBanner } from "@/design-system/feedback/InfoBanner"
import { ComboBoxField } from "@/design-system/forms/ComboBoxField"
import { Select } from "@/design-system/forms/Select"
import {
  guardrailOptions,
  guardrailsForTask,
  matchesQuery,
  taskHelp,
  taskOptions,
} from "@/features/tools/guardrailTasks"

// Choosing a guardrail, in the order an operator actually decides it.
//
// They arrive knowing they want prompt injection caught and not knowing that
// Lakera, Alinia and sixteen others can catch it. So the task comes first and
// the guardrail list is whatever does it, rather than one list of forty names
// from which the operator is expected to recognize the right kind.
//
// The second control is disabled rather than absent before the first is
// answered: it exists and will be usable, which is a different thing from the
// fields below it, which do not exist until a guardrail names them.

export function GuardrailPicker({
  guardrails,
  task,
  guardrailName,
  disabled,
  onChange,
}: {
  guardrails: readonly BuiltInGuardrailSpec[]
  task: string
  guardrailName: string
  disabled?: boolean
  /**
   * The pair, always together. Changing the task clears the guardrail, and that
   * rule lives here rather than at the call site because it is this component's
   * own coupling: a guardrail left selected under a task that no longer offers
   * it is one the operator can no longer see to change.
   */
  onChange: (task: string, guardrailName: string) => void
}) {
  const [query, setQuery] = useState("")
  const tasks = taskOptions(guardrails)

  if (tasks.length === 0) {
    return (
      <InfoBanner tone="warning">
        This build ships no guardrails it can run itself, so there is nothing to
        define here. Guardrails on a separate service are configured above.
      </InfoBanner>
    )
  }

  const forTask = guardrailsForTask(guardrails, task)
  const matched = forTask.filter((spec) => matchesQuery(spec, query))
  const options = guardrailOptions(matched, task)
  const chosen = forTask.find((spec) => spec.guardrail_name === guardrailName)

  return (
    <div className="flex flex-col gap-4">
      <Select
        label="What do you want checked?"
        value={task}
        onChange={(next) => {
          setQuery("")
          onChange(next, "")
        }}
        options={tasks}
        isRequired
        isDisabled={disabled}
        placeholder="Choose a task"
        description={taskHelp(guardrails, task)}
        reserveMessage
      />
      <ComboBoxField
        label="Which guardrail?"
        value={guardrailName}
        onChange={(next) => onChange(task, next)}
        onQueryChange={setQuery}
        options={options}
        isRequired
        isDisabled={disabled || task === ""}
        // `menuTrigger="focus"` rather than `"input"`: this is a pick-from-a-list
        // field that is never the form's first, so opening on focus costs the
        // operator no keystroke.
        menuTrigger="focus"
        placeholder="Choose a guardrail"
        isSourceEmpty={forTask.length === 0}
        emptyMessage="Choose what you want checked first."
        noMatchesMessage="No guardrail here matches that."
        description={
          task === ""
            ? "Choose what you want checked first."
            : (chosen?.description ??
              "What runs the check. Anything dimmed needs a Python extra installed first.")
        }
        reserveMessage
      />
    </div>
  )
}
