import { Link } from "@tanstack/react-router"
import { FiPlus, FiTrash2 } from "react-icons/fi"

import type { PolicyGuardrail } from "@/client"
import { Button } from "@/design-system/actions/Button"
import { IconButton } from "@/design-system/actions/IconButton"
import { Field } from "@/design-system/forms/Field"
import { FieldAction } from "@/design-system/forms/FieldAction"
import { ControlField } from "@/design-system/forms/FieldMessages"

import { ModeToggle } from "./ModeToggle"

function GuardrailsPageLink() {
  return (
    <Link to="/tools/guardrails" className="text-link hover:underline">
      Guardrails page
    </Link>
  )
}

/** The checks every request through a policy runs, whatever the caller asks. */
export function GuardrailsSection({
  guardrails,
  onChange,
  isConfigured,
}: {
  guardrails: PolicyGuardrail[]
  onChange: (next: PolicyGuardrail[]) => void
  /** Whether a guardrails service exists to run them. */
  isConfigured: boolean
}) {
  const update = (index: number, patch: Partial<PolicyGuardrail>) =>
    onChange(
      guardrails.map((guardrail, i) =>
        i === index ? { ...guardrail, ...patch } : guardrail,
      ),
    )

  return (
    <div className="flex flex-col gap-3">
      <ControlField
        label="Always check"
        description="Runs on every request through this policy. Callers can add their own guardrails but cannot weaken these."
      />
      {isConfigured ? null : (
        // Stated beside the disabled control rather than in a tooltip, so it
        // reads on touch and to a screen reader.
        <p
          id="guardrails-unavailable"
          className={`text-caption ${guardrails.length > 0 ? "text-warning" : ""}`}
        >
          {guardrails.length > 0
            ? "No guardrails service is configured, so these cannot run, and with the service set to block every request through this policy is refused. "
            : "No guardrails service is configured, so there would be nothing to call. "}
          Set one up on the <GuardrailsPageLink />.
        </p>
      )}
      {guardrails.map((guardrail, index) => (
        <div key={index} className="flex flex-col gap-3">
          <div className="flex flex-wrap items-end gap-3">
            <Field
              label="Profile"
              value={guardrail.profile}
              onChange={(value) => update(index, { profile: value })}
              placeholder="prompt-injection"
              isRequired
              description="A profile configured on the guardrails service."
            />
            <ModeToggle
              label="Mode"
              value={guardrail.mode}
              onChange={(mode) => update(index, { mode })}
              hint="block rejects a flagged request; monitor records it and serves anyway."
            />
            <ModeToggle
              label="If the service is down"
              value={guardrail.on_unavailable ?? "block"}
              onChange={(mode) => update(index, { on_unavailable: mode })}
              hint="block fails closed, so an outage refuses every request through this policy."
            />
            <FieldAction>
              <IconButton
                variant="ghost"
                isIconOnly
                label={`Remove guardrail ${index + 1}`}
                onPress={() =>
                  onChange(guardrails.filter((_, i) => i !== index))
                }
              >
                <FiTrash2 aria-hidden />
              </IconButton>
            </FieldAction>
          </div>
          {guardrail.mode === "block" &&
          (guardrail.on_unavailable ?? "block") === "block" ? (
            <p className="text-caption text-warning">
              With both set to block, a guardrails-service outage rejects every
              request through this policy, ahead of any backup.
            </p>
          ) : null}
        </div>
      ))}
      <div>
        <Button
          variant="ghost"
          isDisabled={!isConfigured}
          aria-describedby={isConfigured ? undefined : "guardrails-unavailable"}
          onPress={() =>
            onChange([
              ...guardrails,
              { profile: "", mode: "block", on_unavailable: "block" },
            ])
          }
        >
          <FiPlus aria-hidden />
          Add a guardrail
        </Button>
      </div>
      {isConfigured ? (
        <p className="text-caption">
          Profiles and the service are managed on the <GuardrailsPageLink />.
        </p>
      ) : null}
    </div>
  )
}
