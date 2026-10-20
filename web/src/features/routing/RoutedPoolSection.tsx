import { Link } from "@tanstack/react-router"
import { useEffect, useState } from "react"
import { FiArrowDown, FiArrowUp, FiPlus, FiTrash2 } from "react-icons/fi"

import { Button } from "@/design-system/actions/Button"
import { IconButton } from "@/design-system/actions/IconButton"
import { Field } from "@/design-system/forms/Field"
import { FieldAction } from "@/design-system/forms/FieldAction"
import { ControlField } from "@/design-system/forms/FieldMessages"
import { ModelComboBox } from "@/features/models/ModelComboBox"

import {
  type PoolDraft,
  parseWeights,
  poolProblems,
  poolShares,
} from "./policyDraft"
import {
  KNN_BACKEND,
  MAX_CANDIDATES,
  PRIORITY_BACKEND,
  WEIGHTED_BACKEND,
} from "./policyModel"
import { SectionRow } from "./SectionRow"

const COPY = {
  [WEIGHTED_BACKEND]: {
    label: "Split traffic between",
    description:
      "Each request goes to one of these, drawn in proportion to its share. Shares are relative, so 70 and 30 mean the same as 7 and 3. No pricing needed.",
    mark: "Serves on opt-out",
    tooFew: "Splitting traffic one way",
  },
  [PRIORITY_BACKEND]: {
    label: "Send each request to the first with room",
    description:
      "In this order. A model is full when one of its per-model rate limits is, and the next one takes the request. No pricing needed.",
    mark: "Serves on opt-out",
    tooFew: "Ordering one",
  },
  [KNN_BACKEND]: {
    label: "The router chooses between",
    description:
      "For each request, the cheapest of these that past scoring says is good enough. Every model here needs pricing, because the router weighs quality against cost.",
    mark: "Serves when unsure",
    tooFew: "Ranking one",
  },
} as const

/**
 * The models one of which serves each request, for any of the three backends.
 *
 * They share this control because each is "these models, one per request";
 * they differ in what decides, and in whether a share or a move control sits
 * beside each model.
 */
export function RoutedPoolSection({
  pool,
  onChange,
  isAtCap,
}: {
  pool: PoolDraft
  onChange: (next: PoolDraft) => void
  /** Whether the policy already dispatches as many models as it may. */
  isAtCap: boolean
}) {
  const { backend, models, weights, safeIndex } = pool
  const isWeighted = backend === WEIGHTED_BACKEND
  const isPriority = backend === PRIORITY_BACKEND
  const copy = COPY[backend]
  const values = parseWeights(weights)
  const shares = poolShares(pool)
  const problems = poolProblems(pool)
  // The row a model was just moved to, for the focus effect below.
  const [movedTo, setMovedTo] = useState<{
    index: number
    direction: "up" | "down"
  }>()

  const move = (from: number, to: number) => {
    const swap = <T,>(list: T[]) => {
      const next = [...list]
      const [moved] = next.splice(from, 1)
      if (moved !== undefined) next.splice(to, 0, moved)
      return next
    }
    onChange({
      ...pool,
      models: swap(models),
      weights: swap(weights),
      safeIndex: safeIndex === from ? to : safeIndex === to ? from : safeIndex,
    })
    setMovedTo({ index: to, direction: to < from ? "up" : "down" })
  }
  // Rows are keyed by position, so focus would stay on the row a model left.
  // Follow the model instead, onto its other arrow when it reached an end.
  useEffect(() => {
    if (movedTo === undefined) return
    const { index, direction } = movedTo
    const other = direction === "up" ? "down" : "up"
    const target = [direction, other]
      .map((dir) => document.getElementById(`pool-move-${dir}-${index}`))
      .find(
        (button) =>
          button !== null &&
          !button.matches(":disabled, [aria-disabled='true']"),
      )
    target?.focus()
    setMovedTo(undefined)
  }, [movedTo])

  const remove = (index: number) =>
    onChange({
      ...pool,
      models: models.filter((_, i) => i !== index),
      weights: weights.filter((_, i) => i !== index),
      // Keep the mark on the same model; if it went, fall back to the first.
      safeIndex:
        index < safeIndex ? safeIndex - 1 : index === safeIndex ? 0 : safeIndex,
    })

  return (
    <div className="flex flex-col gap-3">
      <ControlField label={copy.label} description={copy.description} />
      {models.map((entry, index) => (
        <SectionRow key={index} id={`pool-hint-${index}`} modelValue={entry}>
          <div className="min-w-56 flex-1">
            <ModelComboBox
              source="catalog"
              label={`Model ${index + 1}`}
              value={entry}
              onChange={(value) =>
                onChange({
                  ...pool,
                  models: models.map((m, i) => (i === index ? value : m)),
                })
              }
              hintPlacement="detached"
              describedBy={`pool-hint-${index}`}
              isRequired
            />
          </div>
          {isWeighted ? (
            // Narrow, so a row's model, share, mark and remove fit one line
            // at the dialog's width.
            <div className="w-28">
              <Field
                label="Share"
                value={weights[index] ?? ""}
                onChange={(value) =>
                  onChange({
                    ...pool,
                    weights: weights.map((w, i) => (i === index ? value : w)),
                  })
                }
                // The percentage, not the number typed: this is the line that
                // says a zero share is drained rather than gone.
                description={
                  !Number.isFinite(values[index] ?? Number.NaN) ||
                  (values[index] ?? 0) < 0
                    ? "A number, zero or more"
                    : (values[index] ?? 0) > 0
                      ? `${Math.round(shares[index] ?? 0)}% of requests`
                      : "No weighted traffic; still tried if another fails"
                }
              />
            </div>
          ) : null}
          <FieldAction>
            <label className="flex items-center gap-2 text-body">
              <input
                type="radio"
                name="router-safe-choice"
                checked={safeIndex === index}
                onChange={() => onChange({ ...pool, safeIndex: index })}
              />
              {copy.mark}
            </label>
          </FieldAction>
          <FieldAction>
            <div className="flex gap-1">
              {isPriority ? (
                <>
                  <IconButton
                    id={`pool-move-up-${index}`}
                    variant="ghost"
                    isIconOnly
                    label={`Move model ${index + 1} up`}
                    isDisabled={index === 0}
                    onPress={() => move(index, index - 1)}
                  >
                    <FiArrowUp aria-hidden />
                  </IconButton>
                  <IconButton
                    id={`pool-move-down-${index}`}
                    variant="ghost"
                    isIconOnly
                    label={`Move model ${index + 1} down`}
                    isDisabled={index === models.length - 1}
                    onPress={() => move(index, index + 1)}
                  >
                    <FiArrowDown aria-hidden />
                  </IconButton>
                </>
              ) : null}
              <IconButton
                variant="ghost"
                isIconOnly
                label={`Remove model ${index + 1}`}
                onPress={() => remove(index)}
              >
                <FiTrash2 aria-hidden />
              </IconButton>
            </div>
          </FieldAction>
        </SectionRow>
      ))}
      {problems.isTooSmall ? (
        <p className="text-caption text-danger">
          Name at least two models. {copy.tooFew} is not a routing decision.
        </p>
      ) : null}
      {problems.hasDuplicate ? (
        <p className="text-caption text-danger">
          Name each model once.{" "}
          {isWeighted
            ? "A model listed twice has one share, not two, so the split saved would not be the one shown."
            : "A pool that repeats a model is refused."}
        </p>
      ) : null}
      {problems.hasBadShare ? (
        <p className="text-caption text-danger">
          Every share is a number of zero or more. Use 0 to drain a model
          without removing it.
        </p>
      ) : problems.isSplitEmpty ? (
        <p className="text-caption text-danger">
          Give at least one model a share above zero, or this policy can never
          send traffic anywhere but its marked model.
        </p>
      ) : null}
      <div className="flex flex-wrap items-center gap-3">
        <Button
          variant="ghost"
          isDisabled={isAtCap}
          onPress={() =>
            onChange({
              ...pool,
              models: [...models, ""],
              // Zero, so adding a model moves no traffic onto it yet.
              weights: [...weights, "0"],
            })
          }
        >
          <FiPlus aria-hidden />
          Add a model
        </Button>
        {isAtCap ? (
          <span className="text-caption">
            A policy dispatches at most {MAX_CANDIDATES} models, counting the
            backups.
          </span>
        ) : null}
      </div>
      <p className="text-caption">
        {isWeighted ? (
          <>
            The marked model serves a caller who sends{" "}
            <code>Otari-Router: off</code>. A model that fails before responding
            moves the request to another in this pool, by the same shares,
            before any backup. Each request is drawn independently, so the
            shares hold over traffic rather than over any ten requests.
          </>
        ) : isPriority ? (
          <>
            Only a model with a per-model limit ever fills; set one under{" "}
            <Link to="/settings" className="text-link hover:underline">
              Settings, Rate limit rules
            </Link>
            . A model that fails before responding moves the request down this
            list before any backup. The marked model serves a caller who sends{" "}
            <code>Otari-Router: off</code>.
          </>
        ) : (
          <>
            The marked model serves whenever the router does not choose: too few
            scored examples, a weakly supported pick, a request carrying tools,
            or a caller sending <code>Otari-Router: off</code>. Until it has
            scored examples (<code>POST /api/v1/routing/preferences/rank</code>
            ), it serves the marked model; open <b>Examples</b> on the policy's
            row to watch it warm up.
          </>
        )}
      </p>
    </div>
  )
}
