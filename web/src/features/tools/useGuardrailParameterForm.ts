import { useEffect, useState } from "react"

import type { GuardrailParameterSpec } from "@/client"
import {
  buildValidateKwargs,
  type ParameterErrors,
  type ParameterValues,
  parameterErrors,
  parseExtraJson,
  type SeededParameters,
  seedParameters,
} from "@/features/tools/guardrailParameters"

/**
 * One kwargs map's half of a guardrail form: the typed values, the raw editor
 * beside them, and the messages a submit produced.
 *
 * A hook rather than five `useState` calls at each call site, which is what
 * keeps the seeding rule in one place: an existing entry and a new one seed from
 * different sources but must both re-seed when the schema arrives, and a catalog
 * that loads a moment after the card does is the ordinary case rather than the
 * edge one.
 *
 * Serves both halves of a guardrail form: the `validate_kwargs` a mandate sends,
 * and the `create_kwargs` a stored definition is built from. The create side
 * reads `values` and assembles them with `buildCreateKwargs`, whose mask rule
 * `build` below does not apply.
 */
export function useGuardrailParameterForm(
  specs: GuardrailParameterSpec[],
  stored: Record<string, unknown> | null | undefined,
  /**
   * What counts as a different selection, and so a re-seed. `profileIdentity`
   * for a mandate; the guardrail class for a stored definition.
   */
  identity: string,
) {
  const [state, setState] = useState<SeededParameters>(() =>
    seedParameters(specs, stored),
  )
  const [issues, setIssues] = useState<ParameterErrors>({})
  const [rawError, setRawError] = useState<string | undefined>(undefined)

  // Two of the three dependencies are serialized, for the reason the workspace
  // scope below is: each is a fresh object on every fetch and on every catalog
  // read, so depending on them by reference would wipe a half-typed parameter
  // whenever any row on the card saved. Parsed back inside the effect so
  // nothing it touches is missing from the dependency list.
  //
  // The identity is the third, because the two above cannot separate two
  // profiles that declare the same parameters, which is the ordinary shape of a
  // pair differing only in the model it pins. Nothing inside the effect reads
  // it.
  const specsJson = JSON.stringify(specs)
  const storedJson = JSON.stringify(stored ?? {})
  // biome-ignore lint/correctness/useExhaustiveDependencies: identity is a re-seed trigger, not an input
  useEffect(() => {
    setState(
      seedParameters(
        JSON.parse(specsJson) as GuardrailParameterSpec[],
        JSON.parse(storedJson) as Record<string, unknown>,
      ),
    )
    setIssues({})
    setRawError(undefined)
  }, [identity, specsJson, storedJson])

  return {
    values: state.values,
    extraJson: state.extraJson,
    issues,
    rawError,
    setValue: (name: string, next: ParameterValues[string]) =>
      setState((current) => ({
        ...current,
        values: { ...current.values, [name]: next },
      })),
    setExtraJson: (next: string) =>
      setState((current) => ({ ...current, extraJson: next })),
    /**
     * Validate on submit and report whether the entry may be sent. Messages
     * appear here rather than on the first keystroke, which is what the forms
     * guide asks for.
     */
    check: (): boolean => {
      const found = parameterErrors(specs, state.values)
      const raw = parseExtraJson(state.extraJson).error
      setIssues(found)
      setRawError(raw)
      return raw === undefined && Object.keys(found).length === 0
    },
    build: () => buildValidateKwargs(specs, state.values, state.extraJson),
  }
}
