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
} from "@/features/guardrails/guardrailParameters"

/**
 * The parameter half of one guardrail form: the typed values, the raw editor
 * beside them, and the messages a submit produced.
 *
 * One hook for every form that edits a guardrail's parameters, which is what
 * keeps the seeding rule in one place: each form seeds from a different source
 * but must re-seed when the schema arrives, and a catalog that loads a moment
 * after the form does is the ordinary case rather than the edge one.
 */
export function useGuardrailParameterForm(
  specs: GuardrailParameterSpec[],
  stored: Record<string, unknown> | null | undefined,
  /** From `profileIdentity`, which says what counts as a different profile. */
  identity: string,
) {
  const [state, setState] = useState<SeededParameters>(() =>
    seedParameters(specs, stored),
  )
  const [issues, setIssues] = useState<ParameterErrors>({})
  const [rawError, setRawError] = useState<string | undefined>(undefined)

  // Two of the three dependencies are serialized: each is a fresh object on
  // every fetch and every catalog read, so depending on them by reference
  // would wipe a half-typed parameter whenever anything else on the page saved.
  // Parsed back inside the effect so nothing it touches is missing from the
  // dependency list.
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
