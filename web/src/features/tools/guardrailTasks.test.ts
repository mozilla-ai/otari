import { describe, expect, it } from "vitest"

import {
  guardrailOptions,
  guardrailsForTask,
  matchesQuery,
  suggestName,
  taskHelp,
  taskLabel,
  taskOptions,
} from "@/features/tools/guardrailTasks"
import { builtInGuardrail } from "@/tests/fixtures"

const LAKERA = builtInGuardrail()
const ALINIA = builtInGuardrail({
  guardrail_name: "alinia",
  display_name: "Alinia",
  vendor: "Alinia AI",
  // Leads on content safety, but detects prompt injection too. The picker has
  // to offer it under both, which is why the filter reads `categories`.
  primary_category: "content_safety",
  categories: ["content_safety", "prompt_injection", "toxicity"],
})
const PROMPT_GUARD = builtInGuardrail({
  guardrail_name: "prompt_guard_2",
  display_name: "Prompt Guard 2",
  vendor: "Meta",
  backend: "local_encoder",
  requires_api_key: false,
  runnable: false,
  missing_extra: "guardrails-local",
  create_parameters: [],
})
const CATALOG = [PROMPT_GUARD, LAKERA, ALINIA]

describe("taskLabel", () => {
  it("names a known category in the operator's words", () => {
    expect(taskLabel("prompt_injection")).toBe("Prompt injection")
    expect(taskLabel("general_judge")).toBe("Your own rule, judged by a model")
  })

  it("falls back to the category's own name for one it has no copy for", () => {
    // A newer gateway can ship a category this bundle predates. Rendering a
    // blank row would be worse than rendering the wire name.
    expect(taskLabel("deepfake_audio")).toBe("Deepfake audio")
  })
})

describe("taskOptions", () => {
  it("offers only the tasks some guardrail can do", () => {
    const values = taskOptions(CATALOG).map((option) => option.value)
    expect(values).toContain("prompt_injection")
    expect(values).toContain("content_safety")
    expect(values).toContain("toxicity")
    expect(values).not.toContain("pii")
  })

  it("keeps the editorial order and sorts an unknown category last", () => {
    const odd = builtInGuardrail({
      guardrail_name: "odd",
      primary_category: "deepfake_audio" as never,
      categories: ["deepfake_audio" as never],
    })
    expect(taskOptions([...CATALOG, odd]).map((o) => o.value)).toEqual([
      "prompt_injection",
      "content_safety",
      "toxicity",
      "deepfake_audio",
    ])
  })

  it("offers nothing for an empty catalog", () => {
    expect(taskOptions([])).toEqual([])
  })
})

describe("taskHelp", () => {
  it("says what the task catches and how many guardrails can do it", () => {
    expect(taskHelp(CATALOG, "prompt_injection")).toBe(
      "Catches an attempt to override your instructions. 3 guardrails can do this.",
    )
  })

  it("counts one guardrail in the singular", () => {
    expect(taskHelp(CATALOG, "toxicity")).toContain("1 guardrail can do this.")
  })
})

describe("guardrailsForTask", () => {
  it("includes a guardrail that lists the task beyond its headline one", () => {
    const names = guardrailsForTask(CATALOG, "prompt_injection").map(
      (spec) => spec.display_name,
    )
    expect(names).toContain("Alinia")
  })

  it("puts what can run first, then orders by name", () => {
    expect(
      guardrailsForTask(CATALOG, "prompt_injection").map((s) => s.display_name),
    ).toEqual(["Alinia", "Lakera Guard", "Prompt Guard 2"])
  })

  it("is empty before a task is chosen", () => {
    expect(guardrailsForTask(CATALOG, "")).toEqual([])
  })
})

describe("guardrailOptions", () => {
  it("names the vendor and the backend in words an operator reads", () => {
    const [alinia] = guardrailOptions(CATALOG, "prompt_injection")
    expect(alinia.hint).toBe("Alinia AI · hosted API · needs an API key")
  })

  it("disables what cannot run and names the extra that would fix it", () => {
    const rows = guardrailOptions(CATALOG, "prompt_injection")
    const local = rows.find((row) => row.value === "prompt_guard_2")
    expect(local?.isDisabled).toBe(true)
    expect(local?.hint).toBe(
      "Meta · local model · install guardrails-local to use",
    )
  })

  it("says a guardrail is unavailable when it names no extra either", () => {
    const rows = guardrailOptions(
      [builtInGuardrail({ runnable: false, missing_extra: null })],
      "prompt_injection",
    )
    expect(rows[0].isDisabled).toBe(true)
    expect(rows[0].hint).toContain("not available in this build")
  })
})

describe("matchesQuery", () => {
  it("matches the display name, the vendor and the backend", () => {
    expect(matchesQuery(LAKERA, "lake")).toBe(true)
    expect(matchesQuery(ALINIA, "alinia ai")).toBe(true)
    expect(matchesQuery(LAKERA, "hosted")).toBe(true)
    expect(matchesQuery(LAKERA, "meta")).toBe(false)
  })

  it("matches everything on an empty query", () => {
    expect(matchesQuery(LAKERA, "  ")).toBe(true)
  })
})

describe("suggestName", () => {
  it("slugs the task", () => {
    expect(suggestName("prompt_injection", [])).toBe("prompt-injection")
  })

  it("steps past a name already taken", () => {
    expect(suggestName("prompt_injection", ["prompt-injection"])).toBe(
      "prompt-injection-2",
    )
    expect(
      suggestName("prompt_injection", [
        "prompt-injection",
        "prompt-injection-2",
      ]),
    ).toBe("prompt-injection-3")
  })

  it("suggests nothing before a task is chosen", () => {
    expect(suggestName("", [])).toBe("")
  })
})
