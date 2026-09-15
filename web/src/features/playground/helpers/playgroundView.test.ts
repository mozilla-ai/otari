import { describe, expect, it } from "vitest"

import { EMPTY_PANEL, type PanelState } from "./playgroundTypes"
import {
  derivePlaygroundGate,
  haveBothPanelsAnswered,
  isAnyPanelBusy,
  shouldShowWelcome,
} from "./playgroundView"

const answered: PanelState = {
  ...EMPTY_PANEL,
  model: "openai:gpt-4o",
  turns: [
    { role: "user", content: "hi" },
    { role: "assistant", content: "hello" },
  ],
}

describe("derivePlaygroundGate", () => {
  const ready = {
    isLoading: false,
    isCatalogError: false,
    hasWorkspace: true,
    modelCount: 3,
  }

  it("is ready when a workspace has models", () => {
    expect(derivePlaygroundGate(ready)).toBe("ready")
  })

  it("reports loading over everything else", () => {
    // Order matters: every answer below would be reported from an empty list.
    expect(
      derivePlaygroundGate({
        isLoading: true,
        isCatalogError: true,
        hasWorkspace: false,
        modelCount: 0,
      }),
    ).toBe("loading")
  })

  it("separates a failed catalog read from an empty catalog", () => {
    // The two send somebody to different places, and "add a provider" is the
    // wrong advice for a fetch that failed.
    expect(
      derivePlaygroundGate({ ...ready, isCatalogError: true, modelCount: 0 }),
    ).toBe("catalogError")
  })

  it("reports no workspace before no models", () => {
    // The catalog is workspace-scoped, so with no workspace a count of zero
    // means nothing was asked rather than nothing exists.
    expect(
      derivePlaygroundGate({ ...ready, hasWorkspace: false, modelCount: 0 }),
    ).toBe("noWorkspace")
  })

  it("reports no models when the catalog is genuinely empty", () => {
    expect(derivePlaygroundGate({ ...ready, modelCount: 0 })).toBe("noModels")
  })
})

describe("isAnyPanelBusy", () => {
  it("is false when both panels are idle", () => {
    expect(isAnyPanelBusy(EMPTY_PANEL, EMPTY_PANEL)).toBe(false)
  })

  it("is true while either is waiting for a first token", () => {
    expect(
      isAnyPanelBusy(
        { ...EMPTY_PANEL, isAwaitingFirstToken: true },
        EMPTY_PANEL,
      ),
    ).toBe(true)
  })

  it("is true while either is still streaming", () => {
    expect(
      isAnyPanelBusy(EMPTY_PANEL, { ...EMPTY_PANEL, isStreaming: true }),
    ).toBe(true)
  })
})

describe("haveBothPanelsAnswered", () => {
  it("is true when comparing and both ended on an answer", () => {
    expect(haveBothPanelsAnswered(true, answered, answered)).toBe(true)
  })

  it("is false in single view, however the panels look", () => {
    expect(haveBothPanelsAnswered(false, answered, answered)).toBe(false)
  })

  it("is false while either is still streaming", () => {
    expect(
      haveBothPanelsAnswered(true, answered, {
        ...answered,
        isStreaming: true,
      }),
    ).toBe(false)
  })

  it("is false once a new question is asked", () => {
    // Which is what makes the rating bar per exchange without a flag: the last
    // turn stops being an answer the moment somebody asks again.
    const asked: PanelState = {
      ...answered,
      turns: [...answered.turns, { role: "user", content: "again" }],
    }
    expect(haveBothPanelsAnswered(true, asked, asked)).toBe(false)
  })

  it("is false when one panel has not answered at all", () => {
    expect(haveBothPanelsAnswered(true, answered, EMPTY_PANEL)).toBe(false)
  })
})

describe("shouldShowWelcome", () => {
  it("shows the greeting before the first question", () => {
    expect(shouldShowWelcome(false, EMPTY_PANEL)).toBe(true)
  })

  it("hides it once a conversation starts", () => {
    expect(shouldShowWelcome(false, answered)).toBe(false)
  })

  it("hides it as soon as a question is in flight", () => {
    expect(
      shouldShowWelcome(false, { ...EMPTY_PANEL, isAwaitingFirstToken: true }),
    ).toBe(false)
  })

  it("never shows it while comparing", () => {
    expect(shouldShowWelcome(true, EMPTY_PANEL)).toBe(false)
  })
})
