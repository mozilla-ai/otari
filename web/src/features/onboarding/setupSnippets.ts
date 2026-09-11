import {
  buildAgentPrompt,
  buildCurlSnippet,
  buildPythonSnippet,
  buildTypescriptSnippet,
  type RequestSnippetInput,
} from "@/shared/helpers/requestSnippets"

/** The four ways the guide offers to make the first request. */
export type SetupSnippetId = "agent" | "curl" | "python" | "typescript"

export const SETUP_TABS: ReadonlyArray<{
  id: SetupSnippetId
  label: string
  /** The line above the block, which names what to do with it. */
  instruction: string
}> = [
  {
    id: "agent",
    label: "Agent",
    instruction: "Paste this into your coding agent",
  },
  { id: "curl", label: "cURL", instruction: "Run this in your terminal" },
  { id: "python", label: "Python", instruction: "Run this Python example" },
  {
    id: "typescript",
    label: "TypeScript",
    instruction: "Run this TypeScript example",
  },
]

/**
 * The agent tab is the default, and the order above is not alphabetical by
 * accident: an operator setting up a gateway in 2026 is usually about to ask an
 * agent to make the call, and the three runnable ones are there for when they
 * would rather do it themselves.
 */
export const DEFAULT_SETUP_TAB: SetupSnippetId = "agent"

/** Which tabs carry the key in their text, and so have something to conceal. */
export function carriesKey(id: SetupSnippetId): boolean {
  return id !== "agent"
}

/**
 * Every tab's snippet, built once from one input.
 *
 * The agent prompt takes no key at all (it names the environment variable
 * instead), so it is built from the rest of the input and is the same string
 * whether the key is revealed or concealed. That is what `carriesKey` above is
 * for: a caller concealing the key must not offer a reveal toggle on the one
 * tab where there is nothing hidden.
 */
export function buildSetupSnippets(
  input: RequestSnippetInput,
): Record<SetupSnippetId, string> {
  return {
    agent: buildAgentPrompt(input),
    curl: buildCurlSnippet(input),
    python: buildPythonSnippet(input),
    typescript: buildTypescriptSnippet(input),
  }
}
