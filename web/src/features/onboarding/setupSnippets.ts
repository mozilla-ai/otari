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

/** The agent prompt names the environment variable; the other tabs carry its value. */
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
