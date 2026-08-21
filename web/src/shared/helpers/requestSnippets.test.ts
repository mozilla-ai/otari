import { describe, expect, it } from "vitest"

import {
  buildCurlSnippet,
  buildPythonSnippet,
  SNIPPET_MODEL_PLACEHOLDER,
} from "@/shared/helpers/requestSnippets"

const INPUT = {
  origin: "https://otari.example.com",
  apiKey: "gw-abc123",
  model: "openai:gpt-4o-mini",
}

describe("buildCurlSnippet", () => {
  it("posts to the completions path on the origin the dashboard was served from", () => {
    expect(buildCurlSnippet(INPUT)).toContain(
      "https://otari.example.com/v1/chat/completions",
    )
  })

  it("authenticates with the header the gateway names itself", () => {
    expect(buildCurlSnippet(INPUT)).toContain('-H "Otari-Key: gw-abc123"')
  })

  it("names the model it was given", () => {
    expect(buildCurlSnippet(INPUT)).toContain('"model": "openai:gpt-4o-mini"')
  })

  it("falls back to the placeholder when the deployment has no model to name", () => {
    const snippet = buildCurlSnippet({ ...INPUT, model: undefined })
    expect(snippet).toContain(`"model": "${SNIPPET_MODEL_PLACEHOLDER}"`)
  })
})

describe("buildPythonSnippet", () => {
  it("points the OpenAI SDK at the gateway", () => {
    expect(buildPythonSnippet(INPUT)).toContain(
      'client = OpenAI(base_url="https://otari.example.com/v1", api_key="gw-abc123")',
    )
  })

  it("carries the model and the message into the call", () => {
    const snippet = buildPythonSnippet({
      ...INPUT,
      message: "Hello from Otari",
    })
    expect(snippet).toContain('model="openai:gpt-4o-mini"')
    expect(snippet).toContain('"content": "Hello from Otari"')
  })
})
