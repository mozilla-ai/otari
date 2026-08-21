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

  it("escapes a model id that would otherwise break the JSON body", () => {
    // A model id is whatever the provider's catalog says it is. Quotes,
    // backslashes and newlines are the three that would leave a snippet the
    // guide advertises as runnable unable to parse.
    const hostile = 'weird"model\\name\nv2'
    const snippet = buildCurlSnippet({ ...INPUT, model: hostile })
    const payload = snippet.slice(
      snippet.indexOf("{"),
      snippet.lastIndexOf("}") + 1,
    )

    expect(JSON.parse(payload)).toEqual({
      model: hostile,
      messages: [{ role: "user", content: "Hello" }],
    })
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

  it("escapes a model id that would otherwise break the Python literal", () => {
    // A JSON string literal is a valid Python one, which is what lets one
    // escaping helper serve both snippets.
    const snippet = buildPythonSnippet({ ...INPUT, model: 'weird"model\\name' })

    expect(snippet).toContain('model="weird\\"model\\\\name"')
  })
})
