import { render, screen } from "@testing-library/react"
import { describe, expect, it } from "vitest"

import { DocstringText, firstSentence } from "@/features/tools/DocstringText"

describe("DocstringText", () => {
  it("renders a reStructuredText literal as code", () => {
    render(<DocstringText>{"Defaults to ``all_pass``."}</DocstringText>)

    expect(screen.getByText("all_pass").tagName).toBe("CODE")
    expect(screen.getByText(/Defaults to/)).toBeInTheDocument()
    expect(screen.queryByText(/``/)).not.toBeInTheDocument()
  })

  it("keeps the text around several literals in order", () => {
    const { container } = render(
      <DocstringText>
        {'Pass ``{"a": 1}`` or ``{"b": 2}`` to tune it.'}
      </DocstringText>,
    )

    expect(container.textContent).toBe('Pass {"a": 1} or {"b": 2} to tune it.')
    expect(container.querySelectorAll("code")).toHaveLength(2)
  })

  it("leaves text with no markup alone", () => {
    const { container } = render(
      <DocstringText>List of blocklist names.</DocstringText>,
    )

    expect(container.textContent).toBe("List of blocklist names.")
    expect(container.querySelector("code")).toBeNull()
  })

  it("leaves an unclosed pair as it was typed rather than eating the rest", () => {
    const { container } = render(
      <DocstringText>{"Set ``threshold to tune it."}</DocstringText>,
    )

    expect(container.textContent).toBe("Set ``threshold to tune it.")
  })

  it("renders nothing for a parameter with no help", () => {
    const { container } = render(<DocstringText>{undefined}</DocstringText>)

    expect(container.textContent).toBe("")
  })
})

describe("firstSentence", () => {
  it("keeps the sentence that says what the argument is", () => {
    expect(
      firstSentence(
        "Alinia API key. If ``None``, it is read from the ``ALINIA_API_KEY`` environment variable.",
      ),
    ).toBe("Alinia API key.")
  })

  it("does not break on a period inside a literal", () => {
    // `{"safety": {"toxicity": 0.8}}` has a period in it, and splitting there
    // would end the sentence mid-example.
    expect(
      firstSentence(
        'Which detections to run and their thresholds such as ``{"toxicity": 0.8}`` here. Then more.',
      ),
    ).toBe(
      'Which detections to run and their thresholds such as ``{"toxicity": 0.8}`` here.',
    )
  })

  it("leaves a one-sentence description alone", () => {
    expect(firstSentence("List of blocklist names.")).toBe(
      "List of blocklist names.",
    )
  })
})

describe("how much DocstringText renders", () => {
  it("shows one sentence by default", () => {
    const { container } = render(
      <DocstringText>
        {"The key. It is read from the environment when absent."}
      </DocstringText>,
    )

    expect(container.textContent).toBe("The key.")
  })

  it("shows every sentence when the place has room", () => {
    const { container } = render(
      <DocstringText full>
        {"The key. It is read from the environment when absent."}
      </DocstringText>,
    )

    expect(container.textContent).toBe(
      "The key. It is read from the environment when absent.",
    )
  })
})
