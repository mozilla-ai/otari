import { expect, it } from "vitest"
import { welcomeGuideLink } from "./docs"

it("sends the guide to the configured documentation, in a new tab", () => {
  expect(welcomeGuideLink({ docs_url: "https://docs.example.com/" })).toEqual({
    href: "https://docs.example.com/",
    label: "Read the documentation",
    target: "_blank",
    rel: "noopener noreferrer",
  })
})

it("keeps the gateway's own welcome page where no documentation is configured", () => {
  expect(welcomeGuideLink({ docs_url: null })).toEqual({
    href: "/welcome",
    label: "Open the welcome guide",
  })
})
