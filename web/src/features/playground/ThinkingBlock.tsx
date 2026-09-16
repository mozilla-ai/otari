import { Markdown } from "@/design-system/content/Markdown"
import { Disclosure } from "@/design-system/navigation/Disclosure"

/** Provider reasoning is a titled section in the flat transcript, matching Disclosure. */
export function ThinkingBlock({ content }: { content: string }) {
  return (
    <Disclosure heading="Reasoning">
      <Markdown>{content}</Markdown>
    </Disclosure>
  )
}
