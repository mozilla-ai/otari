import { Markdown } from "@/design-system/content/Markdown"
import { Disclosure } from "@/design-system/navigation/Disclosure"

/** Only renders reasoning returned by the provider. */
export function ThinkingBlock({ content }: { content: string }) {
  return (
    <Disclosure heading="Reasoning">
      <Markdown>{content}</Markdown>
    </Disclosure>
  )
}
