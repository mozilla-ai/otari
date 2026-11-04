import { Link } from "@tanstack/react-router"

import type { Explanation } from "./explain"

// What a picked span means, ahead of its raw input and output.
export function ExplainCard({ explanation }: { explanation: Explanation }) {
  return (
    <section className="flex flex-col gap-2">
      <p className="text-body">{explanation.sentence}</p>
      {explanation.facts.length > 0 ? (
        <dl className="flex flex-col gap-2">
          {explanation.facts.map(([term, value, link]) => (
            <div
              key={term}
              className="flex flex-col gap-0.5 sm:flex-row sm:gap-3"
            >
              <dt className="w-32 shrink-0 text-caption text-subtle">{term}</dt>
              <dd className="min-w-0 break-words text-body">
                {link ? (
                  <Link
                    to="/tools/mcp-servers/$serverId"
                    params={{ serverId: link.serverId }}
                    search={{ key: link.key, tool: link.tool }}
                    className="text-link hover:text-link-hover"
                  >
                    {value}
                  </Link>
                ) : (
                  value
                )}
              </dd>
            </div>
          ))}
        </dl>
      ) : null}
    </section>
  )
}
