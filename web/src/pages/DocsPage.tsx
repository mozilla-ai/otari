import { Card } from "@heroui/react";
import type { AnchorHTMLAttributes } from "react";
import ReactMarkdown from "react-markdown";
import type { Components } from "react-markdown";
import remarkGfm from "remark-gfm";

import { PageHeader } from "@/components/ui";
// The operator user guide is bundled straight from the repo's docs so the
// running dashboard ships the guide that matches it, instead of pointing at a
// docs site that may describe a different version. Rebuilding the dashboard
// after editing the guide is what keeps them aligned (see AGENTS.md).
import dashboardGuide from "../../../docs/dashboard.md?raw";

// The bundled guide lives among sibling docs (configuration.md, quickstart.md,
// ...) that are not bundled here, so its relative links cannot resolve inside
// the SPA. Point them at the rendered source on GitHub instead; absolute,
// anchor, and mail links are left untouched.
const DOCS_SOURCE_BASE = "https://github.com/mozilla-ai/otari/blob/main/docs/";

function resolveDocHref(href: string | undefined): string | undefined {
  if (!href) return href;
  if (/^[a-z]+:/i.test(href) || href.startsWith("#") || href.startsWith("//")) {
    return href;
  }
  return DOCS_SOURCE_BASE + href.replace(/^\.\//, "");
}

function isExternal(href: string | undefined): boolean {
  return !!href && (/^https?:\/\//i.test(href) || href.startsWith("//"));
}

// The guide opens with its own top-level title ("# Admin dashboard"), which
// would duplicate the page header below. Drop that leading H1 so the page shows
// a single title; the guide's intro paragraph then flows straight under it.
const guideBody = dashboardGuide.replace(/^#[^\n]*\n+/, "");

const markdownComponents: Components = {
  a: ({ href, children, ...props }: AnchorHTMLAttributes<HTMLAnchorElement>) => {
    const resolved = resolveDocHref(href);
    // In-page anchors stay in the tab; every other link (bundled-out doc pages
    // rewritten to GitHub, or already-external URLs) opens in a new tab so the
    // operator does not lose the dashboard.
    const external = href?.startsWith("#") ? false : isExternal(resolved);
    return (
      <a href={resolved} {...(external ? { target: "_blank", rel: "noreferrer" } : {})} {...props}>
        {children}
      </a>
    );
  },
};

export function DocsPage() {
  return (
    <div className="flex flex-col gap-6">
      <PageHeader
        title="User guide"
        description="The operator guide for this dashboard, bundled with and version-matched to the running gateway."
      />
      <Card>
        <Card.Content className="p-5 sm:p-6">
          <div className="otari-markdown">
            <ReactMarkdown remarkPlugins={[remarkGfm]} components={markdownComponents}>
              {guideBody}
            </ReactMarkdown>
          </div>
        </Card.Content>
      </Card>
    </div>
  );
}
