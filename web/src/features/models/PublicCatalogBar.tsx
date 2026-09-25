import { Button, buttonVariants, Drawer } from "@heroui/react"
import { useState } from "react"
import { FaGithub } from "react-icons/fa"
import { FiMenu } from "react-icons/fi"
import {
  publicCatalogHref,
  siteHomeHref,
} from "@/features/models/publicCatalog"
import { docsSourceHref } from "@/shared/helpers/docs"
import { useDeployment } from "@/shared/hooks/useDeployment"

const GITHUB_URL = "https://github.com/mozilla-ai/otari"

// Desktop only, and no display of its own: an `inline-flex` here is emitted
// after `hidden` and would show these at phone width, where the menu has them.
const LINK =
  "hidden min-h-10 items-center px-3 text-sm font-medium text-muted transition-colors hover:text-foreground md:inline-flex"
const DRAWER_LINK =
  "flex min-h-11 items-center gap-2 px-2 text-sm text-foreground transition-colors hover:bg-surface-alt"

/**
 * The public catalog's bar: the same destinations as the site in front of it
 * (Models, Documentation, Log in, Sign up, GitHub), so a visitor who came from
 * that site's navbar finds the same one here. The logo goes back to the site
 * where the deployment names one (`site_url`), else to the catalog.
 *
 * Documentation opens `docs_url` where it is set; otherwise the docs on GitHub,
 * because the bundled guide at `/#/docs` sits behind the sign-in screen.
 */
export function PublicCatalogBar({ onList }: { onList: boolean }) {
  const deployment = useDeployment()
  const { docs_url, open_signup } = deployment
  const [menuOpen, setMenuOpen] = useState(false)
  const docsHref = docs_url ?? docsSourceHref("index.md")

  return (
    <header className="flex h-14 shrink-0 items-center justify-between border-b border-border px-4 md:px-6">
      <a
        href={siteHomeHref(deployment) || publicCatalogHref()}
        aria-label="Otari home"
        className="inline-flex min-h-11 items-center"
      >
        <img
          src={`${import.meta.env.BASE_URL}favicon.svg`}
          alt=""
          className="h-6 w-[1.625rem]"
        />
      </a>
      <nav aria-label="Site" className="flex items-center gap-1 md:gap-2">
        <a
          href={publicCatalogHref()}
          aria-current={onList ? "page" : undefined}
          className={LINK}
        >
          Models
        </a>
        <a
          href={docsHref}
          target="_blank"
          rel="noopener noreferrer"
          className={LINK}
        >
          Documentation
        </a>
        <a href="#/" className={LINK}>
          Log in
        </a>
        {open_signup ? (
          <a
            href="#/signup"
            className={buttonVariants({ size: "sm", variant: "primary" })}
          >
            Sign up
          </a>
        ) : null}
        <a
          href={GITHUB_URL}
          target="_blank"
          rel="noopener noreferrer"
          aria-label="GitHub"
          className={LINK}
        >
          <FaGithub aria-hidden className="size-[1.125rem]" />
        </a>
        <Drawer isOpen={menuOpen} onOpenChange={setMenuOpen}>
          <Drawer.Trigger
            aria-label="Open menu"
            className={`${buttonVariants({ variant: "ghost", isIconOnly: true })} size-11 md:hidden`}
          >
            <FiMenu aria-hidden className="size-5" />
          </Drawer.Trigger>
          <Drawer.Backdrop className="bg-backdrop/30">
            <Drawer.Content placement="right">
              <Drawer.Dialog
                aria-label="Menu"
                className="flex h-full w-72 max-w-[85vw] flex-col"
              >
                <Drawer.Header className="border-b border-border px-4 py-3">
                  <div className="flex items-center justify-between gap-3">
                    <Drawer.Heading className="text-title">Menu</Drawer.Heading>
                    <Button
                      size="sm"
                      variant="ghost"
                      onPress={() => setMenuOpen(false)}
                    >
                      Close
                    </Button>
                  </div>
                </Drawer.Header>
                <Drawer.Body className="flex flex-col gap-1 px-2 py-3">
                  <a
                    href={publicCatalogHref()}
                    aria-current={onList ? "page" : undefined}
                    onClick={() => setMenuOpen(false)}
                    className={DRAWER_LINK}
                  >
                    Models
                  </a>
                  <a
                    href={docsHref}
                    target="_blank"
                    rel="noopener noreferrer"
                    className={DRAWER_LINK}
                  >
                    Documentation
                  </a>
                  <a
                    href={GITHUB_URL}
                    target="_blank"
                    rel="noopener noreferrer"
                    className={DRAWER_LINK}
                  >
                    <FaGithub aria-hidden className="size-4" />
                    GitHub
                  </a>
                  <hr className="my-2 border-border" />
                  <a href="#/" className={DRAWER_LINK}>
                    Log in
                  </a>
                </Drawer.Body>
              </Drawer.Dialog>
            </Drawer.Content>
          </Drawer.Backdrop>
        </Drawer>
      </nav>
    </header>
  )
}
