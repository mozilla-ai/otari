# GitBook documentation branch

The `gitbook-docs` branch holds the **generated** GitBook site for the Otari documentation. GitBook syncs from it.

**Do not edit this branch by hand.** Each publish replaces its whole contents.

## How it is built

1. Publishing an Otari release (`vX.Y.Z`) starts `.github/workflows/otari-docs.yml` at that tag.
2. `scripts/prepare_gitbook_site.py` builds the site from `docs/` at that tag. It publishes the pages that `docs/SUMMARY.md` lists, and turns links to any other file into GitHub links at the tag. It also copies the OpenAPI spec to `api/openapi.json` and generates a page per API tag under `api-endpoints/`. GitBook renders those pages from the spec registered in it as `otari-openapi-spec`, which reads `api/openapi.json` on this branch.
3. The workflow commits the built site to this branch.

To change a page, open a pull request against `main` that edits it under `docs/`. The change reaches the site with the next release after it merges. To publish an existing release again, run the workflow by hand with that release's tag.
