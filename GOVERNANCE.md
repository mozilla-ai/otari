# Governance

This document describes how the Otari open-source project is run, who makes
decisions, and how to get involved. It covers the open-source gateway in this
repository, not the hosted otari.ai service.

## Overview

Otari is an open-source project stewarded by [Mozilla.ai](https://mozilla.ai).
Development happens in the open on GitHub. Anyone can use Otari, file issues,
open pull requests, and take part in discussions.

## Roles

**Users** run Otari and report bugs, request features, and ask questions through
issues and Discussions.

**Contributors** open pull requests. Contributions of any size are welcome, from
documentation fixes to new features. See [CONTRIBUTING.md](CONTRIBUTING.md) to
get started.

**Maintainers** review and merge pull requests, triage issues, cut releases, and
set technical direction. Maintainers are the people with merge access to the
repository. They are responsible for keeping the project healthy and for
upholding the [Code of Conduct](CODE_OF_CONDUCT.md).

## How decisions are made

Most decisions happen in the open through issues and pull requests, by maintainer
consensus. Routine changes are merged once a maintainer approves and CI passes.

Larger changes, such as new public APIs, breaking changes, or shifts in scope,
are discussed in an issue or Discussion first so the reasoning is on the record
and the community can weigh in. When maintainers disagree and consensus isn't
reached, the decision escalates to the Mozilla.ai team that stewards the project.

## Becoming a maintainer

Maintainers are invited from the contributor community based on a sustained track
record: quality contributions over time, helpful review and triage, and good
judgment in discussions. There is no fixed quota. If you want to grow into the
role, the path is to keep contributing and engaging; existing maintainers will
reach out.

## Relationship to Mozilla.ai and otari.ai

Mozilla.ai develops both the open-source Otari gateway and the hosted otari.ai
platform. The two share concepts and a wire protocol, but the gateway in this
repository is and will remain open source under its existing license. It is a
standalone product you can run with no dependency on the hosted service and no
account required.

We develop the gateway in the open, keep its roadmap public, and accept outside
contributions on equal terms. The open-source project is not a funnel for the
hosted product.

## Changing this document

Changes to governance are proposed through a pull request and require maintainer
consensus to merge.
