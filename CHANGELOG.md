# Changelog

All notable changes to Otari will be documented in this file.

The format follows [Conventional Commits](https://www.conventionalcommits.org/).

## [0.12.1](https://github.com/mozilla-ai/otari/releases/tag/v0.12.1) - 2026-09-25



### Features

- **guardrails:** Make a gate's message optional in [#1734](https://github.com/mozilla-ai/otari/pull/1734) by [@agpituk](https://github.com/agpituk) ([`98c44a9`](https://github.com/mozilla-ai/otari/commit/98c44a95a4f1eb114d0c1f496b0b8737f3599312))


**Full Changelog**: https://github.com/mozilla-ai/otari/compare/v0.12.0...v0.12.1
## [0.12.0](https://github.com/mozilla-ai/otari/releases/tag/v0.12.0) - 2026-09-25



### Bug Fixes

- **web:** Align activity pinned columns when no selection column renders in [#1633](https://github.com/mozilla-ai/otari/pull/1633) by [@daavoo](https://github.com/daavoo) ([`02cc6e9`](https://github.com/mozilla-ai/otari/commit/02cc6e9960c855fbad5e94b51f6d52908b32f8e8))
- **hybrid:** Name the last attempt tried on a total failure in [#1709](https://github.com/mozilla-ai/otari/pull/1709) by [@peteski22](https://github.com/peteski22) ([`2215e11`](https://github.com/mozilla-ai/otari/commit/2215e110ba74b70e201ccd54b134e0d03a2c9c0e))
- **hook:** Report every fail-open exit where the agent can see it in [#1721](https://github.com/mozilla-ai/otari/pull/1721) by [@dpoulopoulos](https://github.com/dpoulopoulos) ([`33f7cb2`](https://github.com/mozilla-ai/otari/commit/33f7cb2cc23d053a19cc495e1bda50531a80096e))
- **cli:** Make the agent-side CLI explain itself on a first run in [#1722](https://github.com/mozilla-ai/otari/pull/1722) by [@agpituk](https://github.com/agpituk) ([`7534fde`](https://github.com/mozilla-ai/otari/commit/7534fde9c6c663adc6b20f0e8f2b6f7077addcd8))
- **mcp:** Refuse an unknown server id alike in both modes in [#1706](https://github.com/mozilla-ai/otari/pull/1706) by [@peteski22](https://github.com/peteski22) ([`5e57c06`](https://github.com/mozilla-ai/otari/commit/5e57c063b60f64a564edcfe7cd7a0cd74276f3d8))
- **BREAKING:** **mcp:** Refuse a peer answer that omits the servers key in [#1716](https://github.com/mozilla-ai/otari/pull/1716) by [@peteski22](https://github.com/peteski22) ([`ec42446`](https://github.com/mozilla-ai/otari/commit/ec42446117f4ee9b184842e59e61954f7d753018))


### Features

- **BREAKING:** **policy-checks:** Say when every gate runs, and what it can see there in [#1677](https://github.com/mozilla-ai/otari/pull/1677) by [@agpituk](https://github.com/agpituk) ([`db4b73d`](https://github.com/mozilla-ai/otari/commit/db4b73da15600494d877c7d65a29ea9503c26aa9))
- **BREAKING:** **policy-checks:** Rename Agent Gates to Agent Guardrails in [#1685](https://github.com/mozilla-ai/otari/pull/1685) by [@agpituk](https://github.com/agpituk) ([`d7c771f`](https://github.com/mozilla-ai/otari/commit/d7c771f25e5f2588ed35f810f2c006de5f455024))
- **dashboard:** Link the sign-in header's logo back to the site in [#1686](https://github.com/mozilla-ai/otari/pull/1686) by [@jigjigjig](https://github.com/jigjigjig) ([`463fba5`](https://github.com/mozilla-ai/otari/commit/463fba50bcbee9683e464254859e99276b7d061b))
- **policy-checks:** Check a guardrail offline with `otari guardrails validate` in [#1693](https://github.com/mozilla-ai/otari/pull/1693) by [@agpituk](https://github.com/agpituk) ([`44c2ae4`](https://github.com/mozilla-ai/otari/commit/44c2ae40a0ec2ea4c6b0b160dd795f3a232dc23b))
- **BREAKING:** **policy-checks:** Refuse a read before the file enters the transcript in [#1696](https://github.com/mozilla-ai/otari/pull/1696) by [@agpituk](https://github.com/agpituk) ([`0c06ac4`](https://github.com/mozilla-ai/otari/commit/0c06ac414f2da9724fa20f0b45e27750a59acec6))
- **BREAKING:** **gateway:** In-product feedback is on by default on upgrade; feedback_enabled: false turns it off in [#1676](https://github.com/mozilla-ai/otari/pull/1676) by [@jigjigjig](https://github.com/jigjigjig) ([`55a9336`](https://github.com/mozilla-ai/otari/commit/55a933674e771b2995dfa6e0fba950ca0ad30e52))
- **BREAKING:** **policy-checks:** Compose a guardrail from .otari/guardrails/ in [#1704](https://github.com/mozilla-ai/otari/pull/1704) by [@agpituk](https://github.com/agpituk) ([`a9ebcb4`](https://github.com/mozilla-ai/otari/commit/a9ebcb43694285599fc1c6b647016749cb7a0b6d))
- **BREAKING:** **api:** Rename Otari's X- prefixed response headers in [#1667](https://github.com/mozilla-ai/otari/pull/1667) by [@peteski22](https://github.com/peteski22) ([`7ce2279`](https://github.com/mozilla-ai/otari/commit/7ce227924bd6bf4c21bcd36bb0bff10e807115a8))
- **policy-checks:** Deliver a judge gate's finding to the agent, not only the person in [#1719](https://github.com/mozilla-ai/otari/pull/1719) by [@agpituk](https://github.com/agpituk) ([`981353c`](https://github.com/mozilla-ai/otari/commit/981353c696d4b1e786180814df7c3af9d18a6aad))


**Full Changelog**: https://github.com/mozilla-ai/otari/compare/v0.11.0...v0.12.0
## [0.11.0](https://github.com/mozilla-ai/otari/releases/tag/v0.11.0) - 2026-09-24



### Features

- **catalog:** Let a guest browse the hosted models and start an account from one in [#1652](https://github.com/mozilla-ai/otari/pull/1652) by [@jigjigjig](https://github.com/jigjigjig) ([`538f247`](https://github.com/mozilla-ai/otari/commit/538f24790d0fc787185ccbe24042e33749dbcae2))
- **dashboard:** Send feedback to the Otari team from the top bar in [#1655](https://github.com/mozilla-ai/otari/pull/1655) by [@jigjigjig](https://github.com/jigjigjig) ([`558209f`](https://github.com/mozilla-ai/otari/commit/558209ffe604b98f642ddd9f7e193f80b7dee55b))
- **dashboard:** Seams for a dashboard that reaches several deployments in [#1669](https://github.com/mozilla-ai/otari/pull/1669) by [@tbille](https://github.com/tbille) ([`cf5e1fa`](https://github.com/mozilla-ai/otari/commit/cf5e1fab56446f966f73c3c7fc4dea206773954a))


**Full Changelog**: https://github.com/mozilla-ai/otari/compare/v0.10.0...v0.11.0
## [0.10.0](https://github.com/mozilla-ai/otari/releases/tag/v0.10.0) - 2026-09-24



### Bug Fixes

- **messages:** Let container auto through the managed-credential gate in [#1618](https://github.com/mozilla-ai/otari/pull/1618) by [@hasangzl](https://github.com/hasangzl) ([`f9748a8`](https://github.com/mozilla-ai/otari/commit/f9748a8819e3fa9e2ea00b378beed5e17db1d9ad))
- **dashboard:** Seat the account band on the bottom edge of the rail in [#1641](https://github.com/mozilla-ai/otari/pull/1641) by [@jigjigjig](https://github.com/jigjigjig) ([`1cddb70`](https://github.com/mozilla-ai/otari/commit/1cddb707f3415ff727dfd2a8769d2d043a29f57c))


### Features

- **policy-checks:** Support Codex as a hook harness alongside Claude Code in [#1447](https://github.com/mozilla-ai/otari/pull/1447) by [@agpituk](https://github.com/agpituk) ([`178810f`](https://github.com/mozilla-ai/otari/commit/178810fd2d3b605f8cafcce9ad32e653f43eaeb5))
- **policy-checks:** Evaluate otari hook policies locally by default in [#1448](https://github.com/mozilla-ai/otari/pull/1448) by [@agpituk](https://github.com/agpituk) ([`377fc9e`](https://github.com/mozilla-ai/otari/commit/377fc9eb482d40bb1f49d5e61d43271611309774))
- **policy-checks:** Add otari gates generate and run hook gates concurrently in [#1542](https://github.com/mozilla-ai/otari/pull/1542) by [@agpituk](https://github.com/agpituk) ([`4e404b7`](https://github.com/mozilla-ai/otari/commit/4e404b7209af730cdc9b19c4f95e44a99ee56082))
- **guardrails:** Add an endpoint to test a mandate on its own guardrails service by [@dpoulopoulos](https://github.com/dpoulopoulos) ([`5137351`](https://github.com/mozilla-ai/otari/commit/5137351857a1463778d3eb5712bbdd0e0ba0eb80))
- **dashboard:** Add a test action to mandates on your own guardrails service by [@dpoulopoulos](https://github.com/dpoulopoulos) ([`bde5bf1`](https://github.com/mozilla-ai/otari/commit/bde5bf143a3750759ae8acf62fb2a4c312822369))
- **cli:** Ship the agent-side otari CLI as the otari-agent package in [#1581](https://github.com/mozilla-ai/otari/pull/1581) by [@dpoulopoulos](https://github.com/dpoulopoulos) ([`509314e`](https://github.com/mozilla-ai/otari/commit/509314e4e4f571da4d0f0388f178e6da1bceb20d))
- **cli:** Render the homebrew formula from the lock by [@dpoulopoulos](https://github.com/dpoulopoulos) ([`2445c09`](https://github.com/mozilla-ai/otari/commit/2445c09dad198e4846bb855eb7b92602c9b49580))
- **dashboard:** Show each MCP server's id with a copy button in [#1624](https://github.com/mozilla-ai/otari/pull/1624) by [@HareeshBahuleyan](https://github.com/HareeshBahuleyan) ([`f5f0d9c`](https://github.com/mozilla-ai/otari/commit/f5f0d9cb492dbb1f234c06ac45f84d98ef9167d1))


**Full Changelog**: https://github.com/mozilla-ai/otari/compare/v0.9.0...v0.10.0
## [0.9.0](https://github.com/mozilla-ai/otari/releases/tag/v0.9.0) - 2026-09-23



### Bug Fixes

- **budgets:** Stop a spend ceiling outliving a concurrently deleted workspace in [#1538](https://github.com/mozilla-ai/otari/pull/1538) by [@peteski22](https://github.com/peteski22) ([`dc45c0e`](https://github.com/mozilla-ai/otari/commit/dc45c0e7f37c937cb9095eabdf134663119490d8))
- **tools:** Read the code executor the same way on every input in [#1539](https://github.com/mozilla-ai/otari/pull/1539) by [@peteski22](https://github.com/peteski22) ([`9fc35f4`](https://github.com/mozilla-ai/otari/commit/9fc35f43ced667b889fd07ce4ffa2673c5022aa7))
- **guardrails:** List only the guardrails a stored row can build by [@dpoulopoulos](https://github.com/dpoulopoulos) ([`a7a90c9`](https://github.com/mozilla-ai/otari/commit/a7a90c93c9ad11ebf0b1cb13fe7fab00fcf56012))
- **guardrails:** Regenerate the dashboard client for the validate_kwargs text in [#1495](https://github.com/mozilla-ai/otari/pull/1495) by [@dpoulopoulos](https://github.com/dpoulopoulos) ([`de02ea3`](https://github.com/mozilla-ai/otari/commit/de02ea3a9fdd42605bbd6a07774abe2404d51391))
- **guardrails:** Retry a guardrail build that ran out of time by [@dpoulopoulos](https://github.com/dpoulopoulos) ([`181f8a8`](https://github.com/mozilla-ai/otari/commit/181f8a818961993c1832d2d3b850c72c853f8ac8))
- **dashboard:** Stop discard from submitting the form it discards in [#1546](https://github.com/mozilla-ai/otari/pull/1546) by [@dpoulopoulos](https://github.com/dpoulopoulos) ([`f574590`](https://github.com/mozilla-ai/otari/commit/f5745904d1abf5ae21184fce0eea18f901cadfee))
- **catalog:** Withhold a hosted model the deployment no longer advertises in [#1564](https://github.com/mozilla-ai/otari/pull/1564) by [@tbille](https://github.com/tbille) ([`3919387`](https://github.com/mozilla-ai/otari/commit/39193878d5cdfc63f608b367784e2d000091e476))


### Features

- **guardrails:** Store an organization's own guardrail definition by [@dpoulopoulos](https://github.com/dpoulopoulos) ([`fc397c1`](https://github.com/mozilla-ai/otari/commit/fc397c1ea4043c80af3479cd5b5ca9ac293fff7a))
- **invitations:** Hand admins a shareable accept link and let invitees set a password on accept in [#1562](https://github.com/mozilla-ai/otari/pull/1562) by [@tbille](https://github.com/tbille) ([`d19dd96`](https://github.com/mozilla-ai/otari/commit/d19dd9676b0a1f1c6ccfae6f1eb148f1e7dae0de))
- **guardrails:** Look up one built-in guardrail by name in [#1459](https://github.com/mozilla-ai/otari/pull/1459) by [@dpoulopoulos](https://github.com/dpoulopoulos) ([`336f9e4`](https://github.com/mozilla-ai/otari/commit/336f9e4e466eb52fb389be9d30af1dbb8baff87b))
- **guardrails:** Deny any_llm to an organization's own store by [@dpoulopoulos](https://github.com/dpoulopoulos) ([`8d355b2`](https://github.com/mozilla-ai/otari/commit/8d355b21909eb2c03d0e15d17df1eed321d80ed8))
- **guardrails:** Write an organization's own guardrail definition by [@dpoulopoulos](https://github.com/dpoulopoulos) ([`821e84e`](https://github.com/mozilla-ai/otari/commit/821e84e78b55666d3e58f772cac03568e3a636cc))
- **api:** Serve an organization's guardrail definitions by [@dpoulopoulos](https://github.com/dpoulopoulos) ([`1389549`](https://github.com/mozilla-ai/otari/commit/13895498fe5dcfb2f10dc21c36b93a1a830edee1))
- **guardrails:** Refuse dropping a definition a mandate names by [@dpoulopoulos](https://github.com/dpoulopoulos) ([`425ab36`](https://github.com/mozilla-ai/otari/commit/425ab3694a43acd1a1136ac17ef6eb9f18e4b592))
- **guardrails:** Let an organization mandate a guardrail it defined by [@dpoulopoulos](https://github.com/dpoulopoulos) ([`2e64713`](https://github.com/mozilla-ai/otari/commit/2e6471378367c898c00d7a87bdecad2d13064966))
- **guardrails:** Check a definition's stored endpoint before it is saved by [@dpoulopoulos](https://github.com/dpoulopoulos) ([`3bdb608`](https://github.com/mozilla-ai/otari/commit/3bdb6087b96d936c1899372f700c0ead5248c8e6))
- **guardrails:** Read every enabled definition in one query by [@dpoulopoulos](https://github.com/dpoulopoulos) ([`0fb9f7d`](https://github.com/mozilla-ai/otari/commit/0fb9f7dc149b164b6b565e90795e3c6152170781))
- **guardrails:** Build an organization's definitions and hold them ready by [@dpoulopoulos](https://github.com/dpoulopoulos) ([`5bdeacd`](https://github.com/mozilla-ai/otari/commit/5bdeacd3fa07f77492d5e78e62d29df3a5038cae))
- **gateway:** Prime the guardrail runner at boot and keep it current by [@dpoulopoulos](https://github.com/dpoulopoulos) ([`d83eee6`](https://github.com/mozilla-ai/otari/commit/d83eee681a93a4ccb862f8d9e8936de8921abad7))
- **guardrails:** Carry a mandate's definition id to the request path by [@dpoulopoulos](https://github.com/dpoulopoulos) ([`bef8cb7`](https://github.com/mozilla-ai/otari/commit/bef8cb761ff8cfd85a2b73d2fcc752d751bb342e))
- **guardrails:** Keep the definition that serves a profile through the merge by [@dpoulopoulos](https://github.com/dpoulopoulos) ([`eb3b0b9`](https://github.com/mozilla-ai/otari/commit/eb3b0b9e4cbfe891a3ba509acf30df8dc0d154dd))
- **guardrails:** Answer a check with the guardrail this worker holds by [@dpoulopoulos](https://github.com/dpoulopoulos) ([`dbeb131`](https://github.com/mozilla-ai/otari/commit/dbeb131814918b26b6c4aa17ad48130c91765814))
- **gateway:** Run a mandated definition on the request path by [@dpoulopoulos](https://github.com/dpoulopoulos) ([`4d8898d`](https://github.com/mozilla-ai/otari/commit/4d8898d282a6b59b506aa12e1de32d61cc315b7d))
- **guardrails:** Answer which version of a definition this worker holds by [@dpoulopoulos](https://github.com/dpoulopoulos) ([`a0428ec`](https://github.com/mozilla-ai/otari/commit/a0428ec0dc0d99b9aca399fdfcf74de547f8c578))
- **api:** Report whether a guardrail definition is actually running by [@dpoulopoulos](https://github.com/dpoulopoulos) ([`8d0ac86`](https://github.com/mozilla-ai/otari/commit/8d0ac868609d148d6a2fd326d4b2c277ef47bb59))
- **api:** Rebuild a guardrail definition when it is written by [@dpoulopoulos](https://github.com/dpoulopoulos) ([`0a2d924`](https://github.com/mozilla-ai/otari/commit/0a2d9249b19aac5d1b7cb771c395c0665d5bf250))
- **config:** Size the guardrail thread pool from the deployment by [@dpoulopoulos](https://github.com/dpoulopoulos) ([`60bc87f`](https://github.com/mozilla-ai/otari/commit/60bc87f6b7c64fb3f291f0ad2419f50188f9254a))
- **api:** Publish the organization guardrails surface by [@dpoulopoulos](https://github.com/dpoulopoulos) ([`37a2bc2`](https://github.com/mozilla-ai/otari/commit/37a2bc276fe108d59ba5dc9656b7ff6a9c91d356))
- **dashboard:** Add hooks for the built-in guardrail catalog and definitions by [@dpoulopoulos](https://github.com/dpoulopoulos) ([`0695b07`](https://github.com/mozilla-ai/otari/commit/0695b075f3e617524cd680491b36db6522a4d23f))
- **dashboard:** Read the checks and fields a built-in guardrail offers in [#1545](https://github.com/mozilla-ai/otari/pull/1545) by [@dpoulopoulos](https://github.com/dpoulopoulos) ([`11494e3`](https://github.com/mozilla-ai/otari/commit/11494e360d0c46c277cc339eb75f8a8532ef9ad7))
- **dashboard:** Keep a guardrail definition's secrets across an edit by [@dpoulopoulos](https://github.com/dpoulopoulos) ([`f5f1e3c`](https://github.com/mozilla-ai/otari/commit/f5f1e3c3783e126d0ad543359c19dec1f9a1d876))
- **dashboard:** Add the dialog that sets up a guardrail otari runs itself by [@dpoulopoulos](https://github.com/dpoulopoulos) ([`3b5bf44`](https://github.com/mozilla-ai/otari/commit/3b5bf448b5350f83f1169fb864b7b61aa01d028e))
- **dashboard:** Add the dialog that says where a guardrail runs by [@dpoulopoulos](https://github.com/dpoulopoulos) ([`2a86206`](https://github.com/mozilla-ai/otari/commit/2a86206b2d28c721b240d6cf8f9f2bdafb68afed))
- **dashboard:** Open the two guardrail dialogs from the organization card by [@dpoulopoulos](https://github.com/dpoulopoulos) ([`f6b59b5`](https://github.com/mozilla-ai/otari/commit/f6b59b521ed93e0cc92a2a6a7657018e299bd3cd))
- **dashboard:** Ask for a guardrail's JSON arguments as checkboxes by [@dpoulopoulos](https://github.com/dpoulopoulos) ([`22200c0`](https://github.com/mozilla-ai/otari/commit/22200c0c133ae2df42c672374da5bf06b8447185))
- **dashboard:** Call the definition action configure guardrail by [@dpoulopoulos](https://github.com/dpoulopoulos) ([`825f043`](https://github.com/mozilla-ai/otari/commit/825f043cd33cde4ee7fab0f4e98beca7abaa33e6))
- **dashboard:** Say when an organization guardrail is not running by [@dpoulopoulos](https://github.com/dpoulopoulos) ([`33bbbd2`](https://github.com/mozilla-ai/otari/commit/33bbbd29a14933921ec7ba4baf00c2e0c4888969))
- **dashboard:** Mark the card's mandates whose guardrail is not running in [#1547](https://github.com/mozilla-ai/otari/pull/1547) by [@dpoulopoulos](https://github.com/dpoulopoulos) ([`a23303e`](https://github.com/mozilla-ai/otari/commit/a23303e30662bcd354855b0656bb7fbd647bcc51))
- **dashboard:** Add the organization guardrails page by [@dpoulopoulos](https://github.com/dpoulopoulos) ([`fba88fc`](https://github.com/mozilla-ai/otari/commit/fba88fc94985c21a65c43810f0e79519972ce3c2))
- **dashboard:** Say configure guardrail on the organization page in [#1548](https://github.com/mozilla-ai/otari/pull/1548) by [@dpoulopoulos](https://github.com/dpoulopoulos) ([`be6f82d`](https://github.com/mozilla-ai/otari/commit/be6f82db59c1ba74329f01ce3ee62b60c5494cfa))
- **guardrails:** Test an organization's guardrail definition by [@dpoulopoulos](https://github.com/dpoulopoulos) ([`cc5d552`](https://github.com/mozilla-ai/otari/commit/cc5d552f6eb332c09ae0a08a2a776fc4b016cb41))
- **dashboard:** Test a configured guardrail from its row in [#1550](https://github.com/mozilla-ai/otari/pull/1550) by [@dpoulopoulos](https://github.com/dpoulopoulos) ([`233bada`](https://github.com/mozilla-ai/otari/commit/233bada3464d2081494fdc9da28de4d492a11aa0))
- **dashboard:** Show a workspace the guardrails that apply to it in [#1551](https://github.com/mozilla-ai/otari/pull/1551) by [@dpoulopoulos](https://github.com/dpoulopoulos) ([`bb18e93`](https://github.com/mozilla-ai/otari/commit/bb18e930bfa747428cb95b12a266ef3900086d31))
- Return request id and inline cost on standalone responses by [@daavoo](https://github.com/daavoo) ([`07f97db`](https://github.com/mozilla-ai/otari/commit/07f97dbed145757adb43809fd22808d7441a9d0c))


**Full Changelog**: https://github.com/mozilla-ai/otari/compare/v0.8.0...v0.9.0
## [0.8.0](https://github.com/mozilla-ai/otari/releases/tag/v0.8.0) - 2026-09-22



### Features

- **BREAKING:** **catalog:** Reach a model by vendor/model and pin a provider with provider:vendor/model in [#1521](https://github.com/mozilla-ai/otari/pull/1521) by [@tbille](https://github.com/tbille) ([`fc99acc`](https://github.com/mozilla-ai/otari/commit/fc99acc34bc37adcf7f34f24dc765ffa9940caae))


**Full Changelog**: https://github.com/mozilla-ai/otari/compare/v0.7.0...v0.8.0
## [0.7.0](https://github.com/mozilla-ai/otari/releases/tag/v0.7.0) - 2026-09-22



### Bug Fixes

- **tools:** Drop the X- prefix from the code-execution header in [#1496](https://github.com/mozilla-ai/otari/pull/1496) by [@peteski22](https://github.com/peteski22) ([`cc5c5ed`](https://github.com/mozilla-ai/otari/commit/cc5c5ed624add1c92e141255bd81a25e0e6d2f7d))
- Name this project's install command when an optional extra is missing in [#1515](https://github.com/mozilla-ai/otari/pull/1515) by [@peteski22](https://github.com/peteski22) ([`cc617f3`](https://github.com/mozilla-ai/otari/commit/cc617f3f699f1a1d08e108ef674d9161ba61030a))


### Features

- **catalog:** Search the model catalog and the roster on the server in [#1436](https://github.com/mozilla-ai/otari/pull/1436) by [@khaledosman](https://github.com/khaledosman) ([`cf2968c`](https://github.com/mozilla-ai/otari/commit/cf2968c16563a18a484790e1665cc0219e134e1a))
- **dashboard:** Show providers and model makers with their own marks in [#1427](https://github.com/mozilla-ai/otari/pull/1427) by [@jigjigjig](https://github.com/jigjigjig) ([`4a1015c`](https://github.com/mozilla-ai/otari/commit/4a1015c0c3f3419fcab57c5803e527c67ea37ab1))
- **tenancy:** Answer the members page from one roster row in [#1437](https://github.com/mozilla-ai/otari/pull/1437) by [@khaledosman](https://github.com/khaledosman) ([`b8befec`](https://github.com/mozilla-ai/otari/commit/b8befecb5d99bd46e66427c03fcf5ab3e0181e26))
- **policy-checks:** Let a repo author its own check_passed verifier ([`adaa323`](https://github.com/mozilla-ai/otari/commit/adaa32332f8f33e5a6cf177418ca249b469f7122))
- **policy-checks:** Add a second check_passed verifier, no-stranded-docblocks ([`0021347`](https://github.com/mozilla-ai/otari/commit/0021347048004b48e601d55b1814088790c2e921))
- **tools:** Add the code-execution vocabulary and its settings by [@daavoo](https://github.com/daavoo) ([`1fe96a6`](https://github.com/mozilla-ai/otari/commit/1fe96a6c833031c84df437a9b219b35842d3b2d1))
- **files:** Add the fsspec store and the shared file helpers by [@daavoo](https://github.com/daavoo) ([`a1a2013`](https://github.com/mozilla-ai/otari/commit/a1a20135480d7f261ffd85ed50581678c9e62c64))
- **files:** Serve both SDKs' Files APIs, with the schema and services behind them by [@daavoo](https://github.com/daavoo) ([`d0ca835`](https://github.com/mozilla-ai/otari/commit/d0ca83548b7a5122d3c2dff3c78a677dbfffd9a0))
- **tools:** Run provider-native code execution on the sandbox when the model has none by [@daavoo](https://github.com/daavoo) ([`ff9dfd8`](https://github.com/mozilla-ai/otari/commit/ff9dfd8618d68cb76b3b328172fad502684e97f2))
- **providers:** Offer, price and switch an organization's models on its own keys in [#1456](https://github.com/mozilla-ai/otari/pull/1456) by [@tbille](https://github.com/tbille) ([`1891972`](https://github.com/mozilla-ai/otari/commit/189197204d652f7fa7ceec1985b99755dcaabf9c))
- **files:** Serve Anthropic's GA Files API shape and refuse the files beta header in [#1509](https://github.com/mozilla-ai/otari/pull/1509) by [@peteski22](https://github.com/peteski22) ([`05c7192`](https://github.com/mozilla-ai/otari/commit/05c7192d0128f3ecb33be7b935cce13294c8f9d1))
- **sandbox:** Run code on a hosted provider, through a port rather than a second service by [@daavoo](https://github.com/daavoo) ([`d9cdc26`](https://github.com/mozilla-ai/otari/commit/d9cdc2638ce3298b2f437c451346f1ff85de63ee))
- **sandbox:** Resume a code-execution sandbox across requests by container id in [#1433](https://github.com/mozilla-ai/otari/pull/1433) by [@daavoo](https://github.com/daavoo) ([`5b5c7f9`](https://github.com/mozilla-ai/otari/commit/5b5c7f90410c7cd2ac90e1f1e69f046a15ba75f3))
- **compose:** Add SeaweedFS as a self-hosted object store, and test the fsspec file store against it in [#1514](https://github.com/mozilla-ai/otari/pull/1514) by [@peteski22](https://github.com/peteski22) ([`423624e`](https://github.com/mozilla-ai/otari/commit/423624e1172affba3356509ffe5fe0a61d934be6))
- **files:** Copy the files a provider's code produces into Otari's store in [#1517](https://github.com/mozilla-ai/otari/pull/1517) by [@peteski22](https://github.com/peteski22) ([`ef31842`](https://github.com/mozilla-ai/otari/commit/ef318425dfbf3dda39f024d4247aaee54803b219))


### Maintenance

- **BREAKING:** **tools:** The code executor defaults to auto on upgrade; code_execution_executor: provider keeps forwarding in [#1507](https://github.com/mozilla-ai/otari/pull/1507) by [@peteski22](https://github.com/peteski22) ([`6264380`](https://github.com/mozilla-ai/otari/commit/6264380e06f3f05c96be9e683351bce28907257e))
- **BREAKING:** **files:** GET /api/v1/files pages and the sweep reclaims files on upgrade; files_sweep_interval_sec: 0 disables the sweep in [#1508](https://github.com/mozilla-ai/otari/pull/1508) by [@peteski22](https://github.com/peteski22) ([`e424d08`](https://github.com/mozilla-ai/otari/commit/e424d081ee975fad14e0ce08bcde0a4ce8eb1792))


### Other

- Fixed wrong API URL in docker-compose.yml in [#1435](https://github.com/mozilla-ai/otari/pull/1435) by [@aittalam](https://github.com/aittalam) ([`ae17044`](https://github.com/mozilla-ai/otari/commit/ae170442b360e30311aabaed176b732231807c46))


**Full Changelog**: https://github.com/mozilla-ai/otari/compare/v0.6.5...v0.7.0
## [0.6.5](https://github.com/mozilla-ai/otari/releases/tag/v0.6.5) - 2026-09-21



### Bug Fixes

- **dashboard:** Keep an unsaved settings edit when the server value moves in [#1390](https://github.com/mozilla-ai/otari/pull/1390) by [@khaledosman](https://github.com/khaledosman) ([`b49638d`](https://github.com/mozilla-ai/otari/commit/b49638d39f21eeeb93d250ee8deddfc1c852137e))
- **dashboard:** Read the query cache as the external store it is in [#1397](https://github.com/mozilla-ai/otari/pull/1397) by [@khaledosman](https://github.com/khaledosman) ([`57f245c`](https://github.com/mozilla-ai/otari/commit/57f245ca6291fd5025de7fb39f834eb994b58ae5))
- **tenancy:** Stop a workspace delete leaving a joining member's ceiling behind in [#1395](https://github.com/mozilla-ai/otari/pull/1395) by [@peteski22](https://github.com/peteski22) ([`b330b37`](https://github.com/mozilla-ai/otari/commit/b330b379cf912782c46cafba86d480483e8ec71f))
- **dashboard:** Remove API key table copy action in [#1361](https://github.com/mozilla-ai/otari/pull/1361) by [@jigjigjig](https://github.com/jigjigjig) ([`610a6a5`](https://github.com/mozilla-ai/otari/commit/610a6a5ca6728b62a294b2b0c6d287ffc8e63bd8))
- **dashboard:** Move Providers into organization General settings in [#1305](https://github.com/mozilla-ai/otari/pull/1305) by [@jigjigjig](https://github.com/jigjigjig) ([`472de4b`](https://github.com/mozilla-ai/otari/commit/472de4b5b328434083c31a9e9a3610d06ab591fe))
- **dashboard:** Offer the Playground the models the catalog lists in [#1418](https://github.com/mozilla-ai/otari/pull/1418) by [@tbille](https://github.com/tbille) ([`13419be`](https://github.com/mozilla-ai/otari/commit/13419beb115a76ab24c1e57094f5c5f5c8b5b68b))
- **dashboard:** Page the organization rate overrides table in [#1424](https://github.com/mozilla-ai/otari/pull/1424) by [@khaledosman](https://github.com/khaledosman) ([`f919407`](https://github.com/mozilla-ai/otari/commit/f919407d33b4a6ac9bd30cb8048a9ebcd6a69be6))
- **usage:** Redact the upstream error before it is stored on the usage row in [#1426](https://github.com/mozilla-ai/otari/pull/1426) by [@daavoo](https://github.com/daavoo) ([`9e7531c`](https://github.com/mozilla-ai/otari/commit/9e7531c7f1b7675b736af3d79dae76047551ed92))
- **policy-checks:** Keep the judge gate bounded, out of its own hooks, and within the server's own budget ([`466774d`](https://github.com/mozilla-ai/otari/commit/466774d4fd52984d92f0917d2d096173cd989f48))
- **policy-checks:** Fail open only on truly unfixable judge evidence, never on a fixable gap ([`ac2a9ae`](https://github.com/mozilla-ai/otari/commit/ac2a9ae8731eb0a77ebae3c91e3b93237ec1cd2e))
- **policy-checks:** Guard the gates-file read so it fails open on non-UTF-8 or a race in [#1387](https://github.com/mozilla-ai/otari/pull/1387) ([`03c979a`](https://github.com/mozilla-ai/otari/commit/03c979ad5d717d646e2164eade4007db7a3951a9))
- **dashboard:** Page the organization spend ceilings table in [#1429](https://github.com/mozilla-ai/otari/pull/1429) by [@khaledosman](https://github.com/khaledosman) ([`e1561f2`](https://github.com/mozilla-ai/otari/commit/e1561f2bb13c9c280942d1134eeb00ec778ff62f))
- **hybrid:** Accept the standalone credential headers on the platform path in [#1423](https://github.com/mozilla-ai/otari/pull/1423) by [@tbille](https://github.com/tbille) ([`d1916bb`](https://github.com/mozilla-ai/otari/commit/d1916bb17453476e4332609c43b7fe6b324c3db3))


### Features

- **pricing:** Answer current model rates on their own paged route in [#1419](https://github.com/mozilla-ai/otari/pull/1419) by [@khaledosman](https://github.com/khaledosman) ([`2689f60`](https://github.com/mozilla-ai/otari/commit/2689f60a84880a5de015f1756bfb115498525a29))
- **overview:** Summarize the dashboard overview server-side in [#1428](https://github.com/mozilla-ai/otari/pull/1428) by [@khaledosman](https://github.com/khaledosman) ([`6513128`](https://github.com/mozilla-ai/otari/commit/65131288a2f4307481ecf2cd0154416b5b245794))
- **policy-checks:** Add a judge gate that evaluates a rubric via a local model call ([`011ed0b`](https://github.com/mozilla-ai/otari/commit/011ed0b137b93047703faa6404b1b11225d486a3))


**Full Changelog**: https://github.com/mozilla-ai/otari/compare/v0.6.4...v0.6.5
## [0.6.4](https://github.com/mozilla-ai/otari/releases/tag/v0.6.4) - 2026-09-18



### Bug Fixes

- Preserve retryable sandbox availability errors in [#1233](https://github.com/mozilla-ai/otari/pull/1233) by [@hasangzl](https://github.com/hasangzl) ([`25ad399`](https://github.com/mozilla-ai/otari/commit/25ad3998ed1045c9ce1087f8200bd7c01f69a8e2))
- **release:** Mark breaking changes in the release notes in [#1317](https://github.com/mozilla-ai/otari/pull/1317) by [@peteski22](https://github.com/peteski22) ([`3b9a8a7`](https://github.com/mozilla-ai/otari/commit/3b9a8a7984207cc34a2aa4be27f5d6ce86c340f3))
- **dashboard:** Offer a fresh verification link on the unverified sign-in refusal in [#1324](https://github.com/mozilla-ai/otari/pull/1324) by [@tbille](https://github.com/tbille) ([`c02933b`](https://github.com/mozilla-ai/otari/commit/c02933b7d71d0ff5bb26c9b1425bcb0cbe261421))
- **clipboard:** Restore cleanup when legacyCopy's select() throws in [#1186](https://github.com/mozilla-ai/otari/pull/1186) by [@AmirF194](https://github.com/AmirF194) ([`c2bd608`](https://github.com/mozilla-ai/otari/commit/c2bd608b5dc8c5e98227a7954fff63befe2f4bed))
- **dashboard:** Preserve add-provider dialog state across tab switches in [#1105](https://github.com/mozilla-ai/otari/pull/1105) by [@AloysJehwin](https://github.com/AloysJehwin) ([`854d9d7`](https://github.com/mozilla-ai/otari/commit/854d9d7cc3672aa126c5077002967654d59aa26a))
- **dashboard:** Keep an API base typed before the provider's hints land in [#1333](https://github.com/mozilla-ai/otari/pull/1333) by [@khaledosman](https://github.com/khaledosman) ([`2215cc4`](https://github.com/mozilla-ai/otari/commit/2215cc48584a2c5f87d3cfaf906068c0aaeaf43a))
- **dashboard:** Let the pricing review and share frames be dismissed in [#936](https://github.com/mozilla-ai/otari/pull/936) by [@mikemikimike](https://github.com/mikemikimike) ([`8d80407`](https://github.com/mozilla-ai/otari/commit/8d80407541db9448793b51ab92c4e9bcf492767a))
- **gateway:** Forward ttft_ms in hybrid-mode usage reports in [#1210](https://github.com/mozilla-ai/otari/pull/1210) by [@AmirF194](https://github.com/AmirF194) ([`0e3a38b`](https://github.com/mozilla-ai/otari/commit/0e3a38b37036ca9f6433781593990e1bb18c17f3))
- **dashboard:** Render every number and cost in one locale in [#1365](https://github.com/mozilla-ai/otari/pull/1365) by [@khaledosman](https://github.com/khaledosman) ([`62cbde3`](https://github.com/mozilla-ai/otari/commit/62cbde303949b362fb55ee414809d640b78f77c7))
- **dashboard:** Raise six icon-only buttons to the 44px touch floor in [#1369](https://github.com/mozilla-ai/otari/pull/1369) by [@khaledosman](https://github.com/khaledosman) ([`7ae245e`](https://github.com/mozilla-ai/otari/commit/7ae245e973540443aa7391fe9a01f7182916bcef))
- **dashboard:** Stop measuring layout mid-drag and leaving timers running in [#1370](https://github.com/mozilla-ai/otari/pull/1370) by [@khaledosman](https://github.com/khaledosman) ([`924a19f`](https://github.com/mozilla-ai/otari/commit/924a19fa390fc4371e12fcdc6939a1cb5bb8351b))
- **dashboard:** Use rem for confirmed arbitrary widths in [#1367](https://github.com/mozilla-ai/otari/pull/1367) by [@LejeuneA](https://github.com/LejeuneA) ([`152e189`](https://github.com/mozilla-ai/otari/commit/152e189128169c42c0c16c6a8249effda3a6dd09))
- **dashboard:** Let the reader's own font size decide the root in [#1372](https://github.com/mozilla-ai/otari/pull/1372) by [@khaledosman](https://github.com/khaledosman) ([`80dba5c`](https://github.com/mozilla-ai/otari/commit/80dba5c4f2d03067bf58c172e38ee26689f849c3))


### Features

- **web-fetch:** Part 3 activate managed Fetch in [#979](https://github.com/mozilla-ai/otari/pull/979) by [@HareeshBahuleyan](https://github.com/HareeshBahuleyan) ([`22f8d54`](https://github.com/mozilla-ai/otari/commit/22f8d543190e06e25a62e50afd08c2b9d2cf4cde))
- **welcome:** Redesign the welcome page in the dashboard's divided-surface language in [#1104](https://github.com/mozilla-ai/otari/pull/1104) by [@jigjigjig](https://github.com/jigjigjig) ([`cda3c41`](https://github.com/mozilla-ai/otari/commit/cda3c4174fcfd36919631ebc2ca2be64082bf697))
- **web-fetch:** Part 4 add dashboard operator surfaces in [#1017](https://github.com/mozilla-ai/otari/pull/1017) by [@HareeshBahuleyan](https://github.com/HareeshBahuleyan) ([`19f895e`](https://github.com/mozilla-ai/otari/commit/19f895eb5c63ec7061d81389b96a4fdcf030e772))
- **keys:** Put the API key format behind a port in [#1375](https://github.com/mozilla-ai/otari/pull/1375) by [@tbille](https://github.com/tbille) ([`b3f5f5f`](https://github.com/mozilla-ai/otari/commit/b3f5f5f5cd19d276c5b3be3ade6f9133f4305047))



### New Contributors

- [@LejeuneA](https://github.com/LejeuneA) made their first contribution in [#1367](https://github.com/mozilla-ai/otari/pull/1367)
- [@hasangzl](https://github.com/hasangzl) made their first contribution in [#1233](https://github.com/mozilla-ai/otari/pull/1233)

**Full Changelog**: https://github.com/mozilla-ai/otari/compare/v0.6.3...v0.6.4
## [0.6.3](https://github.com/mozilla-ai/otari/releases/tag/v0.6.3) - 2026-09-17



### Bug Fixes

- **dashboard:** Cap the whole organization with its own id, not the word in [#1263](https://github.com/mozilla-ai/otari/pull/1263) by [@khaledosman](https://github.com/khaledosman) ([`89cd358`](https://github.com/mozilla-ai/otari/commit/89cd35808882d03ecec58c7310db02c455ca0ccd))
- **policy-checks:** Cleanup fixes for 1229  - hooks in [#1257](https://github.com/mozilla-ai/otari/pull/1257) by [@agpituk](https://github.com/agpituk) ([`54ff172`](https://github.com/mozilla-ai/otari/commit/54ff1723201959530998e01c8e0542af1cee2b22))
- **catalog:** List hosted providers for members and flag their models as deployment-supplied in [#1261](https://github.com/mozilla-ai/otari/pull/1261) by [@tbille](https://github.com/tbille) ([`907cd93`](https://github.com/mozilla-ai/otari/commit/907cd93e784248d761952c43c990116ce7081971))
- **playground:** Index the dispatch lookup and mend an unreadable key in [#1297](https://github.com/mozilla-ai/otari/pull/1297) by [@khaledosman](https://github.com/khaledosman) ([`3193937`](https://github.com/mozilla-ai/otari/commit/31939379ee9f9b166c49de6640d38961b0d0632e))
- **dashboard:** Stop a switch of organization asking for the role it left in [#1302](https://github.com/mozilla-ai/otari/pull/1302) by [@khaledosman](https://github.com/khaledosman) ([`df70175`](https://github.com/mozilla-ai/otari/commit/df701759fa86446dcdead704b581a922e9f851ef))
- **dashboard:** Let the user guide use the whole page in [#1301](https://github.com/mozilla-ai/otari/pull/1301) by [@khaledosman](https://github.com/khaledosman) ([`af7cd7a`](https://github.com/mozilla-ai/otari/commit/af7cd7ac5bca653dea071be5b3e733c533c2d577))
- **migrations:** Keep autogenerate from dropping tables another chain owns in [#1293](https://github.com/mozilla-ai/otari/pull/1293) by [@peteski22](https://github.com/peteski22) ([`3818ce7`](https://github.com/mozilla-ai/otari/commit/3818ce7119929a4692d4d75283d1403cc362bc70))
- **policy-checks:** Submit no command evidence when aggregate bounds are exceeded by [@agpituk](https://github.com/agpituk) ([`d358f9f`](https://github.com/mozilla-ai/otari/commit/d358f9fd9536cc165dc96369d015e3832422715c))
- **policy-checks:** Scope command evidence, so a Stop gate cannot dead-end a session by [@daavoo](https://github.com/daavoo) ([`3d52624`](https://github.com/mozilla-ai/otari/commit/3d526240db4c288dcb31b522782af833b61d0d7c))
- **policy-checks:** Regenerate the dashboard client for the command_scope field in [#1278](https://github.com/mozilla-ai/otari/pull/1278) by [@daavoo](https://github.com/daavoo) ([`2fd3eb7`](https://github.com/mozilla-ai/otari/commit/2fd3eb7584ed9ef417cb2587add834f1e7d33ce4))


### Features

- **policy-checks:** Add otari hook setup, so registering the hook is not a manual JSON edit in [#1239](https://github.com/mozilla-ai/otari/pull/1239) by [@agpituk](https://github.com/agpituk) ([`f15d488`](https://github.com/mozilla-ai/otari/commit/f15d488f0a744fa9f284c87edfe114d820d11e83))
- **playground:** Serve the Playground on a hosted control plane in [#1284](https://github.com/mozilla-ai/otari/pull/1284) by [@khaledosman](https://github.com/khaledosman) ([`73ac9d9`](https://github.com/mozilla-ai/otari/commit/73ac9d95a8b8693567355d15776b9dcc903628be))
- **policy-checks:** Add command_if_changed, a gate correlating a changed path with a required command by [@agpituk](https://github.com/agpituk) ([`04582bc`](https://github.com/mozilla-ai/otari/commit/04582bca47c50b23af555901b40529ee3ac9d3a7))



### New Contributors

- [@otari-bot[bot]](https://github.com/otari-bot[bot]) made their first contribution in [#1314](https://github.com/mozilla-ai/otari/pull/1314)

**Full Changelog**: https://github.com/mozilla-ai/otari/compare/v0.6.2...v0.6.3
## [0.6.2](https://github.com/mozilla-ai/otari/releases/tag/v0.6.2) - 2026-09-16



### Bug Fixes

- **dashboard:** Poll the build id at the path the gateway serves it on in [#1140](https://github.com/mozilla-ai/otari/pull/1140) by [@khaledosman](https://github.com/khaledosman) ([`0b4fed9`](https://github.com/mozilla-ai/otari/commit/0b4fed9a2c42094e7390acfea98cd5c5d33f4cfc))
- **dashboard:** Let an OAuth or passkey sign-in set a first password in [#1214](https://github.com/mozilla-ai/otari/pull/1214) by [@khaledosman](https://github.com/khaledosman) ([`c90f287`](https://github.com/mozilla-ai/otari/commit/c90f287ab4ff35a47652870bf9f93fa01f348852))
- **provider-keys:** Report a credential the deployment cannot decrypt in [#1217](https://github.com/mozilla-ai/otari/pull/1217) by [@khaledosman](https://github.com/khaledosman) ([`8b5bd81`](https://github.com/mozilla-ai/otari/commit/8b5bd819f2be03a27e74ba2d0b05b341f404e39f))
- **dashboard:** Show the API key fingerprint in the activation guide in [#1166](https://github.com/mozilla-ai/otari/pull/1166) by [@jigjigjig](https://github.com/jigjigjig) ([`c2d7e2e`](https://github.com/mozilla-ai/otari/commit/c2d7e2ec78c1f93f5aeea65e775caf30a0492012))
- **dashboard:** Polish model catalog navigation and remember its view in [#1219](https://github.com/mozilla-ai/otari/pull/1219) by [@jigjigjig](https://github.com/jigjigjig) ([`b156aee`](https://github.com/mozilla-ai/otari/commit/b156aeeaa0153f6dbaae25325cd56bef9ba0681d))
- **dashboard:** Polish activation modal and simplify dismissal in [#1216](https://github.com/mozilla-ai/otari/pull/1216) by [@jigjigjig](https://github.com/jigjigjig) ([`df1240b`](https://github.com/mozilla-ai/otari/commit/df1240b5e669daf031a6a3925c7da6fab69a0f00))
- **dashboard:** Center auth forms and simplify secondary actions in [#1220](https://github.com/mozilla-ai/otari/pull/1220) by [@jigjigjig](https://github.com/jigjigjig) ([`326507d`](https://github.com/mozilla-ai/otari/commit/326507d21712b469faa436dce79ab80a465652e5))
- **playground:** Improve design of playground in [#1170](https://github.com/mozilla-ai/otari/pull/1170) by [@jigjigjig](https://github.com/jigjigjig) ([`217b6fa`](https://github.com/mozilla-ai/otari/commit/217b6fa2c72080ed421f670ec94bd555f1dbe0b0))
- **dashboard:** Restore auth popover and catalog CI in [#1232](https://github.com/mozilla-ai/otari/pull/1232) by [@jigjigjig](https://github.com/jigjigjig) ([`8a2ac08`](https://github.com/mozilla-ai/otari/commit/8a2ac08fff17fb51f81e1ebb57a420a843c7069a))
- **web:** Clarify API key actions and responsive layouts in [#1225](https://github.com/mozilla-ai/otari/pull/1225) by [@jigjigjig](https://github.com/jigjigjig) ([`df80e69`](https://github.com/mozilla-ai/otari/commit/df80e693cbb1bd0c4d73c396d0e50713bb941eca))
- **db:** Close the request session when a data-plane dependency is torn down in [#1237](https://github.com/mozilla-ai/otari/pull/1237) by [@daavoo](https://github.com/daavoo) ([`3c00bcb`](https://github.com/mozilla-ai/otari/commit/3c00bcb754846ccc167f1a6f1686fbd2fdbadb57))
- **dashboard:** Drop the welcome guide link on a hosted deployment in [#1250](https://github.com/mozilla-ai/otari/pull/1250) by [@khaledosman](https://github.com/khaledosman) ([`ab0bb65`](https://github.com/mozilla-ai/otari/commit/ab0bb654a34f26bbfefa8671a51fbf2917f7a1d6))
- **dashboard:** Offer one way to add an organization member in [#1249](https://github.com/mozilla-ai/otari/pull/1249) by [@khaledosman](https://github.com/khaledosman) ([`ad34441`](https://github.com/mozilla-ai/otari/commit/ad34441c37264f86eff13f585aea4fb360936131))
- **dashboard:** Stop the sign-up form flashing its background and swallowing the terms link in [#1251](https://github.com/mozilla-ai/otari/pull/1251) by [@khaledosman](https://github.com/khaledosman) ([`8d1ca97`](https://github.com/mozilla-ai/otari/commit/8d1ca977ec862a2d1e675f7e9d94acb5ee137298))
- **providers:** Let a tenant read the provider catalog in [#1215](https://github.com/mozilla-ai/otari/pull/1215) by [@khaledosman](https://github.com/khaledosman) ([`5ec79ea`](https://github.com/mozilla-ai/otari/commit/5ec79eacec4b79330f4d9e816fa97a8d0e3c5cef))
- **pricing:** Close hosted-credential gap in organization pricing overrides in [#1189](https://github.com/mozilla-ai/otari/pull/1189) by [@tbille](https://github.com/tbille) ([`15aca94`](https://github.com/mozilla-ai/otari/commit/15aca94f483ccbe8ece944c4f3c26108f8919526))
- **dashboard:** Claim a deployment for an operator who already has an address in [#1002](https://github.com/mozilla-ai/otari/pull/1002) by [@tbille](https://github.com/tbille) ([`5940130`](https://github.com/mozilla-ai/otari/commit/5940130cb6f73145436184c4de4974991239fd00))
- Escape "%" before handing a database URL to alembic's Config in [#1258](https://github.com/mozilla-ai/otari/pull/1258) by [@Sharlie89](https://github.com/Sharlie89) ([`5f17b1c`](https://github.com/mozilla-ai/otari/commit/5f17b1cad34da0e03593689ddad348c698ee7956))


### Features

- **web-fetch:** Part-1 add secure retrieval foundation in [#868](https://github.com/mozilla-ai/otari/pull/868) by [@HareeshBahuleyan](https://github.com/HareeshBahuleyan) ([`ea1a247`](https://github.com/mozilla-ai/otari/commit/ea1a24711dada4bf4240ee43e1e319dc1e30d17d))
- **guardrails:** List only the guardrails a hosted api reaches by [@dpoulopoulos](https://github.com/dpoulopoulos) ([`160bb3d`](https://github.com/mozilla-ai/otari/commit/160bb3dbca9a5d1fce1d50c47f2533fcd3b15493))
- **web-fetch:** Part 2 migrate Search extraction in [#959](https://github.com/mozilla-ai/otari/pull/959) by [@HareeshBahuleyan](https://github.com/HareeshBahuleyan) ([`60479f5`](https://github.com/mozilla-ai/otari/commit/60479f5c23d60421f4b613415dd1ed4c4daf3312))
- **agent-gates:** Add a Hook Server endpoint for repo-owned policy checks in [#1206](https://github.com/mozilla-ai/otari/pull/1206) by [@agpituk](https://github.com/agpituk) ([`c849ffb`](https://github.com/mozilla-ai/otari/commit/c849ffb8778f277c776b496346fcb726d831333f))
- **observability:** Export database connection pool stats on /metrics in [#1242](https://github.com/mozilla-ai/otari/pull/1242) by [@daavoo](https://github.com/daavoo) ([`f30d001`](https://github.com/mozilla-ai/otari/commit/f30d001eb16f6512a5b14f12f257a07b949f67dc))
- **policy-checks:** Add command_match, a second Agent Gates gate type in [#1229](https://github.com/mozilla-ai/otari/pull/1229) by [@agpituk](https://github.com/agpituk) ([`8863803`](https://github.com/mozilla-ai/otari/commit/88638034ab739eaf4888004bbed608727453549e))


### Security

- **auth:** Stop is_superuser from bypassing tenant role checks in [#1014](https://github.com/mozilla-ai/otari/pull/1014) by [@tbille](https://github.com/tbille) ([`09cb17d`](https://github.com/mozilla-ai/otari/commit/09cb17d8d66adf169e691f5dd53325ecce1adcc0))



### New Contributors

- [@Sharlie89](https://github.com/Sharlie89) made their first contribution in [#1258](https://github.com/mozilla-ai/otari/pull/1258)

**Full Changelog**: https://github.com/mozilla-ai/otari/compare/v0.6.1...v0.6.2
## [0.6.1](https://github.com/mozilla-ai/otari/releases/tag/v0.6.1) - 2026-09-15



### Bug Fixes

- **users:** Scope the users router to the caller's organization in [#1182](https://github.com/mozilla-ai/otari/pull/1182) by [@khaledosman](https://github.com/khaledosman) ([`e80d4d3`](https://github.com/mozilla-ai/otari/commit/e80d4d36c80b50432531a3762e9d4f785a321765))
- **dashboard:** Scope the key owner picker to the caller's organization in [#1180](https://github.com/mozilla-ai/otari/pull/1180) by [@khaledosman](https://github.com/khaledosman) ([`25dd554`](https://github.com/mozilla-ai/otari/commit/25dd5542951646fa07a58aeec0337400513c8cb6))
- **dashboard:** Scope the routing list to the selected workspace in [#1181](https://github.com/mozilla-ai/otari/pull/1181) by [@khaledosman](https://github.com/khaledosman) ([`957909e`](https://github.com/mozilla-ai/otari/commit/957909e5d52c065723242142d2c945310076a557))


### Features

- **dashboard:** Compose the deployment rail through the overlay seams in [#1183](https://github.com/mozilla-ai/otari/pull/1183) by [@khaledosman](https://github.com/khaledosman) ([`f6b7657`](https://github.com/mozilla-ai/otari/commit/f6b76571fa0524b863e5d0c791bf082b0b6a4a01))


**Full Changelog**: https://github.com/mozilla-ai/otari/compare/v0.6.0...v0.6.1
## [0.6.0](https://github.com/mozilla-ai/otari/releases/tag/v0.6.0) - 2026-09-15



### Bug Fixes

- **dashboard:** Offer the activation setup guide to tenants, not only deployment operators in [#980](https://github.com/mozilla-ai/otari/pull/980) by [@khaledosman](https://github.com/khaledosman) ([`bb925a8`](https://github.com/mozilla-ai/otari/commit/bb925a8fac81e0067f16ab01e82669a2b7b1294f))
- **gateway:** Read a provider's real status when its response carries none in [#975](https://github.com/mozilla-ai/otari/pull/975) by [@daavoo](https://github.com/daavoo) ([`22f7c88`](https://github.com/mozilla-ai/otari/commit/22f7c88d942eee94e3c6b410907db4ed4093ee2e))
- **hybrid:** Forward Anthropic 1h cache-write tokens in usage reports in [#859](https://github.com/mozilla-ai/otari/pull/859) by [@AloysJehwin](https://github.com/AloysJehwin) ([`fd7b574`](https://github.com/mozilla-ai/otari/commit/fd7b574e2738cff7ed0158b23ffd97e9ef966ff0))
- **dashboard:** Remove bulk price editing from the models page in [#995](https://github.com/mozilla-ai/otari/pull/995) by [@jigjigjig](https://github.com/jigjigjig) ([`5ed641c`](https://github.com/mozilla-ai/otari/commit/5ed641ca4410ded5b717655585555d9a7c26a214))
- **dashboard:** Survive a bootstrap from an older gateway in [#962](https://github.com/mozilla-ai/otari/pull/962) by [@njbrake](https://github.com/njbrake) ([`1d3c8db`](https://github.com/mozilla-ai/otari/commit/1d3c8db6ef3a0668f13e14826b5b73c257e68ba1))
- **dashboard:** Align trailing actions to the input line in routing control rows in [#1003](https://github.com/mozilla-ai/otari/pull/1003) by [@jigjigjig](https://github.com/jigjigjig) ([`4ff5678`](https://github.com/mozilla-ai/otari/commit/4ff567841a3422dd7c468c52e99245196ef79fe3))
- **tests:** Accept an unexported card constant in the ceiling mirror checks in [#1013](https://github.com/mozilla-ai/otari/pull/1013) by [@njbrake](https://github.com/njbrake) ([`5ea6a62`](https://github.com/mozilla-ai/otari/commit/5ea6a62a1328d4efa5fcc2f667a5a76973fade5f))
- **tools:** Enforce the web-search max_uses cap in every format in [#905](https://github.com/mozilla-ai/otari/pull/905) by [@mikemikimike](https://github.com/mikemikimike) ([`cde244c`](https://github.com/mozilla-ai/otari/commit/cde244c0728910dded28ecca7a6ea49c00fa0795))
- **auth:** Check the OAuth state server-side and turn PKCE on in [#1000](https://github.com/mozilla-ai/otari/pull/1000) by [@daavoo](https://github.com/daavoo) ([`dbd917a`](https://github.com/mozilla-ai/otari/commit/dbd917a43ade20ebf061a3cfdef14ec7d8a9f2b1))
- **dashboard:** Stop the create-workspace modal rendering empty in [#1007](https://github.com/mozilla-ai/otari/pull/1007) by [@khaledosman](https://github.com/khaledosman) ([`a5e793a`](https://github.com/mozilla-ai/otari/commit/a5e793a00807cd92e1f24c81c07d9118b2aea86d))
- **keys:** Conceal an issued key until the operator asks to see it in [#1016](https://github.com/mozilla-ai/otari/pull/1016) by [@khaledosman](https://github.com/khaledosman) ([`3c19010`](https://github.com/mozilla-ai/otari/commit/3c19010f661aaa88d808e5f6e51e660062377b50))
- **dashboard:** Keep the measuring harness out of the published catalog in [#1024](https://github.com/mozilla-ai/otari/pull/1024) by [@khaledosman](https://github.com/khaledosman) ([`e4fc71f`](https://github.com/mozilla-ai/otari/commit/e4fc71f223fe9e2ba431d5ee404a0610c2625875))
- **dashboard:** Stop every catalog story querying the real gateway in [#1022](https://github.com/mozilla-ai/otari/pull/1022) by [@khaledosman](https://github.com/khaledosman) ([`b0dd15a`](https://github.com/mozilla-ai/otari/commit/b0dd15a0077dae9382bc98a57ef88e8826b0c61f))
- **dashboard:** Render the published catalog at the path it is served from in [#1029](https://github.com/mozilla-ai/otari/pull/1029) by [@khaledosman](https://github.com/khaledosman) ([`1399555`](https://github.com/mozilla-ai/otari/commit/139955584b83807c15dff4e39a0636e9e46daae3))
- **dashboard:** Put the organization rename behind a confirming dialog in [#1030](https://github.com/mozilla-ai/otari/pull/1030) by [@khaledosman](https://github.com/khaledosman) ([`5529a0e`](https://github.com/mozilla-ai/otari/commit/5529a0eed07a75dcb9c32f3a528425dba916558e))
- **dashboard:** Put every record deletion behind a confirm dialog in [#1034](https://github.com/mozilla-ai/otari/pull/1034) by [@khaledosman](https://github.com/khaledosman) ([`a2942ab`](https://github.com/mozilla-ai/otari/commit/a2942abcfccb244c9123f2f728a1b7f139f55f2b))
- **dashboard:** Say why a dropdown is empty, and share one combo box in [#1039](https://github.com/mozilla-ai/otari/pull/1039) by [@khaledosman](https://github.com/khaledosman) ([`96dccc4`](https://github.com/mozilla-ai/otari/commit/96dccc4292cbab7207d499f78b293cb7345eb032))
- **config:** Describe data_plane_url against the API root it is suffixed with in [#1069](https://github.com/mozilla-ai/otari/pull/1069) by [@peteski22](https://github.com/peteski22) ([`eb9a783`](https://github.com/mozilla-ai/otari/commit/eb9a7838a6e754cb9281e2bed8918ca384ae420b))
- **dashboard:** Carry focus through ListDetail's column swap in [#1072](https://github.com/mozilla-ai/otari/pull/1072) by [@khaledosman](https://github.com/khaledosman) ([`b25c1f8`](https://github.com/mozilla-ai/otari/commit/b25c1f8dd7769e6e0d200b818a18c539ca599b53))
- **dashboard:** Show people by name in the user pickers, not their UUIDs in [#1051](https://github.com/mozilla-ai/otari/pull/1051) by [@khaledosman](https://github.com/khaledosman) ([`057d715`](https://github.com/mozilla-ai/otari/commit/057d715d650593495cd07d40edebb7d2a4333199))
- **dashboard:** Key Storybook's API mock on the API root in [#1052](https://github.com/mozilla-ai/otari/pull/1052) by [@jigjigjig](https://github.com/jigjigjig) ([`199a1f3`](https://github.com/mozilla-ai/otari/commit/199a1f3f69ab029f47b2d72f5ce9edc9c2198e06))
- **dashboard:** Line the passkey button up with the field it submits in [#1084](https://github.com/mozilla-ai/otari/pull/1084) by [@jigjigjig](https://github.com/jigjigjig) ([`0d64c5a`](https://github.com/mozilla-ai/otari/commit/0d64c5ae27a8b37c3a5c9c5ffde90f27c4c04e01))
- **dashboard:** Re-seed guardrail parameters when the profile changes in [#1086](https://github.com/mozilla-ai/otari/pull/1086) by [@khaledosman](https://github.com/khaledosman) ([`3a44606`](https://github.com/mozilla-ai/otari/commit/3a44606bf2f1328dbc70822051bc83539e662c2d))
- **dashboard:** Let the tools pages fill the page column in [#1090](https://github.com/mozilla-ai/otari/pull/1090) by [@khaledosman](https://github.com/khaledosman) ([`f4eb3a7`](https://github.com/mozilla-ai/otari/commit/f4eb3a7bedd8b3def922b404574e89611ec56ae0))
- **dashboard:** Use Otari branding and page-specific tab titles in [#1097](https://github.com/mozilla-ai/otari/pull/1097) by [@jigjigjig](https://github.com/jigjigjig) ([`34ef833`](https://github.com/mozilla-ai/otari/commit/34ef83361ef225f182946178d9ba9a41f46fcdb1))
- **dashboard:** Open every row Edit in a form dialog in [#1102](https://github.com/mozilla-ai/otari/pull/1102) by [@khaledosman](https://github.com/khaledosman) ([`aac0e19`](https://github.com/mozilla-ai/otari/commit/aac0e198349e8059ebb7c89f3aefbb4756e9e9f0))
- **dashboard:** Give every settings row one control lane in [#1100](https://github.com/mozilla-ai/otari/pull/1100) by [@khaledosman](https://github.com/khaledosman) ([`ddb7e77`](https://github.com/mozilla-ai/otari/commit/ddb7e77704d6154d506251fc4187d09d85d14fb9))
- **dashboard:** Open a tooltip in 300ms rather than HeroUI's 1.5s in [#1123](https://github.com/mozilla-ai/otari/pull/1123) by [@jigjigjig](https://github.com/jigjigjig) ([`5e79222`](https://github.com/mozilla-ai/otari/commit/5e79222c2e233f9c611405fdf8db99ad3396befd))
- **dashboard:** Link the exhausted member dialog to Members & roles in [#1136](https://github.com/mozilla-ai/otari/pull/1136) by [@jigjigjig](https://github.com/jigjigjig) ([`ca795f7`](https://github.com/mozilla-ai/otari/commit/ca795f7cada4c04ae181451209a1d009f183de88))
- Point the OAuth callback and emailed links at the interface's own origin in [#1139](https://github.com/mozilla-ai/otari/pull/1139) by [@peteski22](https://github.com/peteski22) ([`ca84000`](https://github.com/mozilla-ai/otari/commit/ca84000fd86185b4964b0f415a36cec98dd46b43))
- **dashboard:** Suggest the real models and provider instances in three dialogs in [#1151](https://github.com/mozilla-ai/otari/pull/1151) by [@khaledosman](https://github.com/khaledosman) ([`2e53cb1`](https://github.com/mozilla-ai/otari/commit/2e53cb16a588f203552b2f4780a5137dfa7fa07b))
- **dashboard:** Name an unnamed budget by what it caps, not by its id in [#1150](https://github.com/mozilla-ai/otari/pull/1150) by [@khaledosman](https://github.com/khaledosman) ([`e70649c`](https://github.com/mozilla-ai/otari/commit/e70649ca968acd3f0fe73c774f63671017b495a3))
- **dashboard:** Draw a placeholder as a hint, not as a saved default in [#1154](https://github.com/mozilla-ai/otari/pull/1154) by [@khaledosman](https://github.com/khaledosman) ([`157bf3d`](https://github.com/mozilla-ai/otari/commit/157bf3df6e8047a0b9c60edcba0abe30f59361f9))
- **dashboard:** Name the person behind a user id on Usage, Activity and Budgets in [#1155](https://github.com/mozilla-ai/otari/pull/1155) by [@khaledosman](https://github.com/khaledosman) ([`03f8f59`](https://github.com/mozilla-ai/otari/commit/03f8f59ae8a166239f5cffb03713ae595f5157ee))
- **dashboard:** Conceal API keys by default and align key controls in [#1161](https://github.com/mozilla-ai/otari/pull/1161) by [@jigjigjig](https://github.com/jigjigjig) ([`ba5c597`](https://github.com/mozilla-ai/otari/commit/ba5c59795e28ee4ba0a65cd4a82db99bca8c7d08))
- **dashboard:** Make KPI sparklines non-interactive in [#1163](https://github.com/mozilla-ai/otari/pull/1163) by [@jigjigjig](https://github.com/jigjigjig) ([`71e000f`](https://github.com/mozilla-ai/otari/commit/71e000f0b769ed8a428b0a0224cedda229b7c136))
- **dashboard:** Match a deployment-supplied model listed under its legacy key in [#1168](https://github.com/mozilla-ai/otari/pull/1168) by [@khaledosman](https://github.com/khaledosman) ([`00c979b`](https://github.com/mozilla-ai/otari/commit/00c979b6c104bfa760ee13ec82244c38c8250a56))
- **dashboard:** Restore activation feedback and center the success burst in [#1167](https://github.com/mozilla-ai/otari/pull/1167) by [@jigjigjig](https://github.com/jigjigjig) ([`29330cc`](https://github.com/mozilla-ai/otari/commit/29330cc76ca225b8fe5501fb766aa3ed5ed76205))
- **dashboard:** Remove activity chart tooltip lag in [#1169](https://github.com/mozilla-ai/otari/pull/1169) by [@jigjigjig](https://github.com/jigjigjig) ([`7e05eff`](https://github.com/mozilla-ai/otari/commit/7e05eff2cd15700a0083131e21e6cc80c0bc8459))


### Features

- **dashboard:** Ask each provider for the credential fields it needs in [#971](https://github.com/mozilla-ai/otari/pull/971) by [@khaledosman](https://github.com/khaledosman) ([`1b96b31`](https://github.com/mozilla-ai/otari/commit/1b96b31aaf9971b35bd9e5a2d8831f8f11046d0c))
- **dashboard:** Give the tenant Overview the budget signal it had none of in [#986](https://github.com/mozilla-ai/otari/pull/986) by [@khaledosman](https://github.com/khaledosman) ([`c7424db`](https://github.com/mozilla-ai/otari/commit/c7424db94a292661fcc971fd3b7684091aecb814))
- **cli:** Add otari import claude-code to backfill transcript usage in [#966](https://github.com/mozilla-ai/otari/pull/966) by [@shoemoney](https://github.com/shoemoney) ([`280dd97`](https://github.com/mozilla-ai/otari/commit/280dd972f7601be17c036b673e1952f7e495403f))
- **config:** Name the API and OTLP mount roots once in [#1005](https://github.com/mozilla-ai/otari/pull/1005) by [@peteski22](https://github.com/peteski22) ([`ddd1ff4`](https://github.com/mozilla-ai/otari/commit/ddd1ff4954a64caa4b5d5f56c432e232a60e372a))
- **dashboard:** Give a workspace's provider-key overrides and model restrictions a surface in [#983](https://github.com/mozilla-ai/otari/pull/983) by [@khaledosman](https://github.com/khaledosman) ([`5b0d202`](https://github.com/mozilla-ai/otari/commit/5b0d2022fae9ac7c7cc58d86478fd63fb0d7f0de))
- **dashboard:** Rebuild the Web search page on an autosaving row grammar in [#985](https://github.com/mozilla-ai/otari/pull/985) by [@jigjigjig](https://github.com/jigjigjig) ([`6405ff3`](https://github.com/mozilla-ai/otari/commit/6405ff341d2d5653e9906e6678767cbdc979bd1a))
- **errors:** Return the provider's own message on a rate-limited request by [@daavoo](https://github.com/daavoo) ([`8a66b66`](https://github.com/mozilla-ai/otari/commit/8a66b6674d06d14c453763de2ae257c836a56089))
- **errors:** Forward the upstream Retry-After on a rate-limited request by [@daavoo](https://github.com/daavoo) ([`3a21d7d`](https://github.com/mozilla-ai/otari/commit/3a21d7d1aedace6458bd94c7386ba04e69305e71))
- **dashboard:** Extract a design system layer with a published catalog in [#970](https://github.com/mozilla-ai/otari/pull/970) by [@khaledosman](https://github.com/khaledosman) ([`8e971fe`](https://github.com/mozilla-ai/otari/commit/8e971feb2fae9b275306cad15fc72703bf4f2680))
- **mcp:** Replace inline MCP execution with stored-server endpoints in [#812](https://github.com/mozilla-ai/otari/pull/812) by [@HareeshBahuleyan](https://github.com/HareeshBahuleyan) ([`8da3c86`](https://github.com/mozilla-ai/otari/commit/8da3c8625b285f0343884da88a17473b6f4a9197))
- **dashboard:** Add FormDialog, the one surface every create form opens in in [#1032](https://github.com/mozilla-ai/otari/pull/1032) by [@jigjigjig](https://github.com/jigjigjig) ([`207008d`](https://github.com/mozilla-ai/otari/commit/207008d632c4a37c3d8121451c615857b4f36912))
- **BREAKING:** **api:** Mount the API under /api/v1 and OTLP ingest under /otlp in [#1026](https://github.com/mozilla-ai/otari/pull/1026) by [@peteski22](https://github.com/peteski22) ([`510c8ed`](https://github.com/mozilla-ai/otari/commit/510c8ed311faa3dc69feaa3aeacda5f6a641656b))
- **BREAKING:** **api:** Name operations by tag and handler, not by path in [#1050](https://github.com/mozilla-ai/otari/pull/1050) by [@peteski22](https://github.com/peteski22) ([`f0b2ae5`](https://github.com/mozilla-ai/otari/commit/f0b2ae5842d8056960f3e7c015af59b557e26a3f))
- **dashboard:** Add ListDetail, the list-and-detail page frame in [#1054](https://github.com/mozilla-ai/otari/pull/1054) by [@khaledosman](https://github.com/khaledosman) ([`f818382`](https://github.com/mozilla-ai/otari/commit/f81838227057a20c9b67011f880e9836c436b40d))
- **guardrails:** Pick a profile and its options instead of typing them in [#972](https://github.com/mozilla-ai/otari/pull/972) by [@khaledosman](https://github.com/khaledosman) ([`98ac172`](https://github.com/mozilla-ai/otari/commit/98ac172b4a8da0ced1185f365e77d1552228a7db))
- **dashboard:** Move the keys page's one-time secret into FormDialog in [#1037](https://github.com/mozilla-ai/otari/pull/1037) by [@jigjigjig](https://github.com/jigjigjig) ([`c2b3321`](https://github.com/mozilla-ai/otari/commit/c2b33212e3abb2919eef074db5ce88d0ef467150))
- **dashboard:** Move the routing policy form into FormDialog in [#1038](https://github.com/mozilla-ai/otari/pull/1038) by [@jigjigjig](https://github.com/jigjigjig) ([`73d5ec2`](https://github.com/mozilla-ai/otari/commit/73d5ec28101d7bd614bd5f8b263f762b58f09411))
- **dashboard:** Move the budget form into FormDialog in [#1040](https://github.com/mozilla-ai/otari/pull/1040) by [@jigjigjig](https://github.com/jigjigjig) ([`185193b`](https://github.com/mozilla-ai/otari/commit/185193b9cc9223d15aa2ee0729ca5dfd2d74f87a))
- **dashboard:** Move both workspace creates and the organization one into FormDialog in [#1041](https://github.com/mozilla-ai/otari/pull/1041) by [@jigjigjig](https://github.com/jigjigjig) ([`eff99e0`](https://github.com/mozilla-ai/otari/commit/eff99e07997804e5b034a43a1363d2736766c219))
- **dashboard:** Move the add-provider form into FormDialog in [#1045](https://github.com/mozilla-ai/otari/pull/1045) by [@jigjigjig](https://github.com/jigjigjig) ([`194033e`](https://github.com/mozilla-ai/otari/commit/194033e334f6f3553f135b135845a04392e653cc))
- **dashboard:** Move the organization provider-key form into FormDialog in [#1048](https://github.com/mozilla-ai/otari/pull/1048) by [@jigjigjig](https://github.com/jigjigjig) ([`96c81c3`](https://github.com/mozilla-ai/otari/commit/96c81c3204027a5932e88878275c50cf1f9790f4))
- **dashboard:** Move the domain claim form into FormDialog in [#1062](https://github.com/mozilla-ai/otari/pull/1062) by [@jigjigjig](https://github.com/jigjigjig) ([`4d429ad`](https://github.com/mozilla-ai/otari/commit/4d429add1ba48062120fa672fee9d5dac774439c))
- **dashboard:** Move the member add and invite forms into FormDialog in [#1068](https://github.com/mozilla-ai/otari/pull/1068) by [@jigjigjig](https://github.com/jigjigjig) ([`85b915b`](https://github.com/mozilla-ai/otari/commit/85b915bb01ecb81f0229b721186060cb6c8d8a4c))
- **dashboard:** Put the two Tools creation forms in a dialog in [#1071](https://github.com/mozilla-ai/otari/pull/1071) by [@jigjigjig](https://github.com/jigjigjig) ([`5855af3`](https://github.com/mozilla-ai/otari/commit/5855af3507c6975bc77e6f6f21819591a62e2de9))
- **dashboard:** Let a routing policy be created for several users at once in [#1056](https://github.com/mozilla-ai/otari/pull/1056) by [@khaledosman](https://github.com/khaledosman) ([`451bc2d`](https://github.com/mozilla-ai/otari/commit/451bc2d8981f22ba1b86dd0640ba7deddc4adb9e))
- **dashboard:** Move the last six forms out of AlertDialog in [#1074](https://github.com/mozilla-ai/otari/pull/1074) by [@jigjigjig](https://github.com/jigjigjig) ([`8bf84da`](https://github.com/mozilla-ai/otari/commit/8bf84da112eab2439283136ce6c00e3d79b251ca))
- **dashboard:** Give every table row action a glyph instead of a word in [#1089](https://github.com/mozilla-ai/otari/pull/1089) by [@khaledosman](https://github.com/khaledosman) ([`9a14cf3`](https://github.com/mozilla-ai/otari/commit/9a14cf35b7a5fd2df8b246a3fb8d2e61c3f72ea1))
- **auth:** Always offer password sign-in, and open signup where configured in [#1092](https://github.com/mozilla-ai/otari/pull/1092) by [@khaledosman](https://github.com/khaledosman) ([`5c4c1b0`](https://github.com/mozilla-ai/otari/commit/5c4c1b0caff5ba992236dc4b681c08b64dd61fab))
- **dashboard:** Add animated background to login and signup in [#1093](https://github.com/mozilla-ai/otari/pull/1093) by [@jigjigjig](https://github.com/jigjigjig) ([`cb7b848`](https://github.com/mozilla-ai/otari/commit/cb7b84835114a7a638ec1d36199583ca2ba8be13))
- **dashboard:** Restore the first-run flow as a guided sheet in [#1101](https://github.com/mozilla-ai/otari/pull/1101) by [@khaledosman](https://github.com/khaledosman) ([`7062690`](https://github.com/mozilla-ai/otari/commit/706269043c755720de82c07057671cbef0a365ae))
- **guardrails:** Build a catalog of the guardrails otari ships by [@dpoulopoulos](https://github.com/dpoulopoulos) ([`932a2fa`](https://github.com/mozilla-ai/otari/commit/932a2faef522cca0bb788586545c2a525cf1085d))
- **api:** Serve the built-in guardrail catalog in [#1116](https://github.com/mozilla-ai/otari/pull/1116) by [@dpoulopoulos](https://github.com/dpoulopoulos) ([`be5b7a6`](https://github.com/mozilla-ai/otari/commit/be5b7a6921adf8d77becf7096718e717b77adf30))
- **dashboard:** Bring back the Playground in [#1131](https://github.com/mozilla-ai/otari/pull/1131) by [@khaledosman](https://github.com/khaledosman) ([`e8153f2`](https://github.com/mozilla-ai/otari/commit/e8153f2c2a51b49f3262cd6af04884bd3a40a893))
- **dashboard:** Make a workspace's provider-key overrides usable in [#1143](https://github.com/mozilla-ai/otari/pull/1143) by [@khaledosman](https://github.com/khaledosman) ([`d95197c`](https://github.com/mozilla-ai/otari/commit/d95197c36e44b288ea202c43ccf59b3106cf153d))
- **dashboard:** Show both ends of an API key, plus fixes from a release-readiness pass in [#1142](https://github.com/mozilla-ai/otari/pull/1142) by [@jigjigjig](https://github.com/jigjigjig) ([`e6d1184`](https://github.com/mozilla-ai/otari/commit/e6d1184118dc841651db2411fecad73f79da1958))
- **dashboard:** Let people set the name they are known by in [#1162](https://github.com/mozilla-ai/otari/pull/1162) by [@khaledosman](https://github.com/khaledosman) ([`17ba56b`](https://github.com/mozilla-ai/otari/commit/17ba56bec9c7b4b5d3649c8641fb80f79ca5dd28))
- **gateway:** Record time-to-first-token on the streaming path in [#1099](https://github.com/mozilla-ai/otari/pull/1099) by [@AmirF194](https://github.com/AmirF194) ([`83167bf`](https://github.com/mozilla-ai/otari/commit/83167bf3cc9133df1b7afb0e884903186f918093))
- **catalog:** Group the model catalog by model, with one offering per provider in [#1015](https://github.com/mozilla-ai/otari/pull/1015) by [@njbrake](https://github.com/njbrake) ([`76d88d9`](https://github.com/mozilla-ai/otari/commit/76d88d96f5fe9a484c9d7b223946d169b9a4d6db))


### Security

- **guardrails:** Mask secret organization guardrail parameters in [#1088](https://github.com/mozilla-ai/otari/pull/1088) by [@khaledosman](https://github.com/khaledosman) ([`1f09106`](https://github.com/mozilla-ai/otari/commit/1f09106254c02be191551410572424270c2f04b4))
- Refuse an organization rate override for a deployment-supplied model in [#1164](https://github.com/mozilla-ai/otari/pull/1164) by [@khaledosman](https://github.com/khaledosman) ([`335afb5`](https://github.com/mozilla-ai/otari/commit/335afb5ea0f260fd9d61f850efe0115f37732061))



### New Contributors

- [@daavoo](https://github.com/daavoo) made their first contribution in [#1019](https://github.com/mozilla-ai/otari/pull/1019)
- [@mikemikimike](https://github.com/mikemikimike) made their first contribution in [#905](https://github.com/mozilla-ai/otari/pull/905)
- [@ZhiXia-coder](https://github.com/ZhiXia-coder) made their first contribution in [#788](https://github.com/mozilla-ai/otari/pull/788)

**Full Changelog**: https://github.com/mozilla-ai/otari/compare/v0.5.1...v0.6.0
## [0.5.1](https://github.com/mozilla-ai/otari/releases/tag/v0.5.1) - 2026-09-08



### Bug Fixes

- **a11y:** Stop sidebar nav disclosures from emitting document headings in [#951](https://github.com/mozilla-ai/otari/pull/951) by [@AloysJehwin](https://github.com/AloysJehwin) ([`8722739`](https://github.com/mozilla-ai/otari/commit/87227391fb2b1114611dcdfe6319aa5e02d143a6))
- **lint:** Strip block comments before matching arbitrary font-size rule in [#941](https://github.com/mozilla-ai/otari/pull/941) by [@AloysJehwin](https://github.com/AloysJehwin) ([`3146bea`](https://github.com/mozilla-ai/otari/commit/3146bea94de73004d5facca3f82d15359783870c))
- **settings:** Prevent database_url value from overflowing Settings card in [#943](https://github.com/mozilla-ai/otari/pull/943) by [@AloysJehwin](https://github.com/AloysJehwin) ([`83d4c24`](https://github.com/mozilla-ai/otari/commit/83d4c2410ef5068835e396224b9cd794028ecb0d))
- **gateway:** Let the platform health probe reach a route outside base_url's own path in [#964](https://github.com/mozilla-ai/otari/pull/964) by [@macaab26](https://github.com/macaab26) ([`2e0f885`](https://github.com/mozilla-ai/otari/commit/2e0f88501011c48de112b576d0c09dd44c1b8e12))


**Full Changelog**: https://github.com/mozilla-ai/otari/compare/v0.5.0...v0.5.1
## [0.5.0](https://github.com/mozilla-ai/otari/releases/tag/v0.5.0) - 2026-09-08



### Bug Fixes

- **providers:** Honor the optional API key for keyless custom endpoints in [#423](https://github.com/mozilla-ai/otari/pull/423) by [@njbrake](https://github.com/njbrake) ([`5fa80d0`](https://github.com/mozilla-ai/otari/commit/5fa80d0571346ed9ae3f760c5931be949feaa38b))
- **dashboard:** Accessibility and UX polish pass in [#435](https://github.com/mozilla-ai/otari/pull/435) by [@njbrake](https://github.com/njbrake) ([`b153949`](https://github.com/mozilla-ai/otari/commit/b15394939d5180007b8d76ff307d7bcf221c9f8c))
- **dashboard:** Address timeline review follow-ups from #445 in [#453](https://github.com/mozilla-ai/otari/pull/453) by [@njbrake](https://github.com/njbrake) ([`75556a9`](https://github.com/mozilla-ai/otari/commit/75556a954968ad271bcb7878a9f022ff4eff2e9a))
- **dashboard:** Distinguish a missing /models endpoint from an unreachable provider in [#454](https://github.com/mozilla-ai/otari/pull/454) by [@njbrake](https://github.com/njbrake) ([`ab4d3ca`](https://github.com/mozilla-ai/otari/commit/ab4d3ca25788adc858a449592960004345c2f939))
- **providers:** Gate provider api_base write path behind SSRF check in [#444](https://github.com/mozilla-ai/otari/pull/444) by [@njbrake](https://github.com/njbrake) ([`cb5d6ec`](https://github.com/mozilla-ai/otari/commit/cb5d6ecafb648eed27b5f9a19c641e5adc22d42c))
- **otlp:** Price imported cache creation at the 5m rate, not 1h in [#468](https://github.com/mozilla-ai/otari/pull/468) by [@njbrake](https://github.com/njbrake) ([`ad4b815`](https://github.com/mozilla-ai/otari/commit/ad4b815d9815c5ae6cce997c2576a427b059564d))
- **models:** Let a keyless local provider opt into discovery in [#471](https://github.com/mozilla-ai/otari/pull/471) by [@njbrake](https://github.com/njbrake) ([`6872df3`](https://github.com/mozilla-ai/otari/commit/6872df3603782831692e59a1d49a1addad4dcfef))
- **pricing:** Follow genai-prices to the live v2 price feed in [#466](https://github.com/mozilla-ai/otari/pull/466) by [@HareeshBahuleyan](https://github.com/HareeshBahuleyan) ([`3773675`](https://github.com/mozilla-ai/otari/commit/3773675884e13c56b01e1b5010efce2ecc367ea6))
- **dashboard:** Clear a provider's connection-test result when it changes in [#467](https://github.com/mozilla-ai/otari/pull/467) by [@njbrake](https://github.com/njbrake) ([`2286e0a`](https://github.com/mozilla-ai/otari/commit/2286e0a20d6d34af7e4bc0f01c05aa4acdb00bfa))
- **lint:** Apply every enclosing layer's rules in the architecture check in [#476](https://github.com/mozilla-ai/otari/pull/476) by [@njbrake](https://github.com/njbrake) ([`e4c79d1`](https://github.com/mozilla-ai/otari/commit/e4c79d14ca5345221a2d59f4a272cdb8ef887b1b))
- **errors:** Classify provider billing exhaustion instead of calling it a bad request in [#483](https://github.com/mozilla-ai/otari/pull/483) by [@njbrake](https://github.com/njbrake) ([`fc41d02`](https://github.com/mozilla-ai/otari/commit/fc41d025906c17ffbc241dba3f1817458ec48590))
- **hybrid:** Prevent pending cost calculations when fallback attempts stop in [#498](https://github.com/mozilla-ai/otari/pull/498) by [@HareeshBahuleyan](https://github.com/HareeshBahuleyan) ([`60864af`](https://github.com/mozilla-ai/otari/commit/60864af7478723c328e28c38f6411759a3264704))
- **dashboard:** Correct the routing copy for mid-plan stops and unloaded plans in [#507](https://github.com/mozilla-ai/otari/pull/507) by [@njbrake](https://github.com/njbrake) ([`e7edbdf`](https://github.com/mozilla-ai/otari/commit/e7edbdf1e967472a05b5b51bb289d3c4fe0998bd))
- **dashboard:** Open the onboarding quickstart link in a new tab in [#508](https://github.com/mozilla-ai/otari/pull/508) by [@njbrake](https://github.com/njbrake) ([`ef9312e`](https://github.com/mozilla-ai/otari/commit/ef9312e955dbcebfb4989f0c2b296d8a290ca034))
- **dashboard:** Report a failed provider query on the Overview page in [#509](https://github.com/mozilla-ai/otari/pull/509) by [@njbrake](https://github.com/njbrake) ([`7b56ce7`](https://github.com/mozilla-ai/otari/commit/7b56ce75c3e3ad248102854b4b808bcd00cef385))
- **hybrid:** Re-raise CancelledError in platform attempt walker in [#511](https://github.com/mozilla-ai/otari/pull/511) by [@njbrake](https://github.com/njbrake) ([`e7d9370`](https://github.com/mozilla-ai/otari/commit/e7d937077522383148e608bd60bff3baf62bd111))
- **sdk-codegen:** Allow non_snake_case in the generated Rust core in [#515](https://github.com/mozilla-ai/otari/pull/515) by [@njbrake](https://github.com/njbrake) ([`65ada5f`](https://github.com/mozilla-ai/otari/commit/65ada5f1b7c791f64ce81414e923b9f3d5251608))
- **sdk-codegen:** Make the gateway the source of truth for endpoint coverage in [#516](https://github.com/mozilla-ai/otari/pull/516) by [@njbrake](https://github.com/njbrake) ([`af2137c`](https://github.com/mozilla-ai/otari/commit/af2137cabe0ffbc3c85445467a15704ec989dfe8))
- **ci:** Flag regen PRs on failing checks, not just age in [#518](https://github.com/mozilla-ai/otari/pull/518) by [@njbrake](https://github.com/njbrake) ([`f9c59f7`](https://github.com/mozilla-ai/otari/commit/f9c59f7a22adf47f5e6a1e3aeecf2def2a6dc1ac))
- **errors:** Return the provider's own message on a caller-fault rejection in [#537](https://github.com/mozilla-ai/otari/pull/537) by [@njbrake](https://github.com/njbrake) ([`8b4f2b4`](https://github.com/mozilla-ai/otari/commit/8b4f2b4e2e045f7c180208b7a7f488ad151a4de8))
- **billing:** Record charge lines for embeddings, images, moderations, and rerank in [#542](https://github.com/mozilla-ai/otari/pull/542) by [@AmirF194](https://github.com/AmirF194) ([`98cf54f`](https://github.com/mozilla-ai/otari/commit/98cf54f1e0ae090effcd166d15c36e9290af9416))
- **router:** Match routing-memory scores on canonical model identity in [#547](https://github.com/mozilla-ai/otari/pull/547) by [@njbrake](https://github.com/njbrake) ([`f75e5b0`](https://github.com/mozilla-ai/otari/commit/f75e5b0eac77d488fd29cc7a6e867f18a6333a31))
- **routing:** Fall back on every provider failure in [#546](https://github.com/mozilla-ai/otari/pull/546) by [@njbrake](https://github.com/njbrake) ([`0024769`](https://github.com/mozilla-ai/otari/commit/002476920daee6507ab8e907a26654c6d794dbef))
- **billing:** Audio charge lines, no token defaults on non-token routes in [#550](https://github.com/mozilla-ai/otari/pull/550) by [@njbrake](https://github.com/njbrake) ([`1ab8a4b`](https://github.com/mozilla-ai/otari/commit/1ab8a4bd6d47fd4da964cafe570624715b4bf734))
- **dashboard:** Stop the expanded table row remounting on every poll in [#558](https://github.com/mozilla-ai/otari/pull/558) by [@njbrake](https://github.com/njbrake) ([`ee2aadd`](https://github.com/mozilla-ai/otari/commit/ee2aadd7971741ed572ef1e6c0e7f86d5d038770))
- **pricing:** Match genai-prices defaults for renamed instances and vendor-prefixed models in [#559](https://github.com/mozilla-ai/otari/pull/559) by [@pss-julien](https://github.com/pss-julien) ([`c0cff4a`](https://github.com/mozilla-ai/otari/commit/c0cff4a0855fda2f94937dba2814be08df03bbee))
- **chat:** Forward service_tier to the provider instead of dropping it in [#561](https://github.com/mozilla-ai/otari/pull/561) by [@pss-julien](https://github.com/pss-julien) ([`17c8a8d`](https://github.com/mozilla-ai/otari/commit/17c8a8d18719833398f446907aedb23daca2d8b9))
- **models:** Tolerate a provider that reports no model `created` in [#562](https://github.com/mozilla-ai/otari/pull/562) by [@pss-julien](https://github.com/pss-julien) ([`659fa79`](https://github.com/mozilla-ai/otari/commit/659fa7964e685cb66a3c0c8529a63b57060f0c35))
- **dashboard:** Fit the share card's hero to its value and rebuild the wide layout in [#566](https://github.com/mozilla-ai/otari/pull/566) by [@njbrake](https://github.com/njbrake) ([`4c7be2f`](https://github.com/mozilla-ai/otari/commit/4c7be2fe00f3dfab47f2968927630c13fc33ed2d))
- **usage:** Pin offset-less usage window bounds to UTC in [#570](https://github.com/mozilla-ai/otari/pull/570) by [@njbrake](https://github.com/njbrake) ([`cd58170`](https://github.com/mozilla-ai/otari/commit/cd58170e8cc76a7c62435933d7907ff7b28087ce))
- **web-search:** Serialize extraction and return its retained heap in [#571](https://github.com/mozilla-ai/otari/pull/571) by [@njbrake](https://github.com/njbrake) ([`1e5e908`](https://github.com/mozilla-ai/otari/commit/1e5e908a2ed45cc44409e6c23ea7a675562a581c))
- **dashboard:** Bound and await sign-out so it can't clobber a fresh sign-in in [#573](https://github.com/mozilla-ai/otari/pull/573) by [@champ18ion](https://github.com/champ18ion) ([`c106d63`](https://github.com/mozilla-ai/otari/commit/c106d635d257b5cf46c5684f9024d6ffcbfafd43))
- Emit Node-compatible imports in TypeScript SDK codegen in [#575](https://github.com/mozilla-ai/otari/pull/575) by [@Copilot](https://github.com/Copilot) ([`9458131`](https://github.com/mozilla-ai/otari/commit/9458131764a90968c9369d83a909bc576cfb9bdf))
- **code-execution:** Three wrong verdicts in the conformance check in [#602](https://github.com/mozilla-ai/otari/pull/602) by [@khaledosman](https://github.com/khaledosman) ([`ead01fc`](https://github.com/mozilla-ai/otari/commit/ead01fc5619d80fed13d8873afc22ddda73df0d4))
- **e2e:** Wait for the provider form to close, not for a button to be relabeled in [#612](https://github.com/mozilla-ai/otari/pull/612) by [@njbrake](https://github.com/njbrake) ([`5522263`](https://github.com/mozilla-ai/otari/commit/5522263e733e48605720f79772f994805213da45))
- **auth:** Move the API key format check off the verify path in [#665](https://github.com/mozilla-ai/otari/pull/665) by [@njbrake](https://github.com/njbrake) ([`49aeeeb`](https://github.com/mozilla-ai/otari/commit/49aeeeb3b6133c6508ae5d818227f27bc6f6bf6f))
- **migrations:** Rejoin the two Alembic heads so upgrades run again in [#673](https://github.com/mozilla-ai/otari/pull/673) by [@khaledosman](https://github.com/khaledosman) ([`e798d3e`](https://github.com/mozilla-ai/otari/commit/e798d3eb9859fc46720f4210aae34de2a71be8d4))
- **batches:** Price a batch one request at a time by [@njbrake](https://github.com/njbrake) ([`0b40722`](https://github.com/mozilla-ai/otari/commit/0b407224259cddaa8f698396a802e133466a2825))
- **pricing:** Round a batch once, and coerce every rate reader by [@njbrake](https://github.com/njbrake) ([`e44c432`](https://github.com/mozilla-ai/otari/commit/e44c43253d0d5a9c6b0d6e7c11bc202f5aa36ca9))
- **pricing:** Chain the money migration onto the current head by [@njbrake](https://github.com/njbrake) ([`5c2f348`](https://github.com/mozilla-ai/otari/commit/5c2f34843c0ba314bd13b7805efd90e07454302a))
- **pricing:** Keep cached tokens in batch pricing, reserve the dearest rate in [#683](https://github.com/mozilla-ai/otari/pull/683) by [@njbrake](https://github.com/njbrake) ([`6c4f83a`](https://github.com/mozilla-ai/otari/commit/6c4f83a704778814bd18550845e429f1ec2febb0))
- **mail:** Correct three comments that describe code this PR changed by [@njbrake](https://github.com/njbrake) ([`dbe5cf2`](https://github.com/mozilla-ai/otari/commit/dbe5cf2cf35e5f978047552e1f763ec5e7fb5384))
- **mail:** Make mail readiness independent of startup validation by [@njbrake](https://github.com/njbrake) ([`8db9ef3`](https://github.com/mozilla-ai/otari/commit/8db9ef3a14d0ef1665500610a01f42cab04e7c02))
- **mail:** Use the right article for the role in an invitation by [@njbrake](https://github.com/njbrake) ([`3472d64`](https://github.com/mozilla-ai/otari/commit/3472d6498da989cc3c38e14c1227a30cbc7ce221))
- **mail:** Sanitize the envelope addresses, and work the review findings by [@njbrake](https://github.com/njbrake) ([`cdea86c`](https://github.com/mozilla-ai/otari/commit/cdea86c1efdbe93b8045cafcd839fef4e304d9c8))
- **pricing:** Map the override write race to 409, and agree on the canonical key in [#677](https://github.com/mozilla-ai/otari/pull/677) by [@khaledosman](https://github.com/khaledosman) ([`f84f3fa`](https://github.com/mozilla-ai/otari/commit/f84f3fa51705c4dd484d098d5a32c1ee014523e5))
- **auth:** Match a held address by casing, and revoke a deactivated identity's sessions in [#701](https://github.com/mozilla-ai/otari/pull/701) by [@khaledosman](https://github.com/khaledosman) ([`97ad75f`](https://github.com/mozilla-ai/otari/commit/97ad75f67efb6c63d05dbd2e2652cd18acde88bf))
- **dashboard:** Wire the password hints to their inputs, and keep the button focusable by [@khaledosman](https://github.com/khaledosman) ([`3d760e5`](https://github.com/mozilla-ai/otari/commit/3d760e58253fdfd5831b5441654ce442ced1eef8))
- **dashboard:** Clear the saved line when the password fields are retyped by [@khaledosman](https://github.com/khaledosman) ([`b03016b`](https://github.com/mozilla-ai/otari/commit/b03016b67c4e65b9f49cd077711e6f2ac9e8bf4b))
- **dashboard:** Let a claim correct the bootstrap the whole tab reads by [@khaledosman](https://github.com/khaledosman) ([`f1f809e`](https://github.com/mozilla-ai/otari/commit/f1f809ef06a70b8748b3d9753bc7207b4ad8195e))
- **dashboard:** Open the organization rail inside the mobile drawer by [@jigjigjig](https://github.com/jigjigjig) ([`24f970d`](https://github.com/mozilla-ai/otari/commit/24f970dab5f19411647d4abcd54e4162ef6dea60))
- **dashboard:** Preserve focus across navigation breakpoints in [#712](https://github.com/mozilla-ai/otari/pull/712) by [@jigjigjig](https://github.com/jigjigjig) ([`3464a78`](https://github.com/mozilla-ai/otari/commit/3464a78c6e2bb3db7459312abbb4ca9424b24c92))
- **deps:** Cap mcp below 2.0.0, which drops streamablehttp_client in [#699](https://github.com/mozilla-ai/otari/pull/699) by [@champ18ion](https://github.com/champ18ion) ([`ca12db9`](https://github.com/mozilla-ai/otari/commit/ca12db97a6e55a95e40fad92ebe3f4342179cd29))
- **dashboard:** Login design improvements in [#733](https://github.com/mozilla-ai/otari/pull/733) by [@jigjigjig](https://github.com/jigjigjig) ([`d3cd90d`](https://github.com/mozilla-ai/otari/commit/d3cd90dc883330e56ab86235dee50241d7f6ed8f))
- **dashboard:** Stop the bulk action bar sliding in from the left in [#735](https://github.com/mozilla-ai/otari/pull/735) by [@jigjigjig](https://github.com/jigjigjig) ([`7d7eb59`](https://github.com/mozilla-ai/otari/commit/7d7eb59a0a4a4fbcc777a10f92ccb23fada25855))
- **auth:** Gate master-key sign-in on the operator's password, not on any in [#746](https://github.com/mozilla-ai/otari/pull/746) by [@njbrake](https://github.com/njbrake) ([`2c06b78`](https://github.com/mozilla-ai/otari/commit/2c06b78b8d95e3cabc00eb51e62dddfbd0a068bd))
- **auth:** Lock the identity row before redeeming a verify or reset token in [#729](https://github.com/mozilla-ai/otari/pull/729) by [@AmirF194](https://github.com/AmirF194) ([`0dfb88d`](https://github.com/mozilla-ai/otari/commit/0dfb88d7e07454dcc411a7841cc1ec1590b29eaf))
- **usage:** Reserve the otari-ai: source prefix in external-usage ingest in [#772](https://github.com/mozilla-ai/otari/pull/772) by [@khaledosman](https://github.com/khaledosman) ([`38efb1f`](https://github.com/mozilla-ai/otari/commit/38efb1fac45f8de72104c21d36a014ac97c1e1a7))
- **usage:** Read a backfilled hosted row as usage this deployment served in [#773](https://github.com/mozilla-ai/otari/pull/773) by [@khaledosman](https://github.com/khaledosman) ([`cb0c62e`](https://github.com/mozilla-ai/otari/commit/cb0c62ee55444abdd54746c9d4db3a21906b9be1))
- **guardrails:** Let a mandated endpoint's failure follow its own on_unavailable setting in [#764](https://github.com/mozilla-ai/otari/pull/764) by [@njbrake](https://github.com/njbrake) ([`5a9c152`](https://github.com/mozilla-ai/otari/commit/5a9c152f4b5a7bddda3d4edb308d4b1981e7fb2c))
- **gateway:** Stop purpose-hint injection from corrupting list-shaped system content in [#730](https://github.com/mozilla-ai/otari/pull/730) by [@shoemoney](https://github.com/shoemoney) ([`899a8d1`](https://github.com/mozilla-ai/otari/commit/899a8d1c1e2f1d5e171b81430e5ecab1700b05cb))
- **tools:** Address review on the sandbox image and tool-set policy by [@njbrake](https://github.com/njbrake) ([`6857615`](https://github.com/mozilla-ai/otari/commit/685761535c76f8cacbe81cc0e209e6e1eec1377d))
- **tools:** Make the storable tool rule and the admission rule one rule by [@njbrake](https://github.com/njbrake) ([`3b705d5`](https://github.com/mozilla-ai/otari/commit/3b705d5077ce75959a3872427d42bb1f2e30e65f))
- **dashboard:** Keep a stale tool policy the deployment is refusing by [@njbrake](https://github.com/njbrake) ([`1e603eb`](https://github.com/mozilla-ai/otari/commit/1e603eb72a6cfb81a7ba20de6cce2f12a029bfc2))
- **migrations:** Re-point onto main's head after the usage-provenance migration in [#766](https://github.com/mozilla-ai/otari/pull/766) by [@njbrake](https://github.com/njbrake) ([`111f62c`](https://github.com/mozilla-ai/otari/commit/111f62c057f1d045bd53512fd34901cd9faf93a9))
- **dashboard:** Stop the pricing alarm from pushing the shell down in [#778](https://github.com/mozilla-ai/otari/pull/778) by [@jigjigjig](https://github.com/jigjigjig) ([`e2f8ed4`](https://github.com/mozilla-ai/otari/commit/e2f8ed4565f92ed0c3fe4a6873d8fa7dbb1d8237))
- **dashboard:** End the create-workspace flow inside the new workspace in [#780](https://github.com/mozilla-ai/otari/pull/780) by [@jigjigjig](https://github.com/jigjigjig) ([`e3a5e8c`](https://github.com/mozilla-ai/otari/commit/e3a5e8c6ec1d4f4f02dcb7f27028670999f6372a))
- **api:** Expose prompt cache keys in [#777](https://github.com/mozilla-ai/otari/pull/777) by [@HareeshBahuleyan](https://github.com/HareeshBahuleyan) ([`1e29bbb`](https://github.com/mozilla-ai/otari/commit/1e29bbb29cc7daf548bfa2e0bcbe2b7eeb595101))
- **mcp:** Reject duplicate MCP server names before they collapse silently in [#792](https://github.com/mozilla-ai/otari/pull/792) by [@AmirF194](https://github.com/AmirF194) ([`6eade83`](https://github.com/mozilla-ai/otari/commit/6eade83784ec6170a44f84326c58bb45e4ca6b8b))
- **gateway:** Address Gateway 502s on max_completion_tokens (OpenAI's current field name) in [#769](https://github.com/mozilla-ai/otari/pull/769) by [@aittalam](https://github.com/aittalam) ([`e676ced`](https://github.com/mozilla-ai/otari/commit/e676cedee403210114bff8e18e31f4dfd35e4d67))
- **dashboard:** Repair the text-subtle contrast failure, and retune the type scale in [#807](https://github.com/mozilla-ai/otari/pull/807) by [@jigjigjig](https://github.com/jigjigjig) ([`b37d7b7`](https://github.com/mozilla-ai/otari/commit/b37d7b73b7e35b33747a174126f4e49008694ee7))
- **dashboard:** Anchor the filter dropdowns under their trigger in [#800](https://github.com/mozilla-ai/otari/pull/800) by [@jigjigjig](https://github.com/jigjigjig) ([`c070c6d`](https://github.com/mozilla-ai/otari/commit/c070c6d59a984d79d06b3b20c3c4b672cfdabfbc))
- **api:** Gate the deployment-wide management plane on operator authority in [#821](https://github.com/mozilla-ai/otari/pull/821) by [@njbrake](https://github.com/njbrake) ([`eb900b8`](https://github.com/mozilla-ai/otari/commit/eb900b85e28d7c05f49c8d8a75f82adf1949f0a4))
- **dashboard:** Point hosted request snippets at the data-plane gateway in [#825](https://github.com/mozilla-ai/otari/pull/825) by [@njbrake](https://github.com/njbrake) ([`8d225d0`](https://github.com/mozilla-ai/otari/commit/8d225d076c013c07863f2fa1f8cb7cefb378abbb))
- **api:** Give hosted mode a management-only posture in [#824](https://github.com/mozilla-ai/otari/pull/824) by [@njbrake](https://github.com/njbrake) ([`b4ed34f`](https://github.com/mozilla-ai/otari/commit/b4ed34ff9a245400fca890fe9134d9013be00584))
- **config:** Refuse a data_plane_url that already names /v1 in [#828](https://github.com/mozilla-ai/otari/pull/828) by [@njbrake](https://github.com/njbrake) ([`1d7bd27`](https://github.com/mozilla-ai/otari/commit/1d7bd27ea68e35a9de4e1d0a760bb6cdd35814cb))
- **api:** Answer a HEAD probe from the mode stub routers in [#831](https://github.com/mozilla-ai/otari/pull/831) by [@khaledosman](https://github.com/khaledosman) ([`6ddc8f8`](https://github.com/mozilla-ai/otari/commit/6ddc8f8646350ce398a966f4907352eeff0054cf))
- **dashboard:** Hand an accepted invitation off to the claim form in [#841](https://github.com/mozilla-ai/otari/pull/841) by [@njbrake](https://github.com/njbrake) ([`fb391e8`](https://github.com/mozilla-ai/otari/commit/fb391e8871dd035c6d59d05b6577ec18278d2585))
- **dashboard:** Take the sidebar's operator answer off the membership context in [#852](https://github.com/mozilla-ai/otari/pull/852) by [@khaledosman](https://github.com/khaledosman) ([`ae2910e`](https://github.com/mozilla-ai/otari/commit/ae2910e1eada058ba5274809524f7db1f312b9ee))
- **dashboard:** Report provider-key encryption on a tenant-readable contract in [#851](https://github.com/mozilla-ai/otari/pull/851) by [@khaledosman](https://github.com/khaledosman) ([`140292b`](https://github.com/mozilla-ai/otari/commit/140292b4fa240ea1b0044287627ce2fa0c16ed86))
- **dashboard:** Serve the Tools pages to non-operators in [#864](https://github.com/mozilla-ai/otari/pull/864) by [@khaledosman](https://github.com/khaledosman) ([`b00696b`](https://github.com/mozilla-ai/otari/commit/b00696b5b6b83cf41605932bd06e4012ef7d5bda))
- **dashboard:** Withhold operator-only budget reads from the workspaces page in [#862](https://github.com/mozilla-ai/otari/pull/862) by [@khaledosman](https://github.com/khaledosman) ([`01b0d28`](https://github.com/mozilla-ai/otari/commit/01b0d28d4e9493bab8aa1f2d722c9c71e395b590))
- **dashboard:** Land every signed-in caller on a usage overview in [#863](https://github.com/mozilla-ai/otari/pull/863) by [@khaledosman](https://github.com/khaledosman) ([`fec213c`](https://github.com/mozilla-ai/otari/commit/fec213c98ddd5729a5b6f89bc0448d9c927ba610))
- **dashboard:** Put the last six bare headings on the type scale in [#865](https://github.com/mozilla-ai/otari/pull/865) by [@khaledosman](https://github.com/khaledosman) ([`4f96eff`](https://github.com/mozilla-ai/otari/commit/4f96eff1fb1637617c091c3c6ba90407c95edf60))
- **dashboard:** Name the person at the foot of the sidebar, not their role in [#869](https://github.com/mozilla-ai/otari/pull/869) by [@njbrake](https://github.com/njbrake) ([`6f7cf23`](https://github.com/mozilla-ai/otari/commit/6f7cf23bf7a628a5c2c54eceffefc009e972b317))
- **dashboard:** Generate the PWA manifest from the build's base path in [#858](https://github.com/mozilla-ai/otari/pull/858) by [@khaledosman](https://github.com/khaledosman) ([`e6dd017`](https://github.com/mozilla-ai/otari/commit/e6dd017f9a04b55bb4a0f602a8e724099724a4c1))
- **tenancy:** Gate the org provider-keys list on organization management in [#873](https://github.com/mozilla-ai/otari/pull/873) by [@khaledosman](https://github.com/khaledosman) ([`7aa952f`](https://github.com/mozilla-ai/otari/commit/7aa952f473bb4a240761d1cb817b5be35edd5b5e))
- **dashboard:** Open Model pricing to organization admins in [#872](https://github.com/mozilla-ai/otari/pull/872) by [@njbrake](https://github.com/njbrake) ([`9b0760c`](https://github.com/mozilla-ai/otari/commit/9b0760c46df75ed0ecac3f07161d6d871a80a06c))
- **dashboard:** Clear the accessibility debt in the key reveal, confirms, type scale and touch targets in [#890](https://github.com/mozilla-ai/otari/pull/890) by [@khaledosman](https://github.com/khaledosman) ([`b40c33e`](https://github.com/mozilla-ai/otari/commit/b40c33e33523fdda0f1c1a9e5c4835bfc8fefe11))
- **hybrid:** Map PLATFORM_HEALTH_PATH and trust only a 2xx from the platform probe in [#888](https://github.com/mozilla-ai/otari/pull/888) by [@njbrake](https://github.com/njbrake) ([`61a670e`](https://github.com/mozilla-ai/otari/commit/61a670e9eab9f15dd678153ae3ded4c287e55d63))
- **dashboard:** Decide the Overview's variant from the organization context in [#894](https://github.com/mozilla-ai/otari/pull/894) by [@khaledosman](https://github.com/khaledosman) ([`6c6f204`](https://github.com/mozilla-ai/otari/commit/6c6f2045e617e6c51bc07ba575566593d23262cf))
- **api:** Gate the deployment-wide routers on the router, not per route in [#895](https://github.com/mozilla-ai/otari/pull/895) by [@khaledosman](https://github.com/khaledosman) ([`d38a633`](https://github.com/mozilla-ai/otari/commit/d38a6333a67632c027a6236589d2c11fdf702ec2))
- **api:** Refuse to cap a gateway user at an organization's budget in [#898](https://github.com/mozilla-ai/otari/pull/898) by [@khaledosman](https://github.com/khaledosman) ([`60a5d8c`](https://github.com/mozilla-ai/otari/commit/60a5d8c07a11f93e362e59247f04e352816a0eac))
- **db:** Hand the request's connection back before the provider call in [#911](https://github.com/mozilla-ai/otari/pull/911) by [@njbrake](https://github.com/njbrake) ([`925e115`](https://github.com/mozilla-ai/otari/commit/925e115bfa321add9a46a5d3c4f6b76ac0908f35))
- **migrations:** Run otari's chain only against otari's own database in [#919](https://github.com/mozilla-ai/otari/pull/919) by [@khaledosman](https://github.com/khaledosman) ([`5018b64`](https://github.com/mozilla-ai/otari/commit/5018b6465269ffa72d71ef7590acbc5fc7cc5598))
- **api:** Refuse a blank provider narrowing on POST /v1/scoped-budgets in [#912](https://github.com/mozilla-ai/otari/pull/912) by [@champ18ion](https://github.com/champ18ion) ([`4d767d1`](https://github.com/mozilla-ai/otari/commit/4d767d1e3dc720253d9070c97911569a6944d62c))
- **messages:** Preserve Anthropic container field in [#916](https://github.com/mozilla-ai/otari/pull/916) by [@HareeshBahuleyan](https://github.com/HareeshBahuleyan) ([`1ca6ac4`](https://github.com/mozilla-ai/otari/commit/1ca6ac4951decdbbd5f2eced5227d36981e0bb7b))
- **usage:** Scope /v1/usage/count to imported rows for the select-all affordance in [#840](https://github.com/mozilla-ai/otari/pull/840) by [@AloysJehwin](https://github.com/AloysJehwin) ([`3a8500b`](https://github.com/mozilla-ai/otari/commit/3a8500b241c458d3617cc214b41fb768972d6ef4))
- **messages:** Adopt any-llm container support in [#938](https://github.com/mozilla-ai/otari/pull/938) by [@HareeshBahuleyan](https://github.com/HareeshBahuleyan) ([`70e3544`](https://github.com/mozilla-ai/otari/commit/70e35441c13141d9ea905e16afa0d4543786c3d0))
- **streaming:** Relax final forwarded-tool timeout in [#827](https://github.com/mozilla-ai/otari/pull/827) by [@HareeshBahuleyan](https://github.com/HareeshBahuleyan) ([`0edd6ab`](https://github.com/mozilla-ai/otari/commit/0edd6ab62fd1979d1774c4fd6eccb5d9d903e82b))
- **mcp:** Stop managed transport redirects in [#948](https://github.com/mozilla-ai/otari/pull/948) by [@HareeshBahuleyan](https://github.com/HareeshBahuleyan) ([`d894843`](https://github.com/mozilla-ai/otari/commit/d894843f72bc2b24c57575d80caa32d03572950c))
- **dashboard:** Correct two sidebar defects and inline the record copy control in [#956](https://github.com/mozilla-ai/otari/pull/956) by [@jigjigjig](https://github.com/jigjigjig) ([`4bf2a88`](https://github.com/mozilla-ai/otari/commit/4bf2a888b8df8dc9868f0dd259e4b7595969d187))
- **files:** Document binary download responses in [#949](https://github.com/mozilla-ai/otari/pull/949) by [@Ovvomii](https://github.com/Ovvomii) ([`d5d355b`](https://github.com/mozilla-ai/otari/commit/d5d355b39d168d3578efe08759e3073baf1f0672))


### Features

- **files:** Add S3-compatible object storage backend in [#411](https://github.com/mozilla-ai/otari/pull/411) by [@champ18ion](https://github.com/champ18ion) ([`3f4f8dc`](https://github.com/mozilla-ai/otari/commit/3f4f8dcea6f3c7a6f69fbe652676e058c4710965))
- **dashboard:** Bundle the operator user guide and serve it at /#/docs in [#434](https://github.com/mozilla-ai/otari/pull/434) by [@njbrake](https://github.com/njbrake) ([`8d1c1e9`](https://github.com/mozilla-ai/otari/commit/8d1c1e9f6910733e2b0ad6b3250912cf92244640))
- **auth:** Rate-limit dashboard sign-in against brute-force attempts in [#437](https://github.com/mozilla-ai/otari/pull/437) by [@champ18ion](https://github.com/champ18ion) ([`2657bc9`](https://github.com/mozilla-ai/otari/commit/2657bc94f9afe8c209aee7ca20cad7147823f123))
- **providers:** Add opt-in SSRF gate for provider api_base in [#441](https://github.com/mozilla-ai/otari/pull/441) by [@njbrake](https://github.com/njbrake) ([`a660931`](https://github.com/mozilla-ai/otari/commit/a660931cde009af4df6f2c9399c3df2cac0d3812))
- **web-search:** Support recency filtering and publish dates in Brave adapter in [#442](https://github.com/mozilla-ai/otari/pull/442) by [@champ18ion](https://github.com/champ18ion) ([`09fe531`](https://github.com/mozilla-ai/otari/commit/09fe531c817c700c1eab07b5cda64ad3b2ce3bb4))
- **dashboard:** Make the activity histogram the time-range selector in [#445](https://github.com/mozilla-ai/otari/pull/445) by [@njbrake](https://github.com/njbrake) ([`ffad6b5`](https://github.com/mozilla-ai/otari/commit/ffad6b57537d85ae756de29c86492eeaddf4dc13))
- **dashboard:** Install the dashboard to a home screen with the Otari icon in [#450](https://github.com/mozilla-ai/otari/pull/450) by [@njbrake](https://github.com/njbrake) ([`e646444`](https://github.com/mozilla-ai/otari/commit/e646444c8986abb2f27c54d96a28dcf9d5d958a7))
- **aliases:** Scope a model alias to a single user in [#455](https://github.com/mozilla-ai/otari/pull/455) by [@njbrake](https://github.com/njbrake) ([`dbbfebc`](https://github.com/mozilla-ai/otari/commit/dbbfebc19e76ee3878987025989c9437680333ea))
- **dashboard:** Surface failing requests so admins see dropped traffic in [#449](https://github.com/mozilla-ai/otari/pull/449) by [@njbrake](https://github.com/njbrake) ([`0b37186`](https://github.com/mozilla-ai/otari/commit/0b3718617a31738aa04763c54b7961af712fe9e7))
- **lint:** Enforce gateway layer rules via check_architecture.py in [#475](https://github.com/mozilla-ai/otari/pull/475) by [@peteski22](https://github.com/peteski22) ([`f348d60`](https://github.com/mozilla-ai/otari/commit/f348d606ace216b1f114685b8ac5fd0ec8859f4d))
- **observability:** Log the remaining gateway-side rejections in [#472](https://github.com/mozilla-ai/otari/pull/472) by [@njbrake](https://github.com/njbrake) ([`1a55d9f`](https://github.com/mozilla-ai/otari/commit/1a55d9f019eb383d4a035ad520436506ddcc9c7b))
- **usage:** Break spend down by session, endpoint, and provider in [#469](https://github.com/mozilla-ai/otari/pull/469) by [@njbrake](https://github.com/njbrake) ([`9d1b3e6`](https://github.com/mozilla-ai/otari/commit/9d1b3e6ec244f7b039863cde9b7073e0d166bb12))
- **search:** Add a billed POST /v1/search pass-through in [#473](https://github.com/mozilla-ai/otari/pull/473) by [@njbrake](https://github.com/njbrake) ([`0e508d9`](https://github.com/mozilla-ai/otari/commit/0e508d95c69cef63d5406aabc3be78e2072a8b92))
- **usage:** Record a status code on usage logs so failures can be classified in [#470](https://github.com/mozilla-ai/otari/pull/470) by [@njbrake](https://github.com/njbrake) ([`86e5353`](https://github.com/mozilla-ai/otari/commit/86e5353713b24bbce47f13f153539dc8c02eac1d))
- **dashboard:** Browsable activity log and rebuilt usage analytics in [#474](https://github.com/mozilla-ai/otari/pull/474) by [@njbrake](https://github.com/njbrake) ([`86bf391`](https://github.com/mozilla-ai/otari/commit/86bf3915b71381ea0cbf740617ef186784a92925))
- **dashboard:** Show the failure diagnostic and status code in the activity detail in [#482](https://github.com/mozilla-ai/otari/pull/482) by [@njbrake](https://github.com/njbrake) ([`8c35a87`](https://github.com/mozilla-ai/otari/commit/8c35a875382d1d0bd39c47c9a6445c4a8b03f4c7))
- **dashboard:** Make ids selectable and copyable in the tables in [#485](https://github.com/mozilla-ai/otari/pull/485) by [@njbrake](https://github.com/njbrake) ([`9f436e2`](https://github.com/mozilla-ai/otari/commit/9f436e2b70d0abc48ede55f2b6a4e684907c495b))
- **dashboard:** Price a model the catalogue does not list in [#490](https://github.com/mozilla-ai/otari/pull/490) by [@njbrake](https://github.com/njbrake) ([`4795754`](https://github.com/mozilla-ai/otari/commit/4795754813eaa86140a4d89187e40a6cfb0fe3f0))
- **routing:** Named routing policies with failover, budget tier-down, and enforced guardrails in [#492](https://github.com/mozilla-ai/otari/pull/492) by [@njbrake](https://github.com/njbrake) ([`e4ed919`](https://github.com/mozilla-ai/otari/commit/e4ed919922b312942af2ea2bde792489c04c853e))
- **hybrid:** Forward provider-specific extra_params through resolved attempts in [#496](https://github.com/mozilla-ai/otari/pull/496) by [@tbille](https://github.com/tbille) ([`0d78ff6`](https://github.com/mozilla-ai/otari/commit/0d78ff65effa7f33dfd50e3618ca25dbf53c9c8c))
- **tools:** Meter and bill gateway-run tool calls in [#504](https://github.com/mozilla-ai/otari/pull/504) by [@njbrake](https://github.com/njbrake) ([`d71a9ce`](https://github.com/mozilla-ai/otari/commit/d71a9cecab998398361f03ce5f96a1cb48149cd7))
- **dashboard:** Show what served a routed request in the activity log by [@njbrake](https://github.com/njbrake) ([`dbf6a51`](https://github.com/mozilla-ai/otari/commit/dbf6a51102ff821508c91583f4f07ce2fc951fd2))
- **messages:** Support Anthropic context compaction in [#512](https://github.com/mozilla-ai/otari/pull/512) by [@HareeshBahuleyan](https://github.com/HareeshBahuleyan) ([`726eb63`](https://github.com/mozilla-ai/otari/commit/726eb63714a0326101fb8f56709216296fca1e3f))
- **keys:** Add per-key ignore_user_mismatch opt-out in [#510](https://github.com/mozilla-ai/otari/pull/510) by [@njbrake](https://github.com/njbrake) ([`19266bd`](https://github.com/mozilla-ai/otari/commit/19266bdd3890b350338c7a0b51a2735636b206a2))
- **dashboard:** Edit provider client_args from the Providers page in [#520](https://github.com/mozilla-ai/otari/pull/520) by [@njbrake](https://github.com/njbrake) ([`d036f1b`](https://github.com/mozilla-ai/otari/commit/d036f1bb193c2d1c5672896037efdd1e52a041fb))
- **dashboard:** Filter usage views on several models, users, or keys in [#521](https://github.com/mozilla-ai/otari/pull/521) by [@njbrake](https://github.com/njbrake) ([`9ef21b1`](https://github.com/mozilla-ai/otari/commit/9ef21b1fb1cb9f8b995347ba785be0549e0e589c))
- **tools:** Accept natural web_search tool declarations and emit native result blocks in [#523](https://github.com/mozilla-ai/otari/pull/523) by [@njbrake](https://github.com/njbrake) ([`611fd62`](https://github.com/mozilla-ai/otari/commit/611fd62e6cc251f0a85a907190b868f5ef500ce4))
- **sdk-codegen:** Queue auto-merge on the regeneration PR in [#519](https://github.com/mozilla-ai/otari/pull/519) by [@njbrake](https://github.com/njbrake) ([`197fb33`](https://github.com/mozilla-ai/otari/commit/197fb338190eed9edb40faf6d9124c169cdb6d01))
- **streaming:** Send SSE keepalives while a provider stream is idle in [#528](https://github.com/mozilla-ai/otari/pull/528) by [@njbrake](https://github.com/njbrake) ([`d6d686b`](https://github.com/mozilla-ai/otari/commit/d6d686b229cb105b648f84edad7072f3a11e5922))
- **router:** Learned kNN router as a routing-policy backend in [#188](https://github.com/mozilla-ai/otari/pull/188) by [@njbrake](https://github.com/njbrake) ([`8b20687`](https://github.com/mozilla-ai/otari/commit/8b20687b89ab4cf0e33cddad3c7b69beefd0cdde))
- **responses:** Preserve context compaction through tool loops in [#513](https://github.com/mozilla-ai/otari/pull/513) by [@HareeshBahuleyan](https://github.com/HareeshBahuleyan) ([`1d8af00`](https://github.com/mozilla-ai/otari/commit/1d8af000f2e4736655dbabb31590568b11bc60cc))
- **code-execution:** Publish the wire contract and type the sandbox client in [#544](https://github.com/mozilla-ai/otari/pull/544) by [@khaledosman](https://github.com/khaledosman) ([`8b7e1d7`](https://github.com/mozilla-ai/otari/commit/8b7e1d72467d5503db7cbd603a422884775ab97f))
- **dashboard:** Show requests in flight on the activity page in [#549](https://github.com/mozilla-ai/otari/pull/549) by [@njbrake](https://github.com/njbrake) ([`4600ae4`](https://github.com/mozilla-ai/otari/commit/4600ae4b21b3421db3ab4bf106d4880fe13eccb0))
- **routing:** Weighted policy for load balancing across providers in [#553](https://github.com/mozilla-ai/otari/pull/553) by [@njbrake](https://github.com/njbrake) ([`709c794`](https://github.com/mozilla-ai/otari/commit/709c7947ebb25f1efde1c8eec55124fe3a639586))
- **dashboard:** Share the usage view as an image in [#555](https://github.com/mozilla-ai/otari/pull/555) by [@njbrake](https://github.com/njbrake) ([`d864e05`](https://github.com/mozilla-ai/otari/commit/d864e05edeee8a467384667f922a59470458c504))
- **otlp:** Capture coding-agent behavioral events into agent_telemetry in [#548](https://github.com/mozilla-ai/otari/pull/548) by [@arthuursantos](https://github.com/arthuursantos) ([`7bd6546`](https://github.com/mozilla-ai/otari/commit/7bd6546794bf47f82bbaefe39e7c633e0950bc29))
- **routing:** Rename policies in place, and stop them hiding candidate prices in [#560](https://github.com/mozilla-ai/otari/pull/560) by [@pss-julien](https://github.com/pss-julien) ([`e0db77d`](https://github.com/mozilla-ai/otari/commit/e0db77d272fdf5a60aa0cdfcbca0a74ee184cd9d))
- **telemetry:** Receive OTLP outcome metrics and read them per unit of spend in [#567](https://github.com/mozilla-ai/otari/pull/567) by [@arthuursantos](https://github.com/arthuursantos) ([`4a95bb5`](https://github.com/mozilla-ai/otari/commit/4a95bb53079ea9ca9a748ed8935bab4e7a42653b))
- **dashboard:** Stop the activity log refreshing itself in [#569](https://github.com/mozilla-ai/otari/pull/569) by [@njbrake](https://github.com/njbrake) ([`f32436c`](https://github.com/mozilla-ai/otari/commit/f32436cacc8441ee429b69daa61b7fb3d23b238f))
- **search:** Serve POST /v1/search from a SearXNG backend in [#598](https://github.com/mozilla-ai/otari/pull/598) by [@njbrake](https://github.com/njbrake) ([`8eca8f5`](https://github.com/mozilla-ai/otari/commit/8eca8f5fb2674d7e76f8e0c65c347271ec92160e))
- **code-execution:** Publish the OpenAPI IDL and a backend conformance check in [#578](https://github.com/mozilla-ai/otari/pull/578) by [@khaledosman](https://github.com/khaledosman) ([`8b9a326`](https://github.com/mozilla-ai/otari/commit/8b9a32614a78b0ffe98df3e32eafc09293bd1f6d))
- **search:** Manage search tools at runtime from the dashboard in [#604](https://github.com/mozilla-ai/otari/pull/604) by [@njbrake](https://github.com/njbrake) ([`af26c38`](https://github.com/mozilla-ai/otari/commit/af26c3842a2856d6d7a710b4e40c729d63f4a2a1))
- **scripts:** Enforce OSS/enterprise boundary in check_architecture.py in [#609](https://github.com/mozilla-ai/otari/pull/609) by [@tbille](https://github.com/tbille) ([`dfffeb3`](https://github.com/mozilla-ai/otari/commit/dfffeb326c906dcb6edce17fb8322dea7e14581c))
- **dashboard:** Select the runtime context from a deployment bootstrap in [#618](https://github.com/mozilla-ai/otari/pull/618) by [@njbrake](https://github.com/njbrake) ([`88c24ac`](https://github.com/mozilla-ai/otari/commit/88c24ac13098cf08037794c64710a13377498540))
- **dashboard:** Render the sidebar from a nav registry with an entitlement gate in [#619](https://github.com/mozilla-ai/otari/pull/619) by [@njbrake](https://github.com/njbrake) ([`492d156`](https://github.com/mozilla-ai/otari/commit/492d1562f2491baaa39ed56af6516b2235ad7931))
- **dashboard:** Rehome the shared design foundation from the platform UI in [#620](https://github.com/mozilla-ai/otari/pull/620) by [@njbrake](https://github.com/njbrake) ([`0fb1b24`](https://github.com/mozilla-ai/otari/commit/0fb1b24edef84617f12c12f1ac8996a5a4a5204a))
- **dashboard:** Rebuild every page on the shared design foundation in [#622](https://github.com/mozilla-ai/otari/pull/622) by [@njbrake](https://github.com/njbrake) ([`df9d01a`](https://github.com/mozilla-ai/otari/commit/df9d01a6967634f824e14be08303f3c164936299))
- **dashboard:** Boot a hybrid gateway into a data-plane landing page in [#623](https://github.com/mozilla-ai/otari/pull/623) by [@njbrake](https://github.com/njbrake) ([`7797e83`](https://github.com/mozilla-ai/otari/commit/7797e83c33b8421e72bce996cc10f95d69045eed))
- **hybrid:** Read the platform's optional caller-identity field in [#615](https://github.com/mozilla-ai/otari/pull/615) by [@tbille](https://github.com/tbille) ([`ad3efac`](https://github.com/mozilla-ai/otari/commit/ad3efacddc447f37bca061f99cfbc0bff98e7cce))
- **reprise:** Retain the resolved workspace and organization ids in [#628](https://github.com/mozilla-ai/otari/pull/628) by [@dpoulopoulos](https://github.com/dpoulopoulos) ([`bfaccef`](https://github.com/mozilla-ai/otari/commit/bfaccef83d94e55f32b2ac483be708e76ec09c4e))
- **tenancy:** Land the reconciled tenancy schema on an async repository base in [#624](https://github.com/mozilla-ai/otari/pull/624) by [@njbrake](https://github.com/njbrake) ([`3ef8ac8`](https://github.com/mozilla-ai/otari/commit/3ef8ac87d7adc3132d63645f79383289a43fbe69))
- **tenancy:** Serve organizations, workspaces, and memberships in [#625](https://github.com/mozilla-ai/otari/pull/625) by [@njbrake](https://github.com/njbrake) ([`9a32cc0`](https://github.com/mozilla-ai/otari/commit/9a32cc06143622bf8f643a75774272c2155ec21b))
- **tenancy:** Give the tenancy model a request plane in [#633](https://github.com/mozilla-ai/otari/pull/633) by [@njbrake](https://github.com/njbrake) ([`5d5ebff`](https://github.com/mozilla-ai/otari/commit/5d5ebffb3f2a9aac3f25da17b1c7285de14f1d8a))
- **dashboard:** Port the tenancy pages onto the rehomed control plane in [#626](https://github.com/mozilla-ai/otari/pull/626) by [@njbrake](https://github.com/njbrake) ([`4812bcf`](https://github.com/mozilla-ai/otari/commit/4812bcf472bfed8ed73e19dd9675e38aa0638aba))
- **tenancy:** Add the nullable credential columns to `user` in [#667](https://github.com/mozilla-ai/otari/pull/667) by [@njbrake](https://github.com/njbrake) ([`c9891ef`](https://github.com/mozilla-ai/otari/commit/c9891ef5fad0e81ccdd016ee7f6bda749b3f95af))
- **dashboard:** Port the nav label-override seam in [#664](https://github.com/mozilla-ai/otari/pull/664) by [@njbrake](https://github.com/njbrake) ([`b4c5861`](https://github.com/mozilla-ai/otari/commit/b4c5861fb3a273f59dcb276e97f88638c1d28359))
- **budgets:** Let a scoped budget reset on a UTC calendar boundary in [#640](https://github.com/mozilla-ai/otari/pull/640) by [@njbrake](https://github.com/njbrake) ([`d37fe93`](https://github.com/mozilla-ai/otari/commit/d37fe9333773b6c7e00be208b3b0da7a6341cfdf))
- **dashboard:** Bring the navigation shell onto the design in [#638](https://github.com/mozilla-ai/otari/pull/638) by [@jigjigjig](https://github.com/jigjigjig) ([`94ed29c`](https://github.com/mozilla-ai/otari/commit/94ed29c8bb10a66b01ee42b919c7e6e595ed83a8))
- **platform:** Return settled cost inline in [#675](https://github.com/mozilla-ai/otari/pull/675) by [@HareeshBahuleyan](https://github.com/HareeshBahuleyan) ([`169a1b4`](https://github.com/mozilla-ai/otari/commit/169a1b4987e062612af1a7edcbf458d959cb1a31))
- **pricing:** Per-organization model rate overrides in [#674](https://github.com/mozilla-ai/otari/pull/674) by [@khaledosman](https://github.com/khaledosman) ([`244bec5`](https://github.com/mozilla-ai/otari/commit/244bec565b3538ea597c6a5d0a1865b8387bb48b))
- **tenancy:** Invitations for organization and workspace membership in [#668](https://github.com/mozilla-ai/otari/pull/668) by [@tbille](https://github.com/tbille) ([`dbc3c69`](https://github.com/mozilla-ai/otari/commit/dbc3c69371817275f14752a5cfc98f0e771d7308))
- **dashboard:** Give dashboard sessions a user identity in [#681](https://github.com/mozilla-ai/otari/pull/681) by [@njbrake](https://github.com/njbrake) ([`80f4074`](https://github.com/mozilla-ai/otari/commit/80f40743d0384ffb4c12e801f74c793d43def266))
- **tenancy:** Workspace per-member budget defaults in [#671](https://github.com/mozilla-ai/otari/pull/671) by [@tbille](https://github.com/tbille) ([`4407789`](https://github.com/mozilla-ai/otari/commit/4407789772fdf01e5411ef20cd50b144d55d4252))
- **provider-keys:** Add organization-scoped BYO provider keys in [#670](https://github.com/mozilla-ai/otari/pull/670) by [@tbille](https://github.com/tbille) ([`a82cd29`](https://github.com/mozilla-ai/otari/commit/a82cd29356147b2c8a789f61083333a83708f0b5))
- **mail:** Make outgoing mail a transport with an honest no-transport mode by [@njbrake](https://github.com/njbrake) ([`51a6787`](https://github.com/mozilla-ai/otari/commit/51a67873fc2adb8a1a70297631378e069ec3697f))
- **auth:** Password sign-in, and the master key retires as the dashboard login in [#684](https://github.com/mozilla-ai/otari/pull/684) by [@njbrake](https://github.com/njbrake) ([`015a8ee`](https://github.com/mozilla-ai/otari/commit/015a8eefd75d888cfaa5cb760d575e683deaccea))
- **dashboard:** Give the dashboard password a page, and open the account menu's row by [@khaledosman](https://github.com/khaledosman) ([`4b9fef7`](https://github.com/mozilla-ai/otari/commit/4b9fef75c0a8a2135acfcd19c9f374d2ab02b1cc))
- **activation:** Guide a new workspace to its first successful request in [#705](https://github.com/mozilla-ai/otari/pull/705) by [@khaledosman](https://github.com/khaledosman) ([`c2dfa30`](https://github.com/mozilla-ai/otari/commit/c2dfa30691b7d50662a017020300cd75756abe4d))
- **auth:** Signup, email verification, and password reset in [#708](https://github.com/mozilla-ai/otari/pull/708) by [@tbille](https://github.com/tbille) ([`754f830`](https://github.com/mozilla-ai/otari/commit/754f830a805e80bce398e74a1746f37e67bc7df1))
- **dashboard:** One budgets page, and a person is a member in [#711](https://github.com/mozilla-ai/otari/pull/711) by [@njbrake](https://github.com/njbrake) ([`40e9c90`](https://github.com/mozilla-ai/otari/commit/40e9c90865cdeed55e7f5c7b854bdd07b7fcdc29))
- **dashboard:** Give signup, verification, and password reset their pages in [#726](https://github.com/mozilla-ai/otari/pull/726) by [@njbrake](https://github.com/njbrake) ([`34741ae`](https://github.com/mozilla-ai/otari/commit/34741ae9dd4fa133d9591dbe62fd7ee63cb9c930))
- **budgets:** Make the spend ledger exact, like the settled rows feeding it in [#722](https://github.com/mozilla-ai/otari/pull/722) by [@njbrake](https://github.com/njbrake) ([`d2ba90e`](https://github.com/mozilla-ai/otari/commit/d2ba90ea2d9d284331ff195a66318b57c621656b))
- **usage:** Record the cached-token convention on usage_logs in [#723](https://github.com/mozilla-ai/otari/pull/723) by [@njbrake](https://github.com/njbrake) ([`d80beab`](https://github.com/mozilla-ai/otari/commit/d80beabcbcf3c5f729c3ec31087f27b6bb5a9a05))
- **mcp:** Let a workspace configure its own MCP servers in [#724](https://github.com/mozilla-ai/otari/pull/724) by [@njbrake](https://github.com/njbrake) ([`a54acfe`](https://github.com/mozilla-ai/otari/commit/a54acfeadc78a83daea710fddb1c784f0776dbef))
- **gateway:** Give a workspace a code-execution policy over the deployment sandbox in [#725](https://github.com/mozilla-ai/otari/pull/725) by [@njbrake](https://github.com/njbrake) ([`0e2634c`](https://github.com/mozilla-ai/otari/commit/0e2634c37d278bdb4a85645a492ef559d93ed2e5))
- **budgets:** Close the last two gaps in workspace budget defaults in [#745](https://github.com/mozilla-ai/otari/pull/745) by [@njbrake](https://github.com/njbrake) ([`86fff80`](https://github.com/mozilla-ai/otari/commit/86fff80769478cffd61563fe20f588af66d5001b))
- **dashboard:** Add the intra-page slot seam for the top bar's balance in [#748](https://github.com/mozilla-ai/otari/pull/748) by [@njbrake](https://github.com/njbrake) ([`27c0a84`](https://github.com/mozilla-ai/otari/commit/27c0a84a69512630ce70909ce3ce8648f53862fa))
- **gateway:** Freeze dashboard sign-ins with a maintenance-mode switch in [#749](https://github.com/mozilla-ai/otari/pull/749) by [@njbrake](https://github.com/njbrake) ([`4cb3a34`](https://github.com/mozilla-ai/otari/commit/4cb3a342a35013b4c2d2ead926de1eddefae8fef))
- **gateway:** Let a workspace configure its own web search in [#744](https://github.com/mozilla-ai/otari/pull/744) by [@njbrake](https://github.com/njbrake) ([`62a25c7`](https://github.com/mozilla-ai/otari/commit/62a25c70472b3b73049a934c8c790608cb492e54))
- **dashboard:** Add a no-op telemetry seam an overlay build can replace in [#754](https://github.com/mozilla-ai/otari/pull/754) by [@njbrake](https://github.com/njbrake) ([`c3a0343`](https://github.com/mozilla-ai/otari/commit/c3a0343500ecd751c29f32a7e71b04ece744868e))
- **architecture:** Land the ports package, composition container, and bootstrap hook in [#756](https://github.com/mozilla-ai/otari/pull/756) by [@njbrake](https://github.com/njbrake) ([`150c492`](https://github.com/mozilla-ai/otari/commit/150c4929cc2267643df4700e05d9eb7e992eaef3))
- Register and sign in with passkeys in [#751](https://github.com/mozilla-ai/otari/pull/751) by [@khaledosman](https://github.com/khaledosman) ([`cf3de46`](https://github.com/mozilla-ai/otari/commit/cf3de46760db2ba4bb5403ddcd5a36cb22f734f7))
- **guardrails:** Let an organization mandate guardrails over its workspaces in [#752](https://github.com/mozilla-ai/otari/pull/752) by [@njbrake](https://github.com/njbrake) ([`698be2b`](https://github.com/mozilla-ai/otari/commit/698be2bde6d003d97d6ed566a1d704db49233b59))
- **architecture:** Resolve captured telemetry through a storage port in [#770](https://github.com/mozilla-ai/otari/pull/770) by [@khaledosman](https://github.com/khaledosman) ([`c68f3cd`](https://github.com/mozilla-ai/otari/commit/c68f3cd18c1181c0bfc607272d74a957b18b66e7))
- **dashboard:** Let a deployment point the Documentation links at its own docs site in [#774](https://github.com/mozilla-ai/otari/pull/774) by [@khaledosman](https://github.com/khaledosman) ([`ab25543`](https://github.com/mozilla-ai/otari/commit/ab255433065a71a5581ddef87b78432a1f8968d8))
- **usage:** Add pricing-provenance columns to usage_logs in [#775](https://github.com/mozilla-ai/otari/pull/775) by [@khaledosman](https://github.com/khaledosman) ([`c0f1fc6`](https://github.com/mozilla-ai/otari/commit/c0f1fc69be029a496639dc7cbacdd255a6259fb6))
- **tenancy:** Create an organization, and switch the active one in [#771](https://github.com/mozilla-ai/otari/pull/771) by [@khaledosman](https://github.com/khaledosman) ([`c33d68c`](https://github.com/mozilla-ai/otari/commit/c33d68c6dc0f9d445be6c37d8631800d5ce99772))
- **dashboard:** Add a seam an overlay resolves entitlements through in [#759](https://github.com/mozilla-ai/otari/pull/759) by [@njbrake](https://github.com/njbrake) ([`7f00911`](https://github.com/mozilla-ai/otari/commit/7f009110a615191cfe43fd9e6063602775deddfb))
- **gateway:** Price imported usage at organization rates in [#761](https://github.com/mozilla-ai/otari/pull/761) by [@njbrake](https://github.com/njbrake) ([`e42afdf`](https://github.com/mozilla-ai/otari/commit/e42afdfcfcb50b1382b07594cdd01bb17039b105))
- **dashboard:** Manage a workspace's MCP servers from Tools in [#760](https://github.com/mozilla-ai/otari/pull/760) by [@njbrake](https://github.com/njbrake) ([`281353e`](https://github.com/mozilla-ai/otari/commit/281353ef573afddb2dbda084ba5802607b708d89))
- **gateway:** Consult ModelProviderPort when no stored credential serves a candidate in [#762](https://github.com/mozilla-ai/otari/pull/762) by [@njbrake](https://github.com/njbrake) ([`26dda60`](https://github.com/mozilla-ai/otari/commit/26dda6028149baf26871272899678a44dbef39d6))
- **tools:** Let a workspace pin its sandbox image and exposed tool set by [@njbrake](https://github.com/njbrake) ([`c3a2335`](https://github.com/mozilla-ai/otari/commit/c3a2335dbf6f847146765076ba2d9d2eb400c2d1))
- Sign in with Google and GitHub, behind a real IdentityProviderPort in [#765](https://github.com/mozilla-ai/otari/pull/765) by [@njbrake](https://github.com/njbrake) ([`cc693e0`](https://github.com/mozilla-ai/otari/commit/cc693e0999dccf6c256cca41f38a44c4adabd06c))
- **budgets:** Give reservations a ledger in [#767](https://github.com/mozilla-ai/otari/pull/767) by [@njbrake](https://github.com/njbrake) ([`4a2ee91`](https://github.com/mozilla-ai/otari/commit/4a2ee9123c3181460f7512ddff4278c2e767d309))
- **tenancy:** Scope the gateway survivals to a workspace in [#784](https://github.com/mozilla-ai/otari/pull/784) by [@khaledosman](https://github.com/khaledosman) ([`a10d46f`](https://github.com/mozilla-ai/otari/commit/a10d46f1cd0f071f8bdfec767cef0b6e6f138359))
- **dashboard:** Give the shell a seam an overlay mounts a post-sign-in step in in [#793](https://github.com/mozilla-ai/otari/pull/793) by [@njbrake](https://github.com/njbrake) ([`124565e`](https://github.com/mozilla-ai/otari/commit/124565eacabdcdf001928259564a6688f1731dbf))
- **tenancy:** Give an operator a surface over the deployment's accounts in [#798](https://github.com/mozilla-ai/otari/pull/798) by [@njbrake](https://github.com/njbrake) ([`065fe51`](https://github.com/mozilla-ai/otari/commit/065fe5119188916683b5fbf80bb9ca94913fbff9))
- **dashboard:** Add a trend chip for period-over-period change in [#801](https://github.com/mozilla-ai/otari/pull/801) by [@jigjigjig](https://github.com/jigjigjig) ([`66b6537`](https://github.com/mozilla-ai/otari/commit/66b6537cca4667bea073e09083b6155227967aa2))
- **dashboard:** Render usage deltas as trend chips in [#815](https://github.com/mozilla-ai/otari/pull/815) by [@jigjigjig](https://github.com/jigjigjig) ([`4cd3f5c`](https://github.com/mozilla-ai/otari/commit/4cd3f5c6dde17e6a8a22a2e643564be78215e19a))
- **dashboard:** Land the organization provider-keys page and a hosted surface set in [#820](https://github.com/mozilla-ai/otari/pull/820) by [@njbrake](https://github.com/njbrake) ([`0dd3d6c`](https://github.com/mozilla-ai/otari/commit/0dd3d6c67a42e8f5bea95ab7240bfa80a253c6b0))
- **dashboard:** Finish the trend-chip migration on the overview page in [#830](https://github.com/mozilla-ai/otari/pull/830) by [@jigjigjig](https://github.com/jigjigjig) ([`240b732`](https://github.com/mozilla-ai/otari/commit/240b732dfdd98d062e9c30453021d34f37acd763))
- **api:** Give a tenant an organization-scoped read of their own usage in [#842](https://github.com/mozilla-ai/otari/pull/842) by [@njbrake](https://github.com/njbrake) ([`ec0d1c7`](https://github.com/mozilla-ai/otari/commit/ec0d1c7b9bd3caeecc8b99ca4bfccc1d86647e7f))
- **BREAKING:** **api:** Remove GET /v1/usage/summary.csv in [#854](https://github.com/mozilla-ai/otari/pull/854) by [@njbrake](https://github.com/njbrake) ([`884221d`](https://github.com/mozilla-ai/otari/commit/884221d8cb510b697521e1da0d6d0af7beea7fe8))
- **dashboard:** Give members a read-only view of the Build pages in [#867](https://github.com/mozilla-ai/otari/pull/867) by [@khaledosman](https://github.com/khaledosman) ([`b25554a`](https://github.com/mozilla-ai/otari/commit/b25554aa6a67daa88528a3aeb3350be93cf3707a))
- **dashboard:** Publish the deployment's legal page addresses in [#870](https://github.com/mozilla-ai/otari/pull/870) by [@njbrake](https://github.com/njbrake) ([`a40a1ca`](https://github.com/mozilla-ai/otari/commit/a40a1ca7f9370077f18de3e3beda5dce6af20468))
- **keys:** Let members create and manage their own API keys in [#866](https://github.com/mozilla-ai/otari/pull/866) by [@khaledosman](https://github.com/khaledosman) ([`75bbf68`](https://github.com/mozilla-ai/otari/commit/75bbf68a84bfccfdb5b78bb308fb98dfb412f60f))
- **budgets:** Give an organization its own budgets and spend ceilings in [#875](https://github.com/mozilla-ai/otari/pull/875) by [@khaledosman](https://github.com/khaledosman) ([`aee4cca`](https://github.com/mozilla-ai/otari/commit/aee4cca97d400cdc44738ad5772215b1b64002d4))
- **growth:** Resolve GrowthSignalPort at signup and a member's first key in [#884](https://github.com/mozilla-ai/otari/pull/884) by [@njbrake](https://github.com/njbrake) ([`aa58b5a`](https://github.com/mozilla-ai/otari/commit/aa58b5ab8a34cb661dab37199c0df33ae778d66f))
- **web-search:** Call Tavily and Brave in-process, and serve them to a data plane in [#882](https://github.com/mozilla-ai/otari/pull/882) by [@njbrake](https://github.com/njbrake) ([`466284c`](https://github.com/mozilla-ai/otari/commit/466284c144a1eb6b096c05643179005ffa109ca7))
- **dashboard:** Give a member their organization's roster in [#893](https://github.com/mozilla-ai/otari/pull/893) by [@khaledosman](https://github.com/khaledosman) ([`feae11f`](https://github.com/mozilla-ai/otari/commit/feae11f840bf6c29839ed31b2e0e3aaf0aa19662))
- **dashboard:** Add an organization-wide usage page for admins in [#897](https://github.com/mozilla-ai/otari/pull/897) by [@khaledosman](https://github.com/khaledosman) ([`196519b`](https://github.com/mozilla-ai/otari/commit/196519bb9a7935fd5b6d70d0d66f00d6c201356e))
- **dashboard:** Give organization admins Edit on the Build pages in [#889](https://github.com/mozilla-ai/otari/pull/889) by [@khaledosman](https://github.com/khaledosman) ([`ea8c750`](https://github.com/mozilla-ai/otari/commit/ea8c7508aa0f4d2c6950479a852efaac6340a8ce))
- **dashboard:** Name the setup-guide step in the telemetry vocabulary in [#908](https://github.com/mozilla-ai/otari/pull/908) by [@khaledosman](https://github.com/khaledosman) ([`bddcc53`](https://github.com/mozilla-ai/otari/commit/bddcc53d0986334070d30ff6cc47a67086ac6533))
- **budgets:** Cap tokens and requests, not dollars alone in [#906](https://github.com/mozilla-ai/otari/pull/906) by [@khaledosman](https://github.com/khaledosman) ([`01bdd8b`](https://github.com/mozilla-ai/otari/commit/01bdd8b7489d51466a9213fc12082adbfe95f50c))
- **tenancy:** Give the invitee a membership inbox to list, accept and decline in [#914](https://github.com/mozilla-ai/otari/pull/914) by [@khaledosman](https://github.com/khaledosman) ([`f2d8363`](https://github.com/mozilla-ai/otari/commit/f2d836377ae7f5cee5bbb0d481de6de8ee105691))
- **tenancy:** Email-domain auto-join, claimed and proven by DNS in [#915](https://github.com/mozilla-ai/otari/pull/915) by [@khaledosman](https://github.com/khaledosman) ([`10c4e6c`](https://github.com/mozilla-ai/otari/commit/10c4e6cfcb8bf9c9c110ed171f7ca5f7a9f78c03))
- **api:** Expose managed MCP activity in Messages streams in [#795](https://github.com/mozilla-ai/otari/pull/795) by [@HareeshBahuleyan](https://github.com/HareeshBahuleyan) ([`9930c86`](https://github.com/mozilla-ai/otari/commit/9930c865e390d3dce4e96a5b57f6e3873c648ef4))
- **dashboard:** Redesign in [#934](https://github.com/mozilla-ai/otari/pull/934) by [@jigjigjig](https://github.com/jigjigjig) ([`12e6770`](https://github.com/mozilla-ai/otari/commit/12e67707fdcbffec5f1bcb1ceea25fd511da5f77))


### Other

- Disable auto_assign_reviewers in configuration in [#551](https://github.com/mozilla-ai/otari/pull/551) by [@njbrake](https://github.com/njbrake) ([`25a9e37`](https://github.com/mozilla-ai/otari/commit/25a9e37ab072b84db1d5f598daf6d49ef5e3c244))


### Performance

- **dashboard:** Take provider dials and whole-table reads off page loads in [#535](https://github.com/mozilla-ai/otari/pull/535) by [@njbrake](https://github.com/njbrake) ([`4b63dce`](https://github.com/mozilla-ai/otari/commit/4b63dce66bccf264b1645801725ca918eade0079))



### New Contributors

- [@Ovvomii](https://github.com/Ovvomii) made their first contribution in [#949](https://github.com/mozilla-ai/otari/pull/949)
- [@jigjigjig](https://github.com/jigjigjig) made their first contribution in [#956](https://github.com/mozilla-ai/otari/pull/956)
- [@HareeshBahuleyan](https://github.com/HareeshBahuleyan) made their first contribution in [#948](https://github.com/mozilla-ai/otari/pull/948)
- [@AmirF194](https://github.com/AmirF194) made their first contribution in [#792](https://github.com/mozilla-ai/otari/pull/792)
- [@shoemoney](https://github.com/shoemoney) made their first contribution in [#730](https://github.com/mozilla-ai/otari/pull/730)
- [@dpoulopoulos](https://github.com/dpoulopoulos) made their first contribution in [#628](https://github.com/mozilla-ai/otari/pull/628)
- [@Copilot](https://github.com/Copilot) made their first contribution in [#575](https://github.com/mozilla-ai/otari/pull/575)
- [@arthuursantos](https://github.com/arthuursantos) made their first contribution in [#567](https://github.com/mozilla-ai/otari/pull/567)
- [@pss-julien](https://github.com/pss-julien) made their first contribution in [#562](https://github.com/mozilla-ai/otari/pull/562)

**Full Changelog**: https://github.com/mozilla-ai/otari/compare/v0.4.0...v0.5.0
## [0.4.0](https://github.com/mozilla-ai/otari/releases/tag/v0.4.0) - 2026-07-24



### Bug Fixes

- **dashboard:** Align input boxes on the tools & guardrails page in [#383](https://github.com/mozilla-ai/otari/pull/383) by [@njbrake](https://github.com/njbrake) ([`fa68234`](https://github.com/mozilla-ai/otari/commit/fa682347676832ac59535e0ed95cc7d14519ff55))
- **models:** List env-credentialed providers in /v1/models in [#388](https://github.com/mozilla-ai/otari/pull/388) by [@njbrake](https://github.com/njbrake) ([`340bd4b`](https://github.com/mozilla-ai/otari/commit/340bd4bba2d68de38eda61d08086ea935e342427))
- Surface actionable tools + reasoning_effort provider error in [#385](https://github.com/mozilla-ai/otari/pull/385) by [@njbrake](https://github.com/njbrake) ([`47c142c`](https://github.com/mozilla-ai/otari/commit/47c142cfccfd7335ae91d422fa0a10ac2c6be596))
- **dashboard:** Make the provider picker open instantly (lazy split) in [#382](https://github.com/mozilla-ai/otari/pull/382) by [@njbrake](https://github.com/njbrake) ([`e16e49f`](https://github.com/mozilla-ai/otari/commit/e16e49fd6328b5b69f75b2f9bba85738e1a01b5f))
- **dashboard:** Persist sign-in with an HttpOnly session cookie in [#384](https://github.com/mozilla-ai/otari/pull/384) by [@njbrake](https://github.com/njbrake) ([`f462b03`](https://github.com/mozilla-ai/otari/commit/f462b03e2832c9ca6345a3c9e2039ca884d7e40e))
- **dashboard:** Make the dashboard mobile friendly in [#406](https://github.com/mozilla-ai/otari/pull/406) by [@njbrake](https://github.com/njbrake) ([`819681d`](https://github.com/mozilla-ai/otari/commit/819681d91c1aa7b4026e119b9a0b799014a2ed2c))
- **hybrid:** Recognize SDK-wrapped timeout/connection errors for fallback in [#381](https://github.com/mozilla-ai/otari/pull/381) by [@tbille](https://github.com/tbille) ([`c463603`](https://github.com/mozilla-ai/otari/commit/c4636033ec0c2abb990324ad04befd0f9f159e91))
- **dashboard:** CodeRabbit follow-ups on budgets, usage, and tool tests in [#404](https://github.com/mozilla-ai/otari/pull/404) by [@njbrake](https://github.com/njbrake) ([`6a8a165`](https://github.com/mozilla-ai/otari/commit/6a8a165c1f3356e9c98cfe416089a8a6a19b10c7))
- **compose:** Bind quickstart postgres to localhost in [#413](https://github.com/mozilla-ai/otari/pull/413) by [@pcornelissen](https://github.com/pcornelissen) ([`a5e8058`](https://github.com/mozilla-ai/otari/commit/a5e80585736788ff89e54add5cccf3185ce8cc73))
- **dashboard:** Stop table selection clicks from re-rendering every row in [#416](https://github.com/mozilla-ai/otari/pull/416) by [@njbrake](https://github.com/njbrake) ([`39428aa`](https://github.com/mozilla-ai/otari/commit/39428aa8bcdc5ae94591702cdf89d7786f7dda30))
- **dashboard:** Reopen the request detail inline under the clicked activity row in [#417](https://github.com/mozilla-ai/otari/pull/417) by [@njbrake](https://github.com/njbrake) ([`4685d03`](https://github.com/mozilla-ai/otari/commit/4685d03698778dac832b9081474717fbaeec2a80))
- **dashboard:** Wrap usage stat tiles into a responsive grid on mobile in [#418](https://github.com/mozilla-ai/otari/pull/418) by [@njbrake](https://github.com/njbrake) ([`50b0cdd`](https://github.com/mozilla-ai/otari/commit/50b0cdd4cb318325aae62bcc3164cdbc70ef7eac))


### Features

- **dashboard:** Code-split route bundles in [#373](https://github.com/mozilla-ai/otari/pull/373) by [@njbrake](https://github.com/njbrake) ([`b04d3cd`](https://github.com/mozilla-ai/otari/commit/b04d3cdce0c66eed34e39a9efdb0a44b1850e1c0))
- **pricing:** Add reviewable price refresh in [#367](https://github.com/mozilla-ai/otari/pull/367) by [@njbrake](https://github.com/njbrake) ([`e5bdbb8`](https://github.com/mozilla-ai/otari/commit/e5bdbb80c595f5765b69f26f93dba751f0df0b9b))
- **pricing:** Meter cache writes and long context in [#379](https://github.com/mozilla-ai/otari/pull/379) by [@njbrake](https://github.com/njbrake) ([`7559257`](https://github.com/mozilla-ai/otari/commit/755925776ef03c3818668bca274ca4341b40ca33))
- **files:** Stream uploads and downloads instead of buffering in memory in [#380](https://github.com/mozilla-ai/otari/pull/380) by [@champ18ion](https://github.com/champ18ion) ([`cbee20e`](https://github.com/mozilla-ai/otari/commit/cbee20ebbd2db5bd5cb8d6cfc3af5927d6355cf6))
- **dashboard:** Adopt recharts and retire hand-rolled UsageChart in [#386](https://github.com/mozilla-ai/otari/pull/386) by [@njbrake](https://github.com/njbrake) ([`de90d8a`](https://github.com/mozilla-ai/otari/commit/de90d8a21c022922bbe6a9159ccdb319b2e9bddd))
- **dashboard:** Disable adding providers when OTARI_SECRET_KEY is unset in [#391](https://github.com/mozilla-ai/otari/pull/391) by [@njbrake](https://github.com/njbrake) ([`661aafc`](https://github.com/mozilla-ai/otari/commit/661aafc8c79a6aad9b6cc627e28471170782e385))
- **usage:** Ingest external usage events for API-equivalent cost tracking in [#392](https://github.com/mozilla-ai/otari/pull/392) by [@njbrake](https://github.com/njbrake) ([`bf90e94`](https://github.com/mozilla-ai/otari/commit/bf90e94a1c09cf6acbeee242a116c5666a6c8a48))
- **dashboard:** Shared table framework with filtering, pagination, selection, and bulk actions in [#415](https://github.com/mozilla-ai/otari/pull/415) by [@njbrake](https://github.com/njbrake) ([`c8e09b3`](https://github.com/mozilla-ai/otari/commit/c8e09b3b9a3b66e2ce86a9cc6a9109ed61034e53))


### Other

- Update README to remove GitHub repository link in [#394](https://github.com/mozilla-ai/otari/pull/394) by [@njbrake](https://github.com/njbrake) ([`ac51544`](https://github.com/mozilla-ai/otari/commit/ac5154469f731bfd2d10711b68ed2fdc6b094c57))



### New Contributors

- [@pcornelissen](https://github.com/pcornelissen) made their first contribution in [#413](https://github.com/mozilla-ai/otari/pull/413)
- [@champ18ion](https://github.com/champ18ion) made their first contribution in [#380](https://github.com/mozilla-ai/otari/pull/380)

**Full Changelog**: https://github.com/mozilla-ai/otari/compare/v0.3.0...v0.4.0
## [0.3.0](https://github.com/mozilla-ai/otari/releases/tag/v0.3.0) - 2026-07-22



### Bug Fixes

- **openapi:** Add ApiKeyAuth security scheme to schema definition in [#248](https://github.com/mozilla-ai/otari/pull/248) by [@Sourav-Tripathy](https://github.com/Sourav-Tripathy) ([`ca5608c`](https://github.com/mozilla-ai/otari/commit/ca5608cbf819b147a0050c342d9f9f33ad5d0839))
- Fixing the render deploy url in [#257](https://github.com/mozilla-ai/otari/pull/257) by [@ojusave](https://github.com/ojusave) ([`d7fd902`](https://github.com/mozilla-ai/otari/commit/d7fd902623e90ba64c567a89a0db5927749e1339))
- **sandbox:** Give exec POST its own timeout above the execution budget in [#279](https://github.com/mozilla-ai/otari/pull/279) by [@njbrake](https://github.com/njbrake) ([`d7ed152`](https://github.com/mozilla-ai/otari/commit/d7ed152eb8f98c90c5478bfb8ff5d1310c087f27))
- **cli:** Honor --log-level names and reject unsupported --workers in [#284](https://github.com/mozilla-ai/otari/pull/284) by [@njbrake](https://github.com/njbrake) ([`36893fc`](https://github.com/mozilla-ai/otari/commit/36893fca684a2deb12ed75e85289e8c4da3e7b6d))
- **metrics:** Label requests by route template, not raw path in [#274](https://github.com/mozilla-ai/otari/pull/274) by [@njbrake](https://github.com/njbrake) ([`a147e42`](https://github.com/mozilla-ai/otari/commit/a147e426aa4ff12ae6636e47d479f5a08af0089a))
- **config:** Honor explicit OTARI_MODE instead of deriving from token alone in [#276](https://github.com/mozilla-ai/otari/pull/276) by [@njbrake](https://github.com/njbrake) ([`2314b23`](https://github.com/mozilla-ai/otari/commit/2314b23012f44b341f8dd5ef7bbb4079a6041bc2))
- **batches:** Enforce user resolution, budget, rate limiting, and batch ownership in [#273](https://github.com/mozilla-ai/otari/pull/273) by [@njbrake](https://github.com/njbrake) ([`3f2c23e`](https://github.com/mozilla-ai/otari/commit/3f2c23efe250de8e1c0b3c03da56e733f8d03a25))
- **usage:** Bill vision side-call before top-up and track usage report tasks in [#277](https://github.com/mozilla-ai/otari/pull/277) by [@njbrake](https://github.com/njbrake) ([`81b0677`](https://github.com/mozilla-ai/otari/commit/81b06774d27aab25107ced9de6f0cf15f3cee4c2))
- **api:** Unify error handling and hybrid streaming fallback across completion routes in [#281](https://github.com/mozilla-ai/otari/pull/281) by [@njbrake](https://github.com/njbrake) ([`ff6acb1`](https://github.com/mozilla-ai/otari/commit/ff6acb1486aa1a2f68a081fa1fd34bbb8aa8d1a5))
- **deps:** Bump any-llm to 1.21.0 to meter streaming /v1/messages usage in [#292](https://github.com/mozilla-ai/otari/pull/292) by [@njbrake](https://github.com/njbrake) ([`01335b2`](https://github.com/mozilla-ai/otari/commit/01335b29c43cad5acdea7375c5564e38d633db29))
- Return 400 for unresolvable model selectors in [#254](https://github.com/mozilla-ai/otari/pull/254) by [@AloysJehwin](https://github.com/AloysJehwin) ([`b07a37c`](https://github.com/mozilla-ai/otari/commit/b07a37c8b9208b409b4b3a9411c4f96812024fb7))
- **config:** Reject invalid OTARI_SECRET_KEY at startup in [#325](https://github.com/mozilla-ai/otari/pull/325) by [@njbrake](https://github.com/njbrake) ([`e2a8cbb`](https://github.com/mozilla-ai/otari/commit/e2a8cbb687060e82bbe443a1dbc48aa724a637ff))
- **observability:** Don't count locked-in tool-loop failures as abandoned in [#326](https://github.com/mozilla-ai/otari/pull/326) by [@njbrake](https://github.com/njbrake) ([`48e3e1d`](https://github.com/mozilla-ai/otari/commit/48e3e1df9132793564c89315bda3fda608b6ea6e))
- **auth:** Accept a raw token on the Otari-Key header, not just Bearer in [#323](https://github.com/mozilla-ai/otari/pull/323) by [@njbrake](https://github.com/njbrake) ([`794eeef`](https://github.com/mozilla-ai/otari/commit/794eeef14bc5e5f6bcc90a4f0dd9e03ed4e220fa))
- **models:** Bound provider discovery, cache failures, single-flight in [#342](https://github.com/mozilla-ai/otari/pull/342) by [@njbrake](https://github.com/njbrake) ([`7c5c79c`](https://github.com/mozilla-ai/otari/commit/7c5c79ca4164368a887c8fc8c8f48d4dca5b4165))
- **providers:** Make API key optional when the provider env var is set in [#344](https://github.com/mozilla-ai/otari/pull/344) by [@njbrake](https://github.com/njbrake) ([`f7b96c5`](https://github.com/mozilla-ai/otari/commit/f7b96c5358f3e4430c5b2796ec9af9f37d0f6b8f))
- Preserve Codex Responses metadata for OpenAI in [#364](https://github.com/mozilla-ai/otari/pull/364) by [@njbrake](https://github.com/njbrake) ([`7915544`](https://github.com/mozilla-ai/otari/commit/7915544602572b4b0b8a9ebf3b1bacc5fbe2f089))
- **dashboard:** Limit automatic provider health checks in [#363](https://github.com/mozilla-ai/otari/pull/363) by [@njbrake](https://github.com/njbrake) ([`e114665`](https://github.com/mozilla-ai/otari/commit/e114665c2988b1d2fffc8bc9ae620ab9f51b5ef1))
- Require any-llm 1.22.1 in [#369](https://github.com/mozilla-ai/otari/pull/369) by [@njbrake](https://github.com/njbrake) ([`3711f4c`](https://github.com/mozilla-ai/otari/commit/3711f4c45e01291e9f6de8eacd221e2e5f09f91d))
- **dashboard:** Streamline first-run navigation in [#368](https://github.com/mozilla-ai/otari/pull/368) by [@njbrake](https://github.com/njbrake) ([`f421a93`](https://github.com/mozilla-ai/otari/commit/f421a9328be333ec06f6b8c8a39252a52b79c6e5))


### Features

- Enforce "PERF" ruff rules by [@MooseTheRebel](https://github.com/MooseTheRebel) ([`6547fe8`](https://github.com/mozilla-ai/otari/commit/6547fe89f7b1647e4acfc7fe1d4b605725045365))
- Enforce ruff PERF rules across codebase in [#245](https://github.com/mozilla-ai/otari/pull/245) by [@MooseTheRebel](https://github.com/MooseTheRebel) ([`dd8b71d`](https://github.com/mozilla-ai/otari/commit/dd8b71d4d55129f56a73a2161c72b38c6ed939fd))
- **models:** Config-level model aliases that map to real selectors in [#229](https://github.com/mozilla-ai/otari/pull/229) by [@njbrake](https://github.com/njbrake) ([`ed1a531`](https://github.com/mozilla-ai/otari/commit/ed1a531454bd109af0fe3793bd6e1a3049ae415b))
- Add API key rotation endpoint in [#285](https://github.com/mozilla-ai/otari/pull/285) by [@njbrake](https://github.com/njbrake) ([`2756f08`](https://github.com/mozilla-ai/otari/commit/2756f0807ea2ec3654043f9d5711d83ea564dbab))
- **config:** Surface hidden otari_env settings on GatewayConfig and fix config docs in [#280](https://github.com/mozilla-ai/otari/pull/280) by [@njbrake](https://github.com/njbrake) ([`34f7a20`](https://github.com/mozilla-ai/otari/commit/34f7a20cc2b7c3fcac770a2192a431987f45e656))
- Otel tool spans telemetry in [#289](https://github.com/mozilla-ai/otari/pull/289) by [@Sourav-Tripathy](https://github.com/Sourav-Tripathy) ([`74b0d58`](https://github.com/mozilla-ai/otari/commit/74b0d5849777954b9f2fd7fb231788f86097d458))
- **ui:** Add a standalone admin dashboard for models, pricing, and settings in [#227](https://github.com/mozilla-ai/otari/pull/227) by [@njbrake](https://github.com/njbrake) ([`fc02390`](https://github.com/mozilla-ai/otari/commit/fc02390f147dda3ad0cc175740b26b5b7bef3130))
- Add agent skills and scoped review instructions in [#296](https://github.com/mozilla-ai/otari/pull/296) by [@njbrake](https://github.com/njbrake) ([`0c3542b`](https://github.com/mozilla-ai/otari/commit/0c3542bab49318e827a337d895b52de667a9106c))
- **ui:** Manage providers, pricing, and aliases at runtime from the dashboard in [#297](https://github.com/mozilla-ai/otari/pull/297) by [@njbrake](https://github.com/njbrake) ([`7d09276`](https://github.com/mozilla-ai/otari/commit/7d09276ff357a81c15f8e2643eab266808e65a74))
- **auth:** Accept x-api-key header so Anthropic-native clients work (fixes #253) in [#315](https://github.com/mozilla-ai/otari/pull/315) by [@AloysJehwin](https://github.com/AloysJehwin) ([`4b4bd96`](https://github.com/mozilla-ai/otari/commit/4b4bd96691586ba6a1c2af9f44fe179d30412281))
- **dashboard:** API key management and per-key model access control in [#318](https://github.com/mozilla-ai/otari/pull/318) by [@njbrake](https://github.com/njbrake) ([`f49d70f`](https://github.com/mozilla-ai/otari/commit/f49d70f82cb100e7d4f368998512cf8e7b7cd287))
- **observability:** Count upstream attempts abandoned before first chunk in [#324](https://github.com/mozilla-ai/otari/pull/324) by [@njbrake](https://github.com/njbrake) ([`b65e669`](https://github.com/mozilla-ai/otari/commit/b65e6690158db1a14454c51aa085175cef3273c2))
- **dashboard:** Budget and user management with two-layer model access in [#322](https://github.com/mozilla-ai/otari/pull/322) by [@njbrake](https://github.com/njbrake) ([`eac679e`](https://github.com/mozilla-ai/otari/commit/eac679e81f6e610b94730f65a05ac5de8c1021d5))
- **BREAKING:** Remove gateway/GATEWAY_ and pre-rename legacy aliases in [#314](https://github.com/mozilla-ai/otari/pull/314) by [@njbrake](https://github.com/njbrake) ([`cb89cd2`](https://github.com/mozilla-ai/otari/commit/cb89cd235f45627872841d67eb7de9d8e5275d76))
- **dashboard:** Activity / request log viewer with per-request latency in [#330](https://github.com/mozilla-ai/otari/pull/330) by [@njbrake](https://github.com/njbrake) ([`631a006`](https://github.com/mozilla-ai/otari/commit/631a006e2bc29fc85dd362271aee4f968eda59bb))
- **batches:** Add a batches table for idempotent accounting, spend folding, and strict ownership in [#340](https://github.com/mozilla-ai/otari/pull/340) by [@njbrake](https://github.com/njbrake) ([`de23b87`](https://github.com/mozilla-ai/otari/commit/de23b87ede08e71e1f91303277707e97c3b75327))
- **dashboard:** Connection toast, resizable columns, and form polish in [#343](https://github.com/mozilla-ai/otari/pull/343) by [@njbrake](https://github.com/njbrake) ([`e2ff7c8`](https://github.com/mozilla-ai/otari/commit/e2ff7c89e7e4bdf3b6392cf7bcf924fde33e5508))
- **dashboard:** Provider health monitor on the Providers page in [#347](https://github.com/mozilla-ai/otari/pull/347) by [@njbrake](https://github.com/njbrake) ([`769c713`](https://github.com/mozilla-ai/otari/commit/769c713b5c3345779175d763fd85ac4c655c9a9d))
- **dashboard:** Usage & analytics page with aggregation endpoints (#303) in [#345](https://github.com/mozilla-ai/otari/pull/345) by [@njbrake](https://github.com/njbrake) ([`2c95f65`](https://github.com/mozilla-ai/otari/commit/2c95f6522c10249aeec8fddbeb81a827be6de139))
- **dashboard:** Expand Settings with a full config viewer, wider settable set, and search in [#346](https://github.com/mozilla-ai/otari/pull/346) by [@njbrake](https://github.com/njbrake) ([`d6d8ccf`](https://github.com/mozilla-ai/otari/commit/d6d8ccff605e9ac868851a2ddf5fb445a769f93f))
- Add ability to edit alias target in dashboard in [#348](https://github.com/mozilla-ai/otari/pull/348) by [@njbrake](https://github.com/njbrake) ([`58ed119`](https://github.com/mozilla-ai/otari/commit/58ed119a870a9410ae4ef498501fdd68b8ecbcc7))
- **dashboard:** Built-in tools & guardrails configuration (#306) in [#350](https://github.com/mozilla-ai/otari/pull/350) by [@njbrake](https://github.com/njbrake) ([`35991e2`](https://github.com/mozilla-ai/otari/commit/35991e2f35876c46e11521987106895e3ff589c4))
- **dashboard:** Link provider name to filtered models page in [#352](https://github.com/mozilla-ai/otari/pull/352) by [@njbrake](https://github.com/njbrake) ([`632b68a`](https://github.com/mozilla-ai/otari/commit/632b68ad146bc0b8863a53c8aed742d33e277166))
- **dashboard:** Overview / home page (#302) in [#354](https://github.com/mozilla-ai/otari/pull/354) by [@njbrake](https://github.com/njbrake) ([`150b2b1`](https://github.com/mozilla-ai/otari/commit/150b2b1b3188155774d3e58a3ee54d691aaf9a27))
- Price cached input tokens in standalone cost calculation in [#356](https://github.com/mozilla-ai/otari/pull/356) by [@njbrake](https://github.com/njbrake) ([`1a8196e`](https://github.com/mozilla-ai/otari/commit/1a8196e44b8150c34a59fcdc863b6ce32f61d65e))
- Add dashboard key lifecycle controls in [#360](https://github.com/mozilla-ai/otari/pull/360) by [@njbrake](https://github.com/njbrake) ([`3f31c9c`](https://github.com/mozilla-ai/otari/commit/3f31c9cb6fe8f6f55a0da3dff29968da19dd0f1a))


### Other

- Add demo gif to readme in [#249](https://github.com/mozilla-ai/otari/pull/249) by [@angpt](https://github.com/angpt) ([`3566b41`](https://github.com/mozilla-ai/otari/commit/3566b410ee2078985aa1d893b5424e8f1f71af3c))


### Performance

- Async MCP/guardrail URL checks, dedupe provider resolution in [#250](https://github.com/mozilla-ai/otari/pull/250) by [@tbille](https://github.com/tbille) ([`878d12a`](https://github.com/mozilla-ai/otari/commit/878d12a1532f9277c55ca1caa0800b92370aa091))



### New Contributors

- [@Shaurya2k06](https://github.com/Shaurya2k06) made their first contribution in [#334](https://github.com/mozilla-ai/otari/pull/334)
- [@AloysJehwin](https://github.com/AloysJehwin) made their first contribution in [#315](https://github.com/mozilla-ai/otari/pull/315)
- [@Sourav-Tripathy](https://github.com/Sourav-Tripathy) made their first contribution in [#289](https://github.com/mozilla-ai/otari/pull/289)
- [@ojusave](https://github.com/ojusave) made their first contribution in [#257](https://github.com/mozilla-ai/otari/pull/257)
- [@MooseTheRebel](https://github.com/MooseTheRebel) made their first contribution in [#245](https://github.com/mozilla-ai/otari/pull/245)

**Full Changelog**: https://github.com/mozilla-ai/otari/compare/v0.2.0...v0.3.0
## [0.2.0](https://github.com/mozilla-ai/otari/releases/tag/v0.2.0) - 2026-07-03



### Bug Fixes

- **client:** Send Postman auth as Authorization Bearer in [#220](https://github.com/mozilla-ai/otari/pull/220) by [@njbrake](https://github.com/njbrake) ([`45ba952`](https://github.com/mozilla-ai/otari/commit/45ba952422559285da7df7c0b98b6128bdc0f379))
- **ui:** Redesign root tutorial page and fix dead quickstart link in [#218](https://github.com/mozilla-ai/otari/pull/218) by [@njbrake](https://github.com/njbrake) ([`e510156`](https://github.com/mozilla-ai/otari/commit/e51015697de7d9e5d67c87d6ca8e392b53d0b667))
- **ui:** Drop "(Proxy Server)" label and add a landing-page favicon in [#222](https://github.com/mozilla-ai/otari/pull/222) by [@njbrake](https://github.com/njbrake) ([`41162c3`](https://github.com/mozilla-ai/otari/commit/41162c30c0aac47642a7968de5be7314b739f547))
- **pricing:** Warn and skip instead of crashing for unlisted providers in [#215](https://github.com/mozilla-ai/otari/pull/215) by [@njbrake](https://github.com/njbrake) ([`80c36d1`](https://github.com/mozilla-ai/otari/commit/80c36d1e92efc61825c5a8b95d1c3c91f7a7858d))


### Features

- **client:** Generate Postman collection from OpenAPI spec in [#216](https://github.com/mozilla-ai/otari/pull/216) by [@njbrake](https://github.com/njbrake) ([`07abe38`](https://github.com/mozilla-ai/otari/commit/07abe38c102a7908ae32b836a4fff12156d81f79))
- **gateway:** Classify hybrid streaming failures and envelope hybrid messages errors in [#202](https://github.com/mozilla-ai/otari/pull/202) by [@khaledosman](https://github.com/khaledosman) ([`52fc430`](https://github.com/mozilla-ai/otari/commit/52fc4300a98e1bf1ed777f45762c5d7538175ad7))
- **providers:** Support multiple named provider instances in [#219](https://github.com/mozilla-ai/otari/pull/219) by [@njbrake](https://github.com/njbrake) ([`c6576a2`](https://github.com/mozilla-ai/otari/commit/c6576a286a9de9398fbbff25741ccb19c63ac7f9))
- **sandbox:** Consume the platform code-execution resolve by [@agpituk](https://github.com/agpituk) ([`d768976`](https://github.com/mozilla-ai/otari/commit/d768976c2b040e4f74323052618948ad4811ec01))
- **hybrid:** Forward request session_label to platform usage report in [#233](https://github.com/mozilla-ai/otari/pull/233) by [@khaledosman](https://github.com/khaledosman) ([`5fdbe75`](https://github.com/mozilla-ai/otari/commit/5fdbe751e54b95cc5cafc60d332cf3dd72bc4c48))
- Opt-in first-chunk grace for the final streaming attempt in [#240](https://github.com/mozilla-ai/otari/pull/240) by [@peteski22](https://github.com/peteski22) ([`73c8837`](https://github.com/mozilla-ai/otari/commit/73c8837c3222cd14c4e6141c7d75cef659bea0e4))


### Other

- Fixed example config file in [#236](https://github.com/mozilla-ai/otari/pull/236) by [@aittalam](https://github.com/aittalam) ([`0f50350`](https://github.com/mozilla-ai/otari/commit/0f5035012966631e14e57c520c42e1518f510b26))



### New Contributors

- [@aittalam](https://github.com/aittalam) made their first contribution in [#236](https://github.com/mozilla-ai/otari/pull/236)

**Full Changelog**: https://github.com/mozilla-ai/otari/compare/v0.1.0...v0.2.0
## [0.1.0](https://github.com/mozilla-ai/otari/releases/tag/v0.1.0) - 2026-06-25



### Bug Fixes

- Add root path bootstrap for migrated test suite by [@tbille](https://github.com/tbille) ([`742ee21`](https://github.com/mozilla-ai/otari/commit/742ee210eae9b4af68a25714aacda19633002bf1))
- Fix preserve existing Vary header values in [#10](https://github.com/mozilla-ai/otari/pull/10) by [@tbille](https://github.com/tbille) ([`c9700dd`](https://github.com/mozilla-ai/otari/commit/c9700ddbe58682458e217cdf9d8099d8064c1686))
- Satisfy lint in platform mode tests and config by [@tbille](https://github.com/tbille) ([`49c2fbb`](https://github.com/mozilla-ai/otari/commit/49c2fbb1231c5b69351b6134afa8f825061a5977))
- Default to platform mode when token is set by [@tbille](https://github.com/tbille) ([`3aea381`](https://github.com/mozilla-ai/otari/commit/3aea38136eab0c1e9237899a51894ccfa5200cae))
- Resolve mypy type errors and regenerate OpenAPI spec by [@tbille](https://github.com/tbille) ([`85469ad`](https://github.com/mozilla-ai/otari/commit/85469add5e966a03e53a4224a8d24e02d7bce83a))
- **chat:** Map stream-creation errors to 502 + narrow route for mypy by [@agpituk](https://github.com/agpituk) ([`b7078ec`](https://github.com/mozilla-ai/otari/commit/b7078ecbca551f3604183dfa0f78aa93c3da646b))
- **streaming:** Bake X-Otari-Request-ID into StreamingResponse headers by [@agpituk](https://github.com/agpituk) ([`100a1d3`](https://github.com/mozilla-ai/otari/commit/100a1d36c84485221944bd0c0f9dc10cde65db46))
- **api:** Normalise naive datetimes from SQLite before tz-aware comparison by [@agpituk](https://github.com/agpituk) ([`8705c11`](https://github.com/mozilla-ai/otari/commit/8705c1127fea5e96b0b67944d5a1388b86f05d59))
- **chat:** Duck-type tool_call detection in MCP loop by [@agpituk](https://github.com/agpituk) ([`bdab428`](https://github.com/mozilla-ai/otari/commit/bdab428b83591eef742839f878f8882eb6240187))
- Address Copilot review feedback on PR #63 by [@agpituk](https://github.com/agpituk) ([`6145a7a`](https://github.com/mozilla-ai/otari/commit/6145a7a056e63f716d7b13a8ff10610ecc3dca56))
- Use requested n as fallback for image count in cost calculation by [@tbille](https://github.com/tbille) ([`cf09b00`](https://github.com/mozilla-ai/otari/commit/cf09b0084999162ec201d51f03ce0ea5ceff0e7d))
- Address Copilot review feedback on audio endpoints by [@tbille](https://github.com/tbille) ([`ba98380`](https://github.com/mozilla-ai/otari/commit/ba983802461de2355c3ae0600bd6f738df58ab0a))
- Address Copilot review feedback on PR #65 by [@agpituk](https://github.com/agpituk) ([`8855127`](https://github.com/mozilla-ai/otari/commit/88551271ffb52b11bda80710949ab34ca2f16c5d))
- **demo:** Auto-start a llamafile via $LLAMAFILE_BIN, no home-dir guessing by [@agpituk](https://github.com/agpituk) ([`bb6a6a1`](https://github.com/mozilla-ai/otari/commit/bb6a6a172185829d8aa075b4fe4d629ccc2ca087))
- **demo:** Handle empty model id from /v1/models and fix HF recipe URL by [@agpituk](https://github.com/agpituk) ([`af8af69`](https://github.com/mozilla-ai/otari/commit/af8af69f8866182b80a4d1bd07d9f23134e5f44e))
- **chat,sandbox:** Address tbille review on PR #65 by [@agpituk](https://github.com/agpituk) ([`1a001c8`](https://github.com/mozilla-ai/otari/commit/1a001c838d49b47005261bb8f8d84fe18ba75b97))
- **deps:** Use git source for any-llm-sdk instead of local path by [@tbille](https://github.com/tbille) ([`ba848ba`](https://github.com/mozilla-ai/otari/commit/ba848ba0a1d2c90c9c587d97642f9a014e101797))
- Import BatchRequestCounts from canonical module to satisfy mypy attr-defined check by [@tbille](https://github.com/tbille) ([`6fc7934`](https://github.com/mozilla-ai/otari/commit/6fc7934447c59032ec1919449ac754d7e2b9803a))
- Inject provider field into all batch endpoint responses by [@tbille](https://github.com/tbille) ([`d54e86f`](https://github.com/mozilla-ai/otari/commit/d54e86f8a739bd850f302a43d907bcd0d6ccb1c8))
- Revert AGENTS.md to main branch version by [@tbille](https://github.com/tbille) ([`ec0c21d`](https://github.com/mozilla-ai/otari/commit/ec0c21dc8e1585400178b80bd9a1e72ad8a58630))
- **deps:** Use any-llm-sdk from PyPI instead of git branch source by [@tbille](https://github.com/tbille) ([`fcfa511`](https://github.com/mozilla-ai/otari/commit/fcfa51198e5b0c748a18494bac47304583ea6bed))
- Address Copilot review feedback on batch endpoints by [@tbille](https://github.com/tbille) ([`8e33cc0`](https://github.com/mozilla-ai/otari/commit/8e33cc0e38334beaf4792143ea13d42983cf90e0))
- Address review feedback on PR #52 by [@tbille](https://github.com/tbille) ([`ce5d7c9`](https://github.com/mozilla-ai/otari/commit/ce5d7c9b43f833d4f7978347f842417362600156))
- **mypy:** Resolve type errors for moderation types and amoderation import by [@tbille](https://github.com/tbille) ([`ac4c01a`](https://github.com/mozilla-ai/otari/commit/ac4c01a55063975bbfac930a47a69b20ddc62706))
- Avoid name redefinition in moderation type shim by [@tbille](https://github.com/tbille) ([`f291039`](https://github.com/mozilla-ai/otari/commit/f291039b4e70f1757ce11836888267f23c90204b))
- Address Copilot review feedback on rerank endpoint by [@tbille](https://github.com/tbille) ([`6102855`](https://github.com/mozilla-ai/otari/commit/6102855a8515bed7ffa1ca84ef2ed32e88c0e28f))
- Use direct arerank import now that SDK exports it by [@tbille](https://github.com/tbille) ([`99bce51`](https://github.com/mozilla-ai/otari/commit/99bce51975286518e968d8874645899235a82e9c))
- **chat:** Align mcp-servers resolver status semantics with credentials resolver by [@agpituk](https://github.com/agpituk) ([`e047521`](https://github.com/mozilla-ai/otari/commit/e047521f1a25894c9979d796debfa1966ea06841))
- **web-search:** Address copilot review on PR #72 by [@agpituk](https://github.com/agpituk) ([`64cd537`](https://github.com/mozilla-ai/otari/commit/64cd5377795cedccba57c5a4e928193255bb513c))
- **web-search:** Silence init errors for unused searxng engines by [@agpituk](https://github.com/agpituk) ([`c20f170`](https://github.com/mozilla-ai/otari/commit/c20f170284f12114a40db09cbc9aa3625d63a663))
- Add missing type annotations for mypy strict mode by [@tbille](https://github.com/tbille) ([`040f26a`](https://github.com/mozilla-ai/otari/commit/040f26ac3d059f29f216d1e9cca05cc10a5ea580))
- Address Copilot review feedback on model discovery by [@tbille](https://github.com/tbille) ([`7af6773`](https://github.com/mozilla-ai/otari/commit/7af677344a066c16481c2e1f723d930706d81058))
- Address remaining Copilot review comments on model discovery by [@tbille](https://github.com/tbille) ([`ab2829d`](https://github.com/mozilla-ai/otari/commit/ab2829df9677e291569df986d9289b282586318d))
- **chat:** Align streaming tool-mode dispatch with tool_mode predicate by [@agpituk](https://github.com/agpituk) ([`934f471`](https://github.com/mozilla-ai/otari/commit/934f4716f3891989c1c5dfcbe567348ffb3433fe))
- **routes:** Address Copilot review on PR 1 by [@agpituk](https://github.com/agpituk) ([`615b5c3`](https://github.com/mozilla-ai/otari/commit/615b5c37c15c32fd79e04bf95e7157dee81b23de))
- **messages:** Map streaming backend-unreachable errors to 502 by [@agpituk](https://github.com/agpituk) ([`a61fda3`](https://github.com/mozilla-ai/otari/commit/a61fda33682af2b2e4978fa75da60853cf366bc1))
- **messages:** Address Copilot review on PR 2 by [@agpituk](https://github.com/agpituk) ([`c154aaf`](https://github.com/mozilla-ai/otari/commit/c154aaf4b1b153a95a27ea5603f69dca143818e2))
- **responses:** Map streaming backend-unreachable errors to 502 by [@agpituk](https://github.com/agpituk) ([`34f89a0`](https://github.com/mozilla-ai/otari/commit/34f89a04e303c5ecc46b5e16343af4217beb55e5))
- **responses:** Address Copilot review on PR 3 by [@agpituk](https://github.com/agpituk) ([`3fc34b6`](https://github.com/mozilla-ai/otari/commit/3fc34b6b612ef0d89fd5c57ed3f27c2af03a478d))
- **platform:** Address Copilot review on PR 4 by [@agpituk](https://github.com/agpituk) ([`c3ab2e0`](https://github.com/mozilla-ai/otari/commit/c3ab2e04d3820d75b18d4ae6d14004c5bc74ce27))
- **messages,responses:** Streaming provider errors → HTTP error + regen OpenAPI spec by [@agpituk](https://github.com/agpituk) ([`1e3273f`](https://github.com/mozilla-ai/otari/commit/1e3273f1949eba8161424572501231bcdcd55f9e))
- **tools:** Typecheck + address Copilot review on PR 86 by [@agpituk](https://github.com/agpituk) ([`f9442e7`](https://github.com/mozilla-ai/otari/commit/f9442e7f84cec7f6712ca41cb4ac8f382eabf85b))
- **budget:** Close cross-user IDOR, overspend race, and unmetered-usage bypasses by [@khaledosman](https://github.com/khaledosman) ([`9b0abbe`](https://github.com/mozilla-ai/otari/commit/9b0abbe5536aff9a77fde17ea3ce0741da636d30))
- **ci:** Typecheck async-generator aclose + regenerate openapi spec by [@khaledosman](https://github.com/khaledosman) ([`69fc2f9`](https://github.com/mozilla-ai/otari/commit/69fc2f9040d1d7e6e8ba3467fa2e6aa478256ac3))
- **review:** Address PR review on budget fix by [@khaledosman](https://github.com/khaledosman) ([`5e17af0`](https://github.com/mozilla-ai/otari/commit/5e17af02eedadc2f0c7c217d755ce2e948b5be94))
- **review:** Harden against negative/zero budget inputs + log estimated cost by [@khaledosman](https://github.com/khaledosman) ([`360bbfa`](https://github.com/mozilla-ai/otari/commit/360bbfa558d4847442d128f4bf11b37cf2e63ec3))
- **guardrails:** Address Copilot review by [@agpituk](https://github.com/agpituk) ([`d134663`](https://github.com/mozilla-ai/otari/commit/d134663f1a4d4c0e64528417da4f2ab01f1ffb48))
- **demo:** Repair guardrails demo_flow /validate formatter by [@agpituk](https://github.com/agpituk) ([`c2d7dc9`](https://github.com/mozilla-ai/otari/commit/c2d7dc9c66cd43079f2395d1e4876d0568dfe621))
- **messages:** Address review on count_tokens endpoint by [@khaledosman](https://github.com/khaledosman) ([`691ac52`](https://github.com/mozilla-ai/otari/commit/691ac526fbcaafea7445c97fa6868f7d5a607280))
- **bootstrap:** Keep bootstrap API key on a single log line in [#131](https://github.com/mozilla-ai/otari/pull/131) by [@njbrake](https://github.com/njbrake) ([`dcf2b00`](https://github.com/mozilla-ai/otari/commit/dcf2b00287b2a084d59077c5f582e5ba5e944fe6))
- **budget:** Refund the reservation on streaming pre-commit provider errors in [#129](https://github.com/mozilla-ai/otari/pull/129) by [@njbrake](https://github.com/njbrake) ([`b0ee34e`](https://github.com/mozilla-ai/otari/commit/b0ee34ee498a28f8116b7c210a25387b7a6a77ce))
- **deploy:** Dispatch to renamed otari-ai platform repo in [#141](https://github.com/mozilla-ai/otari/pull/141) by [@njbrake](https://github.com/njbrake) ([`8823fea`](https://github.com/mozilla-ai/otari/commit/8823feaa60ebd7f6ac06caab11e602716f19b682))
- **sdk-codegen:** Type response schemas in serialization mode so reasoning is a string in [#145](https://github.com/mozilla-ai/otari/pull/145) by [@njbrake](https://github.com/njbrake) ([`55d80f2`](https://github.com/mozilla-ai/otari/commit/55d80f2cdb736e5bd2a9db1867e40e1882ca53a6))
- **files:** Bill vision side-calls, harden delete + retention docs by [@khaledosman](https://github.com/khaledosman) ([`d849cfe`](https://github.com/mozilla-ai/otari/commit/d849cfe8dd380c9360992855d909214f5c7073af))
- **files:** Address review (bare-string Responses input, vision token cap, epoch skew) by [@khaledosman](https://github.com/khaledosman) ([`34e642f`](https://github.com/mozilla-ai/otari/commit/34e642fcaf2962d51008308a55f52052e636ffa0))
- **docker:** Keep python 3.14, bump onnxruntime instead of downgrading in [#159](https://github.com/mozilla-ai/otari/pull/159) by [@njbrake](https://github.com/njbrake) ([`a83442a`](https://github.com/mozilla-ai/otari/commit/a83442ab0c16d0a3c2e341e2b6700d656ee05446))
- **codegen:** Name free-form-object union arrays so Go generates valid syntax in [#166](https://github.com/mozilla-ai/otari/pull/166) by [@njbrake](https://github.com/njbrake) ([`ea973fe`](https://github.com/mozilla-ai/otari/commit/ea973feec324ae944f98258940debf1abea8931d))
- **codegen:** Map bare empty-schema union members to FreeFormObject in [#180](https://github.com/mozilla-ai/otari/pull/180) by [@njbrake](https://github.com/njbrake) ([`69af27e`](https://github.com/mozilla-ai/otari/commit/69af27ecce4e3a0b87369e35752ad7bfb4c7152a))
- **usage:** Treat 402 from the usage-report endpoint as non-retryable by [@khaledosman](https://github.com/khaledosman) ([`3f9c767`](https://github.com/mozilla-ai/otari/commit/3f9c767889d8c1735ee6e37df0631b0c4eea4346))
- **fallback:** Treat 404/405/409/410 as retryable so model-unavailable falls through by [@agpituk](https://github.com/agpituk) ([`670165d`](https://github.com/mozilla-ai/otari/commit/670165d6bc785d171ca4886f125d774c732ff49d))
- **fallback:** Report each failed attempt when the whole chain is exhausted by [@agpituk](https://github.com/agpituk) ([`039ce8c`](https://github.com/mozilla-ai/otari/commit/039ce8c7e15bfc09209c13b1d6482382a1c24a28))
- **fallback:** Log dropped inline usage-report failures; drop em dashes in [#174](https://github.com/mozilla-ai/otari/pull/174) by [@khaledosman](https://github.com/khaledosman) ([`14610be`](https://github.com/mozilla-ai/otari/commit/14610be4a1e985524fb47fd3310ee268ac8f6b6a))
- **fallback:** Report each failed attempt when a streaming chain is exhausted by [@agpituk](https://github.com/agpituk) ([`98be421`](https://github.com/mozilla-ai/otari/commit/98be42158e93ec8531e92c5e0fa72a6a6d4372c2))
- **fallback:** Don't block cancellation or leak the backend stack on all-failed stream in [#176](https://github.com/mozilla-ai/otari/pull/176) by [@khaledosman](https://github.com/khaledosman) ([`ed0978b`](https://github.com/mozilla-ai/otari/commit/ed0978bd3a04f8ee33773a1d400b6a8c1587b7e9))
- **fallback:** Bound the inline usage-report flush; dedupe all-failed test by [@khaledosman](https://github.com/khaledosman) ([`419e19b`](https://github.com/mozilla-ai/otari/commit/419e19b8da444c5c87c5e0cd17b64977ca416d54))


### Features

- Extract gateway runtime into standalone service repo by [@tbille](https://github.com/tbille) ([`522fe09`](https://github.com/mozilla-ai/otari/commit/522fe09d2fa7f3c4860d8e1a90a9854d3a31aa3f))
- Add platform mode startup gating and API surface by [@tbille](https://github.com/tbille) ([`68e9538`](https://github.com/mozilla-ai/otari/commit/68e953884699166f7b2a5074ecfb522ee6c0b0e2))
- Integrate platform resolve and async usage reporting by [@tbille](https://github.com/tbille) ([`c491bba`](https://github.com/mozilla-ai/otari/commit/c491bba152d48fb92555f85e550a3e66c90e5b20))
- **gateway:** Add enable_docs config to disable FastAPI docs endpoints by [@tbille](https://github.com/tbille) ([`786dc75`](https://github.com/mozilla-ai/otari/commit/786dc75266aa123e9a2a90e466188cb63190232a))
- **asyncio:** Convert DB layer to async, add budget strategies and log writer by [@tbille](https://github.com/tbille) ([`3e593b6`](https://github.com/mozilla-ai/otari/commit/3e593b664154224d60fbf128d0c448892aebf94f))
- **pricing:** Add effective_at to model_pricing for historical price tracking by [@tbille](https://github.com/tbille) ([`e1d69b9`](https://github.com/mozilla-ai/otari/commit/e1d69b9de3ece62ec7398fde8eefdb808658bc40))
- **gateway:** Add GET /v1/usage bulk endpoint with time range filters by [@tbille](https://github.com/tbille) ([`139fef9`](https://github.com/mozilla-ai/otari/commit/139fef95ec734d63f47928b293cefcaf569ab525))
- Add POST /v1/responses endpoint for OpenAI Responses API by [@tbille](https://github.com/tbille) ([`b9b0e59`](https://github.com/mozilla-ai/otari/commit/b9b0e59228ffd9fec8db013753c19a59f9101cbf))
- **gateway:** Rename API_KEY_HEADER from X-AnyLLM-Key to AnyLLM-Key by [@peteski22](https://github.com/peteski22) ([`a006df8`](https://github.com/mozilla-ai/otari/commit/a006df8096a072ddd1ff66a1edfa9a6c3dab069e))
- **gateway:** Accept legacy X-AnyLLM-Key header for back-compat by [@tbille](https://github.com/tbille) ([`a6f00e6`](https://github.com/mozilla-ai/otari/commit/a6f00e6d26153c476b4f5966a40128a16884183f))
- **chat:** Walk multi-attempt resolve responses with retry on transient errors by [@agpituk](https://github.com/agpituk) ([`08e4370`](https://github.com/mozilla-ai/otari/commit/08e4370ab2160318b2b599e6e0ca612795823505))
- **streaming:** First-chunk gate for routing-policy fallback (v1.1) by [@agpituk](https://github.com/agpituk) ([`24e39fe`](https://github.com/mozilla-ai/otari/commit/24e39fe307d5223c8f253058e990fa62ea710b57))
- **chat:** MCP tool-use loop with streaming support by [@agpituk](https://github.com/agpituk) ([`850be4c`](https://github.com/mozilla-ai/otari/commit/850be4ce9da76fe240ab5414db2b986a4045ef42))
- **chat:** URL safety, content-block rendering, hardened error handling by [@agpituk](https://github.com/agpituk) ([`5724cba`](https://github.com/mozilla-ai/otari/commit/5724cbad9c4cc69c893370d9fdd979802a8ab146))
- Add POST /v1/images/generations endpoint by [@tbille](https://github.com/tbille) ([`f99061b`](https://github.com/mozilla-ai/otari/commit/f99061b16b7b2485f1a93e4404c606596c3947bd))
- Add POST /v1/audio/transcriptions and /v1/audio/speech endpoints by [@tbille](https://github.com/tbille) ([`6749d47`](https://github.com/mozilla-ai/otari/commit/6749d474d12714e82f9e71f6a665e27cf99412f2))
- **chat:** Code execution via sandbox container with multi-keyword detection by [@agpituk](https://github.com/agpituk) ([`0df1f89`](https://github.com/mozilla-ai/otari/commit/0df1f891d36c0d480251f4cecd7ea8d3ecb2f2b2))
- **chat:** Resolve workspace-scoped MCP server ids in platform mode by [@agpituk](https://github.com/agpituk) ([`fe8bbf1`](https://github.com/mozilla-ai/otari/commit/fe8bbf14a6b0ec3873a4e45c15d0a68e2fe2849b))
- **chat:** Hoist user_token + narrow type guard for platform resolve by [@agpituk](https://github.com/agpituk) ([`3cfbf8a`](https://github.com/mozilla-ai/otari/commit/3cfbf8aabe40a7049fc68a42ec4589ed11676283))
- **chat:** Web_search tool via SearXNG-compatible backend with content extraction by [@agpituk](https://github.com/agpituk) ([`07c14c0`](https://github.com/mozilla-ai/otari/commit/07c14c0e09b76294479a2f4a4c9d1f5fb076efb1))
- Auto-discover models from configured providers for /v1/models by [@tbille](https://github.com/tbille) ([`0c95f3b`](https://github.com/mozilla-ai/otari/commit/0c95f3bc65cf2aac16f365e6196164a8cf8fabb8))
- **chat:** Pre-lock-in fallback for tool-loop requests by [@agpituk](https://github.com/agpituk) ([`b4866ce`](https://github.com/mozilla-ai/otari/commit/b4866ce413fc4b4f8b2140e4be719422dadc6389))
- **messages:** MCP / sandbox / web_search on /v1/messages by [@agpituk](https://github.com/agpituk) ([`1fa3e47`](https://github.com/mozilla-ai/otari/commit/1fa3e47e8ab536b2b87233d5e5faf266d8b8cdef))
- **messages:** Wire on_first_response into anthropic_tool_loop by [@agpituk](https://github.com/agpituk) ([`65ce24a`](https://github.com/mozilla-ai/otari/commit/65ce24a2619063e4667a8c2ccdba19b204a13ffe))
- **responses:** MCP / sandbox / web_search on /v1/responses by [@agpituk](https://github.com/agpituk) ([`92b9aa1`](https://github.com/mozilla-ai/otari/commit/92b9aa12616b08e55032b84bb010ce27b2fde481))
- **responses:** Wire on_first_response into responses_tool_loop by [@agpituk](https://github.com/agpituk) ([`a8bfe78`](https://github.com/mozilla-ai/otari/commit/a8bfe7847dee5b24530905969d1d09c12b2b0379))
- **platform:** Platform-mode routing for /v1/messages and /v1/responses by [@agpituk](https://github.com/agpituk) ([`4cc54f8`](https://github.com/mozilla-ai/otari/commit/4cc54f80b2a58176f3a5fcd65215f7393696df58))
- **BREAKING:** **tools:** Explicit otari_* gateway tool shapes (BREAKING) by [@agpituk](https://github.com/agpituk) ([`259a267`](https://github.com/mozilla-ai/otari/commit/259a267eb11eebdb49d90a32f8fb2ec8fd248fb1))
- **demo:** --brave flag for the web-search demo by [@agpituk](https://github.com/agpituk) ([`b67b413`](https://github.com/mozilla-ai/otari/commit/b67b4136217b8a3f44d34a17d08a21568a1b5096))
- **guardrails:** Request-level guardrails across all endpoints by [@agpituk](https://github.com/agpituk) ([`41bb2a9`](https://github.com/mozilla-ai/otari/commit/41bb2a9558d77c9df608e587cba94bed7aa1a1af))
- **guardrails:** Default to PIGuard via encoderfile, monitor mode by [@agpituk](https://github.com/agpituk) ([`16632cd`](https://github.com/mozilla-ai/otari/commit/16632cd054ec5aee86e98009c1ff7343e32d5f16))
- Add CodeRabbit configuration in [#110](https://github.com/mozilla-ai/otari/pull/110) by [@njbrake](https://github.com/njbrake) ([`f52cdd1`](https://github.com/mozilla-ai/otari/commit/f52cdd1616f3aeb1a197bfbb57a39d49cf5ab8aa))
- **messages:** Add /v1/messages/count_tokens for Claude Code support by [@khaledosman](https://github.com/khaledosman) ([`2664e11`](https://github.com/mozilla-ai/otari/commit/2664e11d7e27ac65613ddca744660c8320a61afe))
- Add PR template and enforcement workflow in [#123](https://github.com/mozilla-ai/otari/pull/123) by [@njbrake](https://github.com/njbrake) ([`690996c`](https://github.com/mozilla-ai/otari/commit/690996c33ac3f99a075dec6488ed88ebbb125059))
- **sdk-codegen:** Emit the rust core as an inlined src/_client module in [#136](https://github.com/mozilla-ai/otari/pull/136) by [@njbrake](https://github.com/njbrake) ([`aca95e3`](https://github.com/mozilla-ai/otari/commit/aca95e34f12971a6d493fe4f40dcca72ba78d650))
- **sdk-codegen:** Coordinate SDK releases with stamped spec version and staleness alert in [#137](https://github.com/mozilla-ai/otari/pull/137) by [@njbrake](https://github.com/njbrake) ([`3996758`](https://github.com/mozilla-ai/otari/commit/39967588208d6884f4569511237d9bd2e836df24))
- **web-search:** Per-workspace web-search policy via platform resolve in [#144](https://github.com/mozilla-ai/otari/pull/144) by [@agpituk](https://github.com/agpituk) ([`a2fc5c8`](https://github.com/mozilla-ai/otari/commit/a2fc5c8bdfe0e196dadbda8d4fd14ee3af3f6428))
- **files:** File uploads + document understanding for local models by [@khaledosman](https://github.com/khaledosman) ([`56ba31e`](https://github.com/mozilla-ai/otari/commit/56ba31e47b48d515445466ff6d7db7d7ec2f90f3))
- **api:** Derive request schemas from any-llm Params to stop param drift in [#160](https://github.com/mozilla-ai/otari/pull/160) by [@njbrake](https://github.com/njbrake) ([`d3948a1`](https://github.com/mozilla-ai/otari/commit/d3948a1ddea7140cfb9e2b040cd7b9a7be19becf))
- **web-search:** Forward X-Gateway-Token to the search backend in [#164](https://github.com/mozilla-ai/otari/pull/164) by [@agpituk](https://github.com/agpituk) ([`2ec9053`](https://github.com/mozilla-ai/otari/commit/2ec9053b5be5a8408a42b6f38a2bb5b88cfa5cdc))
- Generate the changelog and release notes from commits with git-cliff in [#162](https://github.com/mozilla-ai/otari/pull/162) by [@njbrake](https://github.com/njbrake) ([`c2d3eb2`](https://github.com/mozilla-ai/otari/commit/c2d3eb2142ec68f807834d2087961b154d1a27a3))
- **codegen:** Type the image-generation response from any-llm ImagesResponse in [#178](https://github.com/mozilla-ai/otari/pull/178) by [@njbrake](https://github.com/njbrake) ([`90f4523`](https://github.com/mozilla-ai/otari/commit/90f4523bd862af267a157a0fc944d938479ad192))
- **sandbox:** Forward caller token to the sandbox backend as Bearer auth in [#182](https://github.com/mozilla-ai/otari/pull/182) by [@agpituk](https://github.com/agpituk) ([`a74c8bf`](https://github.com/mozilla-ai/otari/commit/a74c8bfafd0d4cc9a85772c684a45befe64a209f))
- **usage:** Forward provider cached-token counts in usage reports in [#196](https://github.com/mozilla-ai/otari/pull/196) by [@tbille](https://github.com/tbille) ([`1b6a80e`](https://github.com/mozilla-ai/otari/commit/1b6a80e6803d86464b7bde64270e3e644110a95c))
- **gateway:** Classify upstream provider errors into specific safe responses in [#197](https://github.com/mozilla-ai/otari/pull/197) by [@khaledosman](https://github.com/khaledosman) ([`01df970`](https://github.com/mozilla-ai/otari/commit/01df97061d0f0a37ca8e286f000f52d4a7b4031f))
- **deploy:** Add Deploy on Railway template scaffold and button in [#210](https://github.com/mozilla-ai/otari/pull/210) by [@njbrake](https://github.com/njbrake) ([`528c9be`](https://github.com/mozilla-ai/otari/commit/528c9bebea0813d3368254ad9bc982070ac05e54))
- **config:** Support full provider/pricing config via environment in [#211](https://github.com/mozilla-ai/otari/pull/211) by [@njbrake](https://github.com/njbrake) ([`022a8dc`](https://github.com/mozilla-ai/otari/commit/022a8dced6b64ba4bd3d16d31c955626c322c8bb))
- **pricing:** Add genai-prices default pricing fallback in [#201](https://github.com/mozilla-ai/otari/pull/201) by [@njbrake](https://github.com/njbrake) ([`9880f58`](https://github.com/mozilla-ai/otari/commit/9880f582f87f04d18d361195fbfc1ea8b806f6d1))


### Other

- Standardize logger placeholders across gateway modules in [#12](https://github.com/mozilla-ai/otari/pull/12) by [@tbille](https://github.com/tbille) ([`8c92ed8`](https://github.com/mozilla-ai/otari/commit/8c92ed84387417751bdf1438fb0b5f6692c2b9fa))
- Add Ruff linting with local and CI enforcement in [#14](https://github.com/mozilla-ai/otari/pull/14) by [@tbille](https://github.com/tbille) ([`e0bd004`](https://github.com/mozilla-ai/otari/commit/e0bd0044d2217f9172972c1bcb881cc749e7280c))
- Make Vertex credential setup request-local in [#17](https://github.com/mozilla-ai/otari/pull/17) by [@tbille](https://github.com/tbille) ([`df52553`](https://github.com/mozilla-ai/otari/commit/df52553fb92aa01d5706d5780c0628da9d83a39f))
- Avoid row lock during preflight budget checks in [#18](https://github.com/mozilla-ai/otari/pull/18) by [@tbille](https://github.com/tbille) ([`1deb7e9`](https://github.com/mozilla-ai/otari/commit/1deb7e95d3c8152475deafd4f63a4eb2cdb42c8c))
- Remove unused domain exception hierarchy in [#23](https://github.com/mozilla-ai/otari/pull/23) by [@tbille](https://github.com/tbille) ([`87e30f5`](https://github.com/mozilla-ai/otari/commit/87e30f5986685371e2cf245c13e217c6657f5518))
- Drop user_id labels from rate and budget counters in [#21](https://github.com/mozilla-ai/otari/pull/21) by [@tbille](https://github.com/tbille) ([`a88a432`](https://github.com/mozilla-ai/otari/commit/a88a4324a46e730c6e014488e77e86a7c9381a32))
- Adding release actions by [@macaab26](https://github.com/macaab26) ([`eed4b97`](https://github.com/mozilla-ai/otari/commit/eed4b97946e800fdaebf084ba5f4f0ccc402ff6d))
- Changing tags naming by [@macaab26](https://github.com/macaab26) ([`59550be`](https://github.com/mozilla-ai/otari/commit/59550be0ec5ef884eb95c6abc09153b0feef3f50))
- Any-llm → otari across the project by [@tbille](https://github.com/tbille) ([`0153cc2`](https://github.com/mozilla-ai/otari/commit/0153cc22c98bf78aaa472e4633ef04176468e325))
- Docker image mzdotai/gateway → mzdotai/otari by [@tbille](https://github.com/tbille) ([`2ffd357`](https://github.com/mozilla-ai/otari/commit/2ffd3578ed42182f7ba782e00623d4a77131eb4c))
- Rename AWS ECS cluster and service to otari-ai naming convention by [@tbille](https://github.com/tbille) ([`d5ab256`](https://github.com/mozilla-ai/otari/commit/d5ab2568af5b3a506f9c6d8043cea805b5f84ae5))
- Update to readme to include Otari in title by [@tbille](https://github.com/tbille) ([`d537202`](https://github.com/mozilla-ai/otari/commit/d537202597411b7f0f7a9720ff284defd415d9c2))
- Add batch API endpoints for asynchronous LLM processing by [@tbille](https://github.com/tbille) ([`08d77e5`](https://github.com/mozilla-ai/otari/commit/08d77e5998dbb9dc9a5ab0e00c97887010d86885))
- Fix batch API review issues: add FIXME for results endpoint, usage logging, error handling, and missing tests by [@tbille](https://github.com/tbille) ([`4960a8b`](https://github.com/mozilla-ai/otari/commit/4960a8b199e0276658daffbdb87acd47f8381469))
- Move Batch import out of TYPE_CHECKING and add SDK batch result imports by [@tbille](https://github.com/tbille) ([`a042fd6`](https://github.com/mozilla-ai/otari/commit/a042fd691c794b367e1205c5237cf1756f5a2967))
- Remove dead db parameter from log_batch_usage and fix model sentinel by [@tbille](https://github.com/tbille) ([`a0b6c1d`](https://github.com/mozilla-ai/otari/commit/a0b6c1dd459c58ac3063d3777511bab1418d3419))
- Switch get_db to get_db_if_needed and add BackgroundTasks parameter by [@tbille](https://github.com/tbille) ([`b8c1159`](https://github.com/mozilla-ai/otari/commit/b8c1159ab4afe8e9e8cd24f9579316588c66e191))
- Use aretrieve_batch_results and BatchNotCompleteError in results endpoint by [@tbille](https://github.com/tbille) ([`28f6ed8`](https://github.com/mozilla-ai/otari/commit/28f6ed8ae66add8ddac9dd305d5da71d57026c58))
- Mock split_model_provider in invalid model format test by [@tbille](https://github.com/tbille) ([`f523fd4`](https://github.com/mozilla-ai/otari/commit/f523fd4f4631efa1c81d24fbff067de8a2a085e1))
- Update results endpoint tests to use aretrieve_batch_results by [@tbille](https://github.com/tbille) ([`e2912a5`](https://github.com/mozilla-ai/otari/commit/e2912a52c146b63ac0335ee27006b3fb6dc31dd2))
- Pin any-llm-sdk to local source with batch result types by [@tbille](https://github.com/tbille) ([`d70ca88`](https://github.com/mozilla-ai/otari/commit/d70ca88baa96f3131b5d7c8206f688d46447727b))
- Update AGENTS.md with batch API spec and sync uv.lock for local SDK by [@tbille](https://github.com/tbille) ([`cae8709`](https://github.com/mozilla-ai/otari/commit/cae8709044dae686a7ae7c0b6b066d4fa09a4933))
- Add moderations route mirroring /v1/embeddings by [@tbille](https://github.com/tbille) ([`85446bc`](https://github.com/mozilla-ai/otari/commit/85446bccabc349fb7b535fb23d7f9b5797626272))
- Register moderations router in standalone mode by [@tbille](https://github.com/tbille) ([`0557127`](https://github.com/mozilla-ai/otari/commit/0557127dbbb1b946b8e65e06698b89a0e7537965))
- Add integration tests for /v1/moderations endpoint by [@tbille](https://github.com/tbille) ([`e80faf3`](https://github.com/mozilla-ai/otari/commit/e80faf3a28177c65cdcff2ebfd4f14d7e5cffaa8))
- Document /v1/moderations in README API surface by [@tbille](https://github.com/tbille) ([`69c9198`](https://github.com/mozilla-ai/otari/commit/69c919814a6fa9491124844b9c5fc873e0068884))
- Add local ModerationResponse/ModerationResult types by [@tbille](https://github.com/tbille) ([`776c477`](https://github.com/mozilla-ai/otari/commit/776c477b6674cfa2ec99ed251edd72935ac2229b))
- Fix moderations route import against older any-llm-sdk by [@tbille](https://github.com/tbille) ([`317b3f1`](https://github.com/mozilla-ai/otari/commit/317b3f1278c746c82234044427cc8c21899aec25))
- Import moderation types from gateway-local module in tests by [@tbille](https://github.com/tbille) ([`db30a43`](https://github.com/mozilla-ai/otari/commit/db30a43b3e7f414a5a6ce464e46cd3cb6099a161))
- Regenerate OpenAPI spec for /v1/moderations route by [@tbille](https://github.com/tbille) ([`7a799d6`](https://github.com/mozilla-ai/otari/commit/7a799d6234002bfa026dc6a2f6b06b420e4c2e81))
- Final uncommitted changes for add-moderation-api-support by [@tbille](https://github.com/tbille) ([`7161e1f`](https://github.com/mozilla-ai/otari/commit/7161e1fa1e8865beda6a09424c45bf0999bf6d9d))
- Add POST /v1/rerank route handler and register router by [@tbille](https://github.com/tbille) ([`cdc1ea4`](https://github.com/mozilla-ai/otari/commit/cdc1ea49be78db5c130f2273b44fa6847b99fefa))
- Add integration tests for /v1/rerank endpoint by [@tbille](https://github.com/tbille) ([`a77162d`](https://github.com/mozilla-ai/otari/commit/a77162d253699c6cf18533440b4844226e84c4ef))
- Regenerate OpenAPI spec with /v1/rerank endpoint by [@tbille](https://github.com/tbille) ([`d624ccb`](https://github.com/mozilla-ai/otari/commit/d624ccbbd9f31fc745284bf76e4e6fd997e8d12b))
- Default PLATFORM_BASE_URL for otari.ai by [@tbille](https://github.com/tbille) ([`d373516`](https://github.com/mozilla-ai/otari/commit/d3735167388c03b80f37cc41e44cb492cccde7ef))
- Rename platform token env var to OTARI_AI_TOKEN by [@tbille](https://github.com/tbille) ([`bc93f37`](https://github.com/mozilla-ai/otari/commit/bc93f37b1d9bfe6d5788782921ea25aee9e1d8b9))
- Support OTARI_* env vars for core settings by [@tbille](https://github.com/tbille) ([`fdfcc35`](https://github.com/mozilla-ai/otari/commit/fdfcc359e812a976bf1308f14d2f4fa19c040192))
- Deleting deployment actions by [@macaab26](https://github.com/macaab26) ([`6519e60`](https://github.com/mozilla-ai/otari/commit/6519e60a3657a9ca10e0fa78e0de82dd3b9cfed1))
- Adding trigger for the platform by [@macaab26](https://github.com/macaab26) ([`bd29d0f`](https://github.com/mozilla-ai/otari/commit/bd29d0f13efc453f3ab7e463cbf94773d6c45386))
- Accept gw_ API key prefix by [@tbille](https://github.com/tbille) ([`5c548ef`](https://github.com/mozilla-ai/otari/commit/5c548ef1552cbcc0ada99bed51757040bb40d5b2))
- Add CLAUDE.md as a symlink to AGENTS.md in [#98](https://github.com/mozilla-ai/otari/pull/98) by [@njbrake](https://github.com/njbrake) ([`156fd4e`](https://github.com/mozilla-ai/otari/commit/156fd4eb28832e755dd5fa2343f235ef325e919c))
- Refresh README: ecosystem framing + SDK links in [#104](https://github.com/mozilla-ai/otari/pull/104) by [@njbrake](https://github.com/njbrake) ([`c46de4e`](https://github.com/mozilla-ai/otari/commit/c46de4e845e91550a8ed2d18ca7c99466a95cb65))
- SDK automatic codegen from OpenAPI spec in [#99](https://github.com/mozilla-ai/otari/pull/99) by [@njbrake](https://github.com/njbrake) ([`ac8bb58`](https://github.com/mozilla-ai/otari/commit/ac8bb58f3b1fbb0f21a274c1119d3b2cbf0a4f89))
- Pin OpenAPI Generator via openapitools.json in SDK codegen workflow in [#109](https://github.com/mozilla-ai/otari/pull/109) by [@njbrake](https://github.com/njbrake) ([`3be3d54`](https://github.com/mozilla-ai/otari/commit/3be3d542235bd812b77b9ec20a01c90cba0d2bb9))
- Tech-debt cleanup: delete dead src/ shims, ToolBackend Protocol, document narrowing asserts in [#117](https://github.com/mozilla-ai/otari/pull/117) by [@njbrake](https://github.com/njbrake) ([`d87e3fa`](https://github.com/mozilla-ai/otari/commit/d87e3fa5dabe76504e6f68bcf472f333b5fe5fbe))
- Fix intermittent auth 500s from SQLite lock contention (#106) in [#115](https://github.com/mozilla-ai/otari/pull/115) by [@njbrake](https://github.com/njbrake) ([`bde1b1f`](https://github.com/mozilla-ai/otari/commit/bde1b1f228d01a485144becb646a989b2bc58ef4))
- Docs/refactor in [#119](https://github.com/mozilla-ai/otari/pull/119) by [@angpt](https://github.com/angpt) ([`1286093`](https://github.com/mozilla-ai/otari/commit/1286093710fa4a63c27c130d9a1c1da2ef6fc7b2))
- Consolidate the chat/messages/responses request pipelines onto a shared core in [#130](https://github.com/mozilla-ai/otari/pull/130) by [@njbrake](https://github.com/njbrake) ([`172325d`](https://github.com/mozilla-ai/otari/commit/172325db39eaf90f474aa94d9b06fdc221d58a75))
- Change repository name in gateway-docker.yml by [@njbrake](https://github.com/njbrake) ([`d4b29f7`](https://github.com/mozilla-ai/otari/commit/d4b29f7a7d346bb5764e1aa927f970d51da74c4c))
- Rename "Otari Gateway" to "Otari" in [#142](https://github.com/mozilla-ai/otari/pull/142) by [@njbrake](https://github.com/njbrake) ([`8cd4f7f`](https://github.com/mozilla-ai/otari/commit/8cd4f7f91a282a81c7be8b508f6038bd55a21977))


### Performance

- Optimize rate limiter hot path with deque in [#20](https://github.com/mozilla-ai/otari/pull/20) by [@tbille](https://github.com/tbille) ([`93d3e2e`](https://github.com/mozilla-ai/otari/commit/93d3e2ec95ad4daf4b0a20c832ef213d0636e440))
- Throttle api key last_used_at commit frequency in [#19](https://github.com/mozilla-ai/otari/pull/19) by [@tbille](https://github.com/tbille) ([`02a41aa`](https://github.com/mozilla-ai/otari/commit/02a41aa3de3428f35ffca256675e7c8329f95083))


### Security

- Require auth for pricing read endpoints in [#24](https://github.com/mozilla-ai/otari/pull/24) by [@tbille](https://github.com/tbille) ([`60eeb61`](https://github.com/mozilla-ai/otari/commit/60eeb614e9d58f1d43280fb8ba74acc6f67fe2b5))



### New Contributors

- [@njbrake](https://github.com/njbrake) made their first contribution in [#214](https://github.com/mozilla-ai/otari/pull/214)
- [@khaledosman](https://github.com/khaledosman) made their first contribution in [#212](https://github.com/mozilla-ai/otari/pull/212)
- [@dependabot[bot]](https://github.com/dependabot[bot]) made their first contribution in [#193](https://github.com/mozilla-ai/otari/pull/193)
- [@tbille](https://github.com/tbille) made their first contribution in [#196](https://github.com/mozilla-ai/otari/pull/196)
- [@angpt](https://github.com/angpt) made their first contribution in [#119](https://github.com/mozilla-ai/otari/pull/119)
- [@macaab26](https://github.com/macaab26) made their first contribution in [#85](https://github.com/mozilla-ai/otari/pull/85)


