# Catalog smoke

A hand-run demo, not a CI gate: nothing runs this, and unlike
`scripts/oss_edition_smoke.py` it wants `uv`, `pnpm` and a dashboard build. It
puts a standalone gateway on your machine, reachable from the LAN, with three
people to sign in as. Bring your own provider keys; the catalog is empty until
you do.

Do not leave it running. It binds `0.0.0.0`, serves the catalog publicly, and
prints the master key, so anyone on the network who reaches the port can use
the deployment. The sign-in password is drawn at random per `.state/`
directory (`OTARI_SMOKE_PASSWORD` overrides it), which keeps a passer-by from
signing in as the platform admin, and is not a reason to leave it up.

```sh
demo/catalog-smoke/run.sh            # builds the dashboard, boots, seeds, prints who to sign in as
demo/catalog-smoke/run.sh --no-build # skip the dashboard build on a later run
demo/catalog-smoke/run.sh --reset    # start the database over
```

Needs `uv`, `pnpm` and `python3`. State lives in `.state/` beside the script
and survives runs: the master key, the credential-encryption key, the config,
the SQLite database and the gateway log. The card the script prints at the
end says the addresses, the password and the master key.

## Who you can be

| Sign in as | Standing | What to try |
| --- | --- | --- |
| `operator@otari.local` | Platform admin: the deployment's operator, owner of the default organization and of Acme | Settings; Deployment providers (add a real key); Providers, where a key's models arrive priced, each with a Serving switch and a rate you can override; Accounts; switch into Acme from the organization menu |
| `admin@acme.local` | Org admin of Acme | Providers shows the deployment's prices read-only and a rate on any of Acme's own offered models editable; a model page's rows say "Set your rate"; Members, Email domains, Spend & budgets |
| `member@acme.local` | Member of Acme | Models at Acme's rates with nothing to edit; open a row for the selector and request; make an API key on API keys and send a request with it; Usage and Activity show only their own |

Signed out, `#/models` is the public catalog at the deployment's list rates.

## Provider keys

The config ships with no providers. Either add a `providers:` block to
`.state/config.yml` and restart, or sign in as the platform admin and add the
provider on Deployment providers, which stores the key encrypted with the key in
`.state/secret-key`. Discovery is on, so a provider's models appear in the
catalog as soon as its key works, priced from the genai-prices defaults until
you set a rate. An organization's own key goes on Providers instead, where its
models are pulled and priced when the key is added.

## Mail

`mail_transport: console` writes every mail to `.state/gateway.log` instead
of sending it. An invitation sent from Members lands there with its accept
link, which is how the seed verified the two Acme accounts.
