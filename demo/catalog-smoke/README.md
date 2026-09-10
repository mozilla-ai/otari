# Catalog smoke

A standalone gateway on your machine, reachable from the LAN, with the
grouped catalog seeded and three people to sign in as.

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
| `operator@otari.local` | Platform admin: the deployment's operator, owner of the default organization and of Acme | Settings; Model pricing (the catalog policy, "Check for price updates", a rate on any selector, the drift column); Providers (add a real key); Accounts; switch into Acme from the organization menu |
| `admin@acme.local` | Org admin of Acme | Model pricing shows the deployment's prices read-only and Acme's own rate overrides editable; a model page's rows say "Set your rate"; Members, Email domains, Spend & budgets, Acme's provider keys |
| `member@acme.local` | Member of Acme | Models at Acme's rates with nothing to edit; open a row for the selector and request; make an API key on API keys and send a request with it; Usage and Activity show only their own |

Signed out, `#/models` is the public catalog at the deployment's list rates.

## Provider keys

Every key in `config.template.yml` is fake. To reach a real provider, either
edit `.state/config.yml` and restart, or sign in as the platform admin and add
the provider on Providers, which stores the key encrypted with the key in
`.state/secret-key`. Discovery is on, so a real key lists that provider's
models; a fake one fails its discovery and the catalog keeps the priced
selectors from the config.

## Mail

`mail_transport: console` writes every mail to `.state/gateway.log` instead
of sending it. An invitation sent from Members lands there with its accept
link, which is how the seed verified the two Acme accounts.
