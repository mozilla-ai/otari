# Move config.yml providers to hosted providers

The `providers:` block of `config.yml` is deprecated in favor of
[hosted providers](hosted-providers.md), which the Hosted providers page and
`/api/v1/hosted-providers` manage. A start copies what it can for you; this
guide is what is left to do. It also covers providers stored on the retired
Deployment providers page.

## Before you upgrade

1. Set `OTARI_SECRET_KEY` if it is not set yet (`otari gen-secret-key`). A
   hosted provider's key is stored encrypted, and an entry whose key cannot be
   encrypted is not copied.
2. Decide whether to move now. To keep serving every entry from `config.yml`
   for the time being, set `import_configured_providers: false`
   (`OTARI_IMPORT_CONFIGURED_PROVIDERS=false`) and skip the rest of this guide.

## Move the `providers:` entries

1. Start the upgraded gateway. Each entry a hosted provider can hold is copied
   onto a hosted provider of the same name, with every model its provider lists
   switched on, priced or not. The rules are in
   [Copies of `providers:` entries](hosted-providers.md#copies-of-providers-entries).
2. Read the startup log. For each copied entry it says the entry is served as a
   hosted provider and no longer read. For each entry it kept, it says why.
3. Remove the copied entries from `config.yml`. They are no longer read, so
   leaving them only misleads the next reader.
4. Rotate a copied provider's key on the Hosted providers page from now on, not
   in the environment: the `${VAR}` was read once, at the copy.
5. For each entry the start kept, either leave it in `config.yml`, where it
   keeps serving, or remove what stops the copy (for example a `models` list or
   `max_retries`) and restart. An entry whose provider listed no models is tried
   again at every start.

Removing a copied provider on the Hosted providers page sticks: the entry is not
copied again, and it does not serve from `config.yml` either.

### What changes for callers

A copied provider is a hosted provider, the last rung of the
[credential ladder](hosted-providers.md#the-credential-ladder). An organization
that holds its own key for the provider is now served on that key, where the
`config.yml` entry was asked before it. Pricing follows the credential that
served; see [Pricing](hosted-providers.md#pricing).

## Providers from the Deployment providers page

Earlier releases stored runtime providers on a Deployment providers page
(`/api/v1/provider-credentials`), which is gone. At startup, each provider it
stored becomes a hosted provider of the same name, served by the implementation
its `provider_type` names, so an `instance:model` selector, routing policy or
alias naming it keeps working.

The start asks each provider for its models, so the first start after the
upgrade can take a little longer. It offers every model the provider lists
switched on, priced or not, as the stored provider served it, on every path:
completions, routing-policy fallbacks, the pass-through endpoints and batches.
A moved provider starts with "Serve models not on its list" off. A model it does
not list is refused until you add it on the Hosted providers page, or turn that
setting on.

A provider the move cannot take is logged at startup, listed on the Hosted
providers page with the reason (`GET /api/v1/hosted-providers/migration`), and
serves nothing until it is added again. The reasons are:

- the provider lists no models;
- a hosted provider already holds the name;
- no configured `OTARI_SECRET_KEY` reads its key;
- the instance is named after another provider (`groq` declared as `openai`);
- it is a keyless Bedrock or SageMaker provider served on the server's own cloud
  role, which belongs under `providers:` in `config.yml` instead.

Once the cause is fixed, move it from the banner on the Hosted providers page,
or Discard it, since a row left behind serves nothing.

Two things behave differently once moved:

- A hosted provider is asked after an organization's own key, where a stored
  provider was asked before it, so an organization holding its own key is
  served on that key.
- A `config.yml` provider of the same name serves ahead of it, where a stored
  provider overrode the configured one.
