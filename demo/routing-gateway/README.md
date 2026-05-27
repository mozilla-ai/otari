# Routing Gateway Demo

This demo exercises the standalone routing control plane without requiring an
otari.ai platform service. The default walkthrough uses management and dry-run
endpoints, so it does not spend tokens. You can opt into one real provider call
to populate route traces.

## Start

```bash
cd demo/routing-gateway
cp .env.example .env
./start.sh -d
```

If you are testing unreleased local changes, build the image first:

```bash
docker build -t mzdotai/otari:latest ../..
```

## Run The Flow

```bash
./demo_flow.sh
```

The flow:

- creates a lowest-cost default routing policy
- attaches a uniquely named project to that policy
- dry-runs `model: "default_routing"`
- creates a 25 percent canary policy for `tenant=vip`
- clones a policy into a draft
- dry-runs the draft by explicit `policy_id`
- updates and rolls back a policy by applying revision 1
- prints route trace summary buckets

To send one real provider request, put a provider key in `.env` and set:

```bash
RUN_PROVIDER_CALL=1
```

Then rerun `./demo_flow.sh`.

Set `DEMO_ID=my-demo` before running if you want stable project and user IDs
instead of the timestamped default.

## Stop

```bash
./stop.sh
```

Use `./stop.sh -v` to remove the demo Postgres volume.
