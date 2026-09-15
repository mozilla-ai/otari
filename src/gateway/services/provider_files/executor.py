"""Stateless cleanup executor, with all scheduling authority in the control plane."""

import asyncio

from gateway.core.config import GatewayConfig
from gateway.services.provider_files.client import PlatformFilesClient
from gateway.services.provider_files.contracts import CleanupLease, WireModel
from gateway.services.provider_files.transport import provider_client, provider_error


class ClaimedCleanup(WireModel):
    lease: CleanupLease | None = None


async def run_provider_file_cleanup(config: GatewayConfig) -> None:
    client = PlatformFilesClient(config.platform["base_url"], config.platform_token or "", None)
    while True:
        try:
            claimed = await client.post("cleanup/claim", {"limit": 20}, ClaimedCleanup)
            if claimed.lease is not None:
                lease = claimed.lease
                results: dict[str, bool] = {}
                async with asyncio.timeout(240), provider_client(lease.account) as provider:
                    for item in lease.items:
                        try:
                            await provider.adelete_file(item.file_id, max_retries=0)
                            results[str(item.binding_id)] = True
                        except Exception as exc:
                            results[str(item.binding_id)] = provider_error(exc).status_code == 404
                await client.retry(
                    f"cleanup/{lease.id}/result",
                    {
                        "token": lease.token.get_secret_value(),
                        "results": results,
                    },
                    WireModel,
                )
        except Exception:
            # The lease remains durable and may be reclaimed after its deadline.
            pass
        await asyncio.sleep(60)
