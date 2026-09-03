"""The DNS half of proving an organization controls a domain.

Split from `organization_domain_service` so the service's rules can be tested
without a resolver: this is the only part that leaves the process, and the only
part a test has to stand in for.

``dns.asyncresolver`` rather than ``dns.resolver``: the blocking resolver would
hold the event loop for the whole lookup, and a claimed domain whose
nameservers are gone is exactly the case that takes the full timeout. One
unreachable domain would otherwise stall every request this worker is serving.
"""

import dns.asyncresolver
import dns.exception

from gateway.log_config import logger

# How long a verification lookup may take. Short, because a person is waiting on
# it behind a button, and a missing record is the expected answer rather than an
# exceptional one: an admin who has just published a TXT record and clicks too
# early should get "not found yet" quickly rather than a request that hangs.
_RESOLVE_TIMEOUT_SECONDS = 5.0


async def resolve_txt_records(domain: str) -> list[str]:
    """Return the TXT strings published at ``domain``'s apex.

    A TXT record over 255 bytes arrives as several fragments and is joined back
    into the one string that was published, so a long record is not silently
    unmatchable.

    Every failure mode returns an empty list rather than raising: a domain that
    does not exist, one with no TXT records, unreachable nameservers, a timeout,
    and a resolver that is misconfigured all mean the same thing to the only
    caller ("the proof is not there"), and telling them apart would only give an
    admin a distinction they cannot act on differently. The reason is logged, so
    an operator debugging a verification that "should work" can still see that
    the deployment's own resolver was the problem rather than the record.
    """
    resolver = dns.asyncresolver.Resolver()
    resolver.timeout = _RESOLVE_TIMEOUT_SECONDS
    resolver.lifetime = _RESOLVE_TIMEOUT_SECONDS
    try:
        answers = await resolver.resolve(domain, "TXT")
    except dns.exception.DNSException as exc:
        logger.info("TXT lookup for %s found no usable answer: %s", domain, exc)
        return []
    return ["".join(part.decode("utf-8", "replace") for part in answer.strings) for answer in answers]


__all__ = ["resolve_txt_records"]
