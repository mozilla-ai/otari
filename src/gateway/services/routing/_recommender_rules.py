"""What makes a subagent recommender an operator adds one that can work.

The endpoint rules are the ones a ``decision_providers`` entry is held to,
from the configuration; the price rule is the recommender's own.
"""

from gateway.core.config import decision_endpoint_problem
from gateway.services.routing._recommender_pricing import RecommenderPrice


def check_recommender(*, provider: str, has_api_key: bool, api_base: str | None, price: RecommenderPrice) -> None:
    """Raise ``ValueError`` naming what is wrong with a recommender."""
    problem = decision_endpoint_problem(provider, has_api_key=has_api_key, api_base=api_base)
    if problem is not None:
        field, complaint = problem
        msg = f"{field} {complaint}"
        raise ValueError(msg)
    rates = (price.input_price_per_million, price.output_price_per_million)
    if any(rate < 0 for rate in rates):
        msg = "a price cannot be negative"
        raise ValueError(msg)
    # The rows' price is the only one a recommendation is charged, so a zero pair would make it free.
    if not any(rate > 0 for rate in rates):
        msg = "set an input or output token price above zero"
        raise ValueError(msg)
