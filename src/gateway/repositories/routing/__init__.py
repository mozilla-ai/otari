"""Data access for the routing domain's tables."""

from gateway.repositories.routing.agent_recommender_repository import AgentRecommenderRepository
from gateway.repositories.routing.routing_policy_repository import RoutingPolicyRepository

__all__ = ["AgentRecommenderRepository", "RoutingPolicyRepository"]
