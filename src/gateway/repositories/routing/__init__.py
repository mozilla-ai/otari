"""Data access for the routing domain's tables."""

from gateway.repositories.routing.agent_recommender_repository import (
    AgentRecommenderRepository,
    AgentRecommenderSwitchConflict,
)

__all__ = ["AgentRecommenderRepository", "AgentRecommenderSwitchConflict"]
