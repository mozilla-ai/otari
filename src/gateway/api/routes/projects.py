from datetime import UTC, datetime
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.api.deps import get_db, verify_master_key
from gateway.models.entities import Budget, Project, RoutingPolicy
from gateway.services.budget_service import calculate_next_reset
from gateway.services.routing_policy_service import ACTIVE_ROUTING_POLICY_STATUS

router = APIRouter(prefix="/v1/projects", tags=["projects"])


class CreateProjectRequest(BaseModel):
    """Request model for creating a project."""

    project_id: str | None = Field(default=None, min_length=1)
    name: str | None = None
    routing_policy_id: str | None = None
    budget_id: str | None = None
    blocked: bool = False
    is_active: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)


class UpdateProjectRequest(BaseModel):
    """Request model for updating a project."""

    name: str | None = None
    routing_policy_id: str | None = None
    budget_id: str | None = None
    blocked: bool | None = None
    is_active: bool | None = None
    metadata: dict[str, Any] | None = None


class ProjectResponse(BaseModel):
    """Response model for project information."""

    project_id: str
    name: str | None
    routing_policy_id: str | None
    spend: float
    budget_id: str | None
    budget_started_at: str | None
    next_budget_reset_at: str | None
    blocked: bool
    is_active: bool
    metadata: dict[str, Any]
    created_at: str
    updated_at: str

    @classmethod
    def from_model(cls, project: Project) -> "ProjectResponse":
        """Create a response from an ORM model."""
        return cls(
            project_id=project.project_id,
            name=project.name,
            routing_policy_id=project.routing_policy_id,
            spend=float(project.spend),
            budget_id=project.budget_id,
            budget_started_at=project.budget_started_at.isoformat() if project.budget_started_at else None,
            next_budget_reset_at=project.next_budget_reset_at.isoformat() if project.next_budget_reset_at else None,
            blocked=bool(project.blocked),
            is_active=bool(project.is_active),
            metadata=dict(project.metadata_) if project.metadata_ else {},
            created_at=project.created_at.isoformat(),
            updated_at=project.updated_at.isoformat(),
        )


async def _ensure_policy_exists(db: AsyncSession, policy_id: str) -> None:
    policy = await db.get(RoutingPolicy, policy_id)
    if policy is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Routing policy '{policy_id}' not found",
        )
    if policy.status != ACTIVE_ROUTING_POLICY_STATUS:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"Routing policy '{policy_id}' is not active",
        )


async def _get_budget_or_404(db: AsyncSession, budget_id: str) -> Budget:
    budget = await db.get(Budget, budget_id)
    if budget is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Budget with id '{budget_id}' not found",
        )
    return budget


@router.post("", dependencies=[Depends(verify_master_key)])
async def create_project(
    request: CreateProjectRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> ProjectResponse:
    """Create a project."""
    if request.project_id:
        existing = await db.get(Project, request.project_id)
        if existing is not None:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Project '{request.project_id}' already exists",
            )
    if request.routing_policy_id:
        await _ensure_policy_exists(db, request.routing_policy_id)
    budget: Budget | None = None
    if request.budget_id:
        budget = await _get_budget_or_404(db, request.budget_id)

    project_kwargs: dict[str, Any] = {
        "name": request.name,
        "routing_policy_id": request.routing_policy_id,
        "budget_id": request.budget_id,
        "blocked": request.blocked,
        "is_active": request.is_active,
        "metadata_": request.metadata,
    }
    if request.project_id is not None:
        project_kwargs["project_id"] = request.project_id
    project = Project(**project_kwargs)
    if budget is not None:
        now = datetime.now(UTC)
        project.budget_started_at = now
        if budget.budget_duration_sec:
            project.next_budget_reset_at = calculate_next_reset(now, budget.budget_duration_sec)
        else:
            project.next_budget_reset_at = None
    db.add(project)
    try:
        await db.commit()
    except SQLAlchemyError:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database error",
        ) from None
    await db.refresh(project)
    return ProjectResponse.from_model(project)


@router.get("", dependencies=[Depends(verify_master_key)])
async def list_projects(
    db: Annotated[AsyncSession, Depends(get_db)],
    skip: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=1000)] = 100,
) -> list[ProjectResponse]:
    """List projects."""
    result = await db.execute(select(Project).order_by(Project.created_at.desc()).offset(skip).limit(limit))
    projects = result.scalars().all()
    return [ProjectResponse.from_model(project) for project in projects]


@router.get("/{project_id}", dependencies=[Depends(verify_master_key)])
async def get_project(
    project_id: str,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> ProjectResponse:
    """Get a project."""
    project = await db.get(Project, project_id)
    if project is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project '{project_id}' not found",
        )
    return ProjectResponse.from_model(project)


@router.patch("/{project_id}", dependencies=[Depends(verify_master_key)])
async def update_project(
    project_id: str,
    request: UpdateProjectRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> ProjectResponse:
    """Update a project."""
    project = await db.get(Project, project_id)
    if project is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project '{project_id}' not found",
        )

    payload = request.model_dump(exclude_unset=True)
    if "routing_policy_id" in payload and payload["routing_policy_id"] is not None:
        await _ensure_policy_exists(db, str(payload["routing_policy_id"]))
        project.routing_policy_id = str(payload["routing_policy_id"])
    elif "routing_policy_id" in payload:
        project.routing_policy_id = None
    if "budget_id" in payload and payload["budget_id"] is not None:
        budget = await _get_budget_or_404(db, str(payload["budget_id"]))
        project.budget_id = str(payload["budget_id"])
        now = datetime.now(UTC)
        project.budget_started_at = now
        if budget.budget_duration_sec:
            project.next_budget_reset_at = calculate_next_reset(now, budget.budget_duration_sec)
        else:
            project.next_budget_reset_at = None
    elif "budget_id" in payload:
        project.budget_id = None
        project.budget_started_at = None
        project.next_budget_reset_at = None
    if "name" in payload:
        project.name = payload["name"]
    if "blocked" in payload and payload["blocked"] is not None:
        project.blocked = bool(payload["blocked"])
    if "is_active" in payload and payload["is_active"] is not None:
        project.is_active = bool(payload["is_active"])
    if "metadata" in payload:
        project.metadata_ = dict(payload["metadata"] or {})

    try:
        await db.commit()
    except SQLAlchemyError:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database error",
        ) from None
    await db.refresh(project)
    return ProjectResponse.from_model(project)


@router.delete(
    "/{project_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    dependencies=[Depends(verify_master_key)],
)
async def delete_project(
    project_id: str,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> None:
    """Delete a project."""
    project = await db.get(Project, project_id)
    if project is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project '{project_id}' not found",
        )

    await db.delete(project)
    try:
        await db.commit()
    except SQLAlchemyError:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database error",
        ) from None
