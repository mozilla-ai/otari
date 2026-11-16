"""Reading the workspace id a stored workspace assignment names."""

import uuid

from gateway.models.organizations import WorkspaceAssignment


def workspace_id_of(assignment: WorkspaceAssignment) -> uuid.UUID | None:
    """Return the workspace id an assignment names, or None when the stored value is not a UUID."""
    try:
        return uuid.UUID(str(assignment.get("workspace_id")))
    except ValueError:
        return None
