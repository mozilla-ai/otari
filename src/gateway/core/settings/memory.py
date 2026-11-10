"""Settings for agent memory: whether the deployment offers it, the models it uses, and its limits."""

from typing import Annotated

from pydantic import BaseModel, Field

from gateway.core.settings_view import OMITTED, SettingsGroup, Shown


class MemorySettings(BaseModel):
    memory_enabled: Annotated[bool, Shown(SettingsGroup.TOOLS)] = Field(
        default=False,
        description=(
            "Offer agents a memory each workspace shares: agents propose what they learn, and a consolidator decides "
            "what is kept. Every entry and its embedding are stored sealed under the deployment's content key "
            "(trace_content_key_backend), and a workspace keeps nothing until its managers turn memory on. Off by "
            "default. Requires restart."
        ),
    )
    memory_embedding_model: Annotated[str | None, Shown(SettingsGroup.TOOLS)] = Field(
        default=None,
        description=(
            "The model every memory entry and search is embedded with, as provider:model. One model for the whole "
            "deployment, so every agent searches the same index whatever model it runs on. Changing it re-embeds "
            "nothing: entries embedded by another model are left out of search."
        ),
    )
    memory_consolidation_model: Annotated[str | None, Shown(SettingsGroup.TOOLS)] = Field(
        default=None,
        description=(
            "The model that decides what each proposal does to memory (add, update, merge, no change or conflict), "
            "as provider:model. It answers with a decision, never with text to store."
        ),
    )
    memory_max_entries_per_workspace: Annotated[int, OMITTED] = Field(
        default=5_000,
        gt=0,
        description="Active entries one workspace may hold. Search ranks them in memory, so this bounds its cost.",
    )
    memory_entry_max_length: Annotated[int, OMITTED] = Field(
        default=2_000,
        gt=0,
        le=16_000,
        description="Longest text, in characters, one proposal or entry may hold.",
    )
    memory_retention_days: Annotated[int, OMITTED] = Field(
        default=180,
        gt=0,
        description="Delete an entry, or a version it replaced, once nobody has used it for this many days.",
    )
    memory_key_rotation_days: Annotated[int, OMITTED] = Field(
        default=30,
        gt=0,
        description=(
            "How long one workspace data key seals new entries before a new one is made. Keys already made keep "
            "opening what they sealed."
        ),
    )
