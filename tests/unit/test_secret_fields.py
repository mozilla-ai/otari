"""Credential-shaped entries in a free-form settings dict never round-trip.

Four tables hold arbitrary operator-supplied JSON, and all four are places a
real credential legitimately lives. Three of them hand it to a provider SDK:
standalone Bedrock keeps its ``aws_secret_access_key`` in ``client_args``,
because any-llm's BedrockProvider never forwards ``api_key`` into the boto3
client it builds. ``OrgProviderKey`` has masked its own since it shipped;
otari-ai#1880 is the other two, where ``ProviderCredential.client_args``
returned a live AWS secret to anyone who could reach
``GET /api/v1/provider-credentials``. The fourth is
``organization_guardrails.validate_kwargs``, which a guardrail class can take
its vendor key in (otari-ai#2118).

Masking on read creates the other half of the problem, so it is asserted here
too: an editor that loads a row and saves it back would otherwise store the mask
over the credential it was never shown.
"""

import uuid
from datetime import UTC, datetime

from gateway.models.entities import OrganizationGuardrail, ProviderCredential, SearchToolCredential
from gateway.models.secret_fields import (
    _MAX_NESTING_DEPTH,
    REDACTED_VALUE,
    redact_secret_like_values,
    restore_redacted_values,
)
from gateway.services.tenancy.organization_guardrail_service import OrganizationGuardrailPublic


class TestRedactSecretLikeValues:
    def test_masks_by_key_name_and_passes_the_rest_through(self) -> None:
        assert redact_secret_like_values(
            {
                "region_name": "us-east-1",
                "timeout": 1800,
                "aws_access_key_id": "AKIAIOSFODNN7EXAMPLE",
                "aws_secret_access_key": "wJalrXUtnFEMI/K7MDENG",
            }
        ) == {
            "region_name": "us-east-1",
            "timeout": 1800,
            "aws_access_key_id": REDACTED_VALUE,
            "aws_secret_access_key": REDACTED_VALUE,
        }

    def test_matches_a_substring_so_a_variant_spelling_cannot_slip_through(self) -> None:
        # Substring and not an exact-name allow-list: an operator names a client
        # kwarg however the SDK expects it, and a fixed set would miss a variant.
        submitted = {"projectAuthorizationToken": "t", "X-Api-Key": "k", "service_account_credentials": "c"}

        assert redact_secret_like_values(submitted) == dict.fromkeys(submitted, REDACTED_VALUE)

    def test_none_stays_none(self) -> None:
        assert redact_secret_like_values(None) is None


class TestNestedRedaction:
    """otari#1125: the walk stopped at depth one, so a credential one level down
    was returned in clear. Every cell here pairs the new masking with the restore
    that has to move with it: a mask that reaches a nested entry while the
    restore still walks one level writes ``***`` over the credential on the next
    PATCH, which is worse than the leak it was fixing."""

    def test_a_nested_credential_is_masked(self) -> None:
        assert redact_secret_like_values({"headers": {"api_key": "secret"}, "region_name": "eu-west-1"}) == {
            "headers": {"api_key": REDACTED_VALUE},
            "region_name": "eu-west-1",
        }

    def test_a_credential_inside_a_list_of_objects_is_masked(self) -> None:
        assert redact_secret_like_values({"extra_headers": [{"name": "x", "token": "live"}, {"name": "y"}]}) == {
            "extra_headers": [{"name": "x", "token": REDACTED_VALUE}, {"name": "y"}]
        }

    def test_a_matching_key_masks_its_whole_subtree(self) -> None:
        # Same thing a matching top-level key has always done to a non-scalar:
        # the name is the signal, so nothing under it is shown either.
        assert redact_secret_like_values({"credentials": {"user": "bob", "passphrase": "p"}}) == {
            "credentials": REDACTED_VALUE
        }

    def test_a_bare_mask_in_a_list_is_never_produced(self) -> None:
        # A list element has no key to match on, so masking one would be a guess.
        # `restore` leans on this: an element that looks like the mask came from
        # the caller and means itself.
        assert redact_secret_like_values({"values": ["***", "plain"]}) == {"values": ["***", "plain"]}

    def test_deep_nesting_is_masked_rather_than_walked_forever(self) -> None:
        # Fail closed past the bound: the alternatives are a RecursionError
        # turning a read into a 500, or a depth the masking never reaches.
        deep: dict[str, object] = {"leaf": "visible"}
        for _ in range(40):
            deep = {"nest": deep}

        assert redact_secret_like_values(deep) != deep
        assert REDACTED_VALUE in str(redact_secret_like_values(deep))


class TestNestedRoundTrip:
    def test_a_nested_credential_survives_an_edit_of_its_sibling(self) -> None:
        # The failure this prevents: the dashboard loads a row, changes one
        # visible field, and saves the whole object back.
        stored = {"headers": {"api_key": "live-secret", "trace": "off"}}
        echoed = redact_secret_like_values(stored)
        assert echoed is not None
        echoed["headers"]["trace"] = "on"

        assert restore_redacted_values(echoed, stored) == {"headers": {"api_key": "live-secret", "trace": "on"}}

    def test_a_masked_subtree_is_restored_whole(self) -> None:
        stored = {"credentials": {"user": "bob", "passphrase": "p"}}
        echoed = redact_secret_like_values(stored)

        assert restore_redacted_values(echoed, stored) == stored

    def test_a_credential_inside_a_list_survives_the_round_trip(self) -> None:
        stored = {"extra_headers": [{"name": "x", "token": "live"}, {"name": "y"}]}
        echoed = redact_secret_like_values(stored)

        assert restore_redacted_values(echoed, stored) == stored

    def test_entries_that_mask_to_the_same_thing_are_not_guessed_between(self) -> None:
        # Both stored entries mask to {"token": "***"}, so the one that was kept
        # cannot be told from the one that was dropped. Guessing would put a
        # credential under a different header; the mask is kept literally.
        stored = {"extra_headers": [{"token": "live-a"}, {"token": "live-b"}]}
        submitted = {"extra_headers": [{"token": REDACTED_VALUE}]}

        assert restore_redacted_values(submitted, stored) == {"extra_headers": [{"token": REDACTED_VALUE}]}


class TestListElementIdentity:
    """A list element has no key, so it is paired with the stored element it IS.

    The editor echoes each entry masked; an entry the caller did not edit is
    therefore identical to its stored entry's masked form, wherever it moved.
    """

    STORED = {"extra_headers": [{"name": "x", "token": "live-a"}, {"name": "y", "token": "live-b"}]}

    def test_reordering_entries_keeps_each_entry_its_own_credential(self) -> None:
        # The report on #1129: pairing by index handed each entry the token of
        # whatever used to sit at its position.
        submitted = {"extra_headers": [{"name": "y", "token": REDACTED_VALUE}, {"name": "x", "token": REDACTED_VALUE}]}

        assert restore_redacted_values(submitted, self.STORED) == {
            "extra_headers": [{"name": "y", "token": "live-b"}, {"name": "x", "token": "live-a"}]
        }

    def test_appending_an_entry_keeps_the_credentials_of_the_others(self) -> None:
        submitted = {
            "extra_headers": [
                {"name": "x", "token": REDACTED_VALUE},
                {"name": "y", "token": REDACTED_VALUE},
                {"name": "z", "token": "live-c"},
            ]
        }

        assert restore_redacted_values(submitted, self.STORED) == {
            "extra_headers": [
                {"name": "x", "token": "live-a"},
                {"name": "y", "token": "live-b"},
                {"name": "z", "token": "live-c"},
            ]
        }

    def test_dropping_an_entry_keeps_the_credential_of_the_one_left(self) -> None:
        submitted = {"extra_headers": [{"name": "y", "token": REDACTED_VALUE}]}

        assert restore_redacted_values(submitted, self.STORED) == {"extra_headers": [{"name": "y", "token": "live-b"}]}

    def test_editing_one_entry_in_place_keeps_its_credential(self) -> None:
        # The dashboard flow the restore exists for: change one visible field of
        # one entry and save the whole object back.
        submitted = {"extra_headers": [{"name": "x", "token": REDACTED_VALUE}, {"name": "y2", "token": REDACTED_VALUE}]}

        assert restore_redacted_values(submitted, self.STORED) == {
            "extra_headers": [{"name": "x", "token": "live-a"}, {"name": "y2", "token": "live-b"}]
        }

    def test_an_edited_entry_that_also_moved_is_not_paired_by_position(self) -> None:
        # "x" moved to index 1, so index 0 no longer means the entry stored
        # there. The edited entry cannot be identified and keeps the mask
        # rather than taking x's token.
        submitted = {"extra_headers": [{"name": "y2", "token": REDACTED_VALUE}, {"name": "x", "token": REDACTED_VALUE}]}

        assert restore_redacted_values(submitted, self.STORED) == {
            "extra_headers": [{"name": "y2", "token": REDACTED_VALUE}, {"name": "x", "token": "live-a"}]
        }

    def test_two_edited_entries_are_not_paired_by_position(self) -> None:
        # Every visible field was edited, so nothing the entries still share
        # shows whether they also moved, and neither is guessed at.
        submitted = {
            "extra_headers": [{"name": "y2", "token": REDACTED_VALUE}, {"name": "x2", "token": REDACTED_VALUE}]
        }

        assert restore_redacted_values(submitted, self.STORED) == submitted

    LABELED = {
        "extra_headers": [
            {"name": "a", "token": "secretA", "label": "old-a"},
            {"name": "b", "token": "secretB", "label": "old-b"},
        ]
    }

    def test_editing_two_entries_in_place_keeps_both_credentials(self) -> None:
        # The review on #1129: relabeling both entries lost both tokens, where
        # pairing by index had kept them. The names they kept identify them.
        submitted = {
            "extra_headers": [
                {"name": "a", "token": REDACTED_VALUE, "label": "new-a"},
                {"name": "b", "token": REDACTED_VALUE, "label": "new-b"},
            ]
        }

        assert restore_redacted_values(submitted, self.LABELED) == {
            "extra_headers": [
                {"name": "a", "token": "secretA", "label": "new-a"},
                {"name": "b", "token": "secretB", "label": "new-b"},
            ]
        }

    def test_two_edited_entries_that_also_moved_keep_their_own_credentials(self) -> None:
        # Pairing the edited entries by index would cross the tokens here, as it
        # did for the unedited swap this PR started from.
        submitted = {
            "extra_headers": [
                {"name": "b", "token": REDACTED_VALUE, "label": "new-b"},
                {"name": "a", "token": REDACTED_VALUE, "label": "new-a"},
            ]
        }

        assert restore_redacted_values(submitted, self.LABELED) == {
            "extra_headers": [
                {"name": "b", "token": "secretB", "label": "new-b"},
                {"name": "a", "token": "secretA", "label": "new-a"},
            ]
        }

    def test_a_field_every_entry_shares_does_not_tell_edited_entries_apart(self) -> None:
        stored = {
            "extra_headers": [
                {"scheme": "bearer", "name": "x", "token": "live-a"},
                {"scheme": "bearer", "name": "y", "token": "live-b"},
            ]
        }
        submitted = {
            "extra_headers": [
                {"scheme": "bearer", "name": "x2", "token": REDACTED_VALUE},
                {"scheme": "bearer", "name": "y2", "token": REDACTED_VALUE},
            ]
        }

        assert restore_redacted_values(submitted, stored) == submitted

    def test_one_stored_credential_is_never_handed_to_two_entries(self) -> None:
        # Both edited entries are closest to the first stored entry. Only the
        # closer one takes its token; the other is not the second stored entry
        # either, so it keeps the mask.
        stored = {
            "extra_headers": [
                {"a": 1, "b": 1, "c": 1, "d": 1, "token": "live-a"},
                {"a": 2, "b": 2, "c": 2, "d": 2, "token": "live-b"},
            ]
        }
        submitted = {
            "extra_headers": [
                {"a": 1, "b": 1, "c": 7, "d": 7, "token": REDACTED_VALUE},
                {"a": 1, "b": 1, "c": 1, "d": 9, "token": REDACTED_VALUE},
            ]
        }

        assert restore_redacted_values(submitted, stored) == {
            "extra_headers": [
                {"a": 1, "b": 1, "c": 7, "d": 7, "token": REDACTED_VALUE},
                {"a": 1, "b": 1, "c": 1, "d": 9, "token": "live-a"},
            ]
        }

    def test_duplicate_entries_with_different_credentials_are_not_guessed_between(self) -> None:
        stored = {"extra_headers": [{"name": "x", "token": "live-a"}, {"name": "x", "token": "live-b"}]}
        submitted = {"extra_headers": [{"name": "x", "token": REDACTED_VALUE}, {"name": "x", "token": REDACTED_VALUE}]}

        assert restore_redacted_values(submitted, stored) == submitted

    def test_a_real_nested_value_still_replaces_the_stored_one(self) -> None:
        # The control for the whole pairing: restoring must not mean "the caller
        # can never change a nested credential".
        stored = {"headers": {"api_key": "old"}}

        assert restore_redacted_values({"headers": {"api_key": "new"}}, stored) == {"headers": {"api_key": "new"}}

    def test_a_nested_entry_the_caller_dropped_stays_dropped(self) -> None:
        stored = {"headers": {"api_key": "live", "trace": "on"}}

        assert restore_redacted_values({"headers": {"trace": "on"}}, stored) == {"headers": {"trace": "on"}}

    def test_the_mask_at_the_depth_bound_with_nothing_stored_stays_the_mask(self) -> None:
        # The shallow rule is that a mask with no stored counterpart is taken
        # literally, because dropping the caller's entry is worse than keeping a
        # placeholder. The bound has to answer the same way: handing back the
        # absent stored value writes null over what the caller actually sent.
        submitted: object = REDACTED_VALUE
        for _ in range(_MAX_NESTING_DEPTH):
            submitted = {"a": submitted}

        restored = restore_redacted_values(submitted, {"unrelated": "x"})  # type: ignore[arg-type]

        node: object = restored
        for _ in range(_MAX_NESTING_DEPTH):
            assert isinstance(node, dict)
            node = node["a"]
        assert node == REDACTED_VALUE

    def test_a_list_at_the_depth_bound_is_not_overwritten_by_its_own_mask(self) -> None:
        # The depth bound is the ONLY thing that masks a bare list element:
        # nothing else does, because an element has no key name to match on.
        # The restore walk pairs a mask with its stored value BY KEY, so an
        # element had no way back and an unchanged save wrote *** over the
        # credential. The window is one level wide: a list one below the bound
        # has its elements masked individually, a list AT the bound is masked
        # whole as its parent's value and comes back through the key pairing.
        def nest(depth: int, leaf: object) -> object:
            node = leaf
            for _ in range(depth):
                node = {"a": node}
            return node

        for list_depth in (_MAX_NESTING_DEPTH - 2, _MAX_NESTING_DEPTH - 1, _MAX_NESTING_DEPTH):
            stored = nest(list_depth, ["live-token", "second"])
            echoed = redact_secret_like_values(stored)  # type: ignore[arg-type]

            assert restore_redacted_values(echoed, stored) == stored, f"list at depth {list_depth}"  # type: ignore[arg-type]


class TestRestoreRedactedValues:
    def test_the_mask_echoed_back_keeps_the_stored_value(self) -> None:
        stored = {"region_name": "us-east-1", "aws_secret_access_key": "wJalrXUtnFEMI"}
        submitted = {"region_name": "eu-west-1", "aws_secret_access_key": REDACTED_VALUE}

        assert restore_redacted_values(submitted, stored) == {
            "region_name": "eu-west-1",
            "aws_secret_access_key": "wJalrXUtnFEMI",
        }

    def test_a_real_new_value_replaces_the_stored_one(self) -> None:
        stored = {"aws_secret_access_key": "old"}
        assert restore_redacted_values({"aws_secret_access_key": "new"}, stored) == {"aws_secret_access_key": "new"}

    def test_a_dropped_entry_stays_dropped(self) -> None:
        # The caller is submitting the whole object, so an entry it left out is a
        # removal and not something to put back.
        assert restore_redacted_values({"region_name": "us-east-1"}, {"api_key": "live"}) == {
            "region_name": "us-east-1"
        }

    def test_the_mask_on_a_key_that_is_not_stored_is_taken_literally(self) -> None:
        # Nothing to restore, so there is no stored value to prefer; storing the
        # mask is the only answer available and beats dropping the entry.
        assert restore_redacted_values({"api_key": REDACTED_VALUE}, None) == {"api_key": REDACTED_VALUE}


class TestSerializers:
    def test_provider_credential_masks_its_client_args(self) -> None:
        row = ProviderCredential(
            instance="bedrock",
            provider_type="bedrock",
            last4="MPLE",
            client_args={"region_name": "us-east-1", "aws_secret_access_key": "wJalrXUtnFEMI"},
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )

        public = row.to_public_dict()

        assert public["client_args"] == {"region_name": "us-east-1", "aws_secret_access_key": REDACTED_VALUE}
        # The encrypted key was never on the wire to begin with; only last4 is.
        assert "encrypted_api_key" not in public
        assert public["last4"] == "MPLE"

    def test_search_tool_credential_masks_its_options(self) -> None:
        row = SearchToolCredential(
            name="searxng",
            provider="searxng",
            options={"engines": "google", "secondary_api_key": "live-key"},
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )

        assert row.to_public_dict()["options"] == {"engines": "google", "secondary_api_key": REDACTED_VALUE}

    def test_an_empty_settings_dict_still_serializes_as_an_object(self) -> None:
        # The column is non-null and the response type is a dict, so masking must
        # not turn "no options" into null.
        row = ProviderCredential(instance="openai", client_args={}, created_at=datetime.now(UTC))
        assert row.to_public_dict()["client_args"] == {}

    def test_organization_guardrail_masks_its_validate_kwargs(self) -> None:
        row = OrganizationGuardrail(
            id=uuid.uuid4(),
            organization_id=uuid.uuid4(),
            profile="patronus",
            mode="block",
            on_unavailable="block",
            validate_kwargs={"threshold": 0.8, "patronus_api_key": "live-key"},
            enabled=True,
            applies_to_all_workspaces=True,
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )

        public = OrganizationGuardrailPublic.from_model(row, workspace_ids=[])

        assert public.validate_kwargs == {"threshold": 0.8, "patronus_api_key": REDACTED_VALUE}

    def test_organization_guardrail_with_no_validate_kwargs_stays_null(self) -> None:
        # The column is nullable here, unlike the two above, and null is what the
        # API stores for "no kwargs": masking must not turn it into an object.
        row = OrganizationGuardrail(
            id=uuid.uuid4(),
            organization_id=uuid.uuid4(),
            profile="prompt-injection",
            mode="monitor",
            on_unavailable="block",
            validate_kwargs=None,
            enabled=True,
            applies_to_all_workspaces=True,
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )

        assert OrganizationGuardrailPublic.from_model(row, workspace_ids=[]).validate_kwargs is None
