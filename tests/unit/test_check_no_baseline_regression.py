"""Tests for the non-regression gate over the architecture check's baselines."""

from __future__ import annotations

import importlib.util
import io
import sys
from pathlib import Path
from types import ModuleType

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "check_no_baseline_regression.py"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("check_no_baseline_regression", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


check = _load()


def _architecture_check() -> str:
    return (Path(__file__).resolve().parents[2] / "scripts" / "check_architecture.py").read_text(encoding="utf-8")


def test_a_tree_that_changed_nothing_is_clean() -> None:
    source = "SIZE_BASELINE = 3\n"
    assert check.check_baselines(source, source) == []


def test_the_real_check_measured_against_itself_is_clean() -> None:
    source = _architecture_check()
    assert check.check_baselines(source, source) == []


def test_every_baseline_the_real_check_names_is_read() -> None:
    found = check.baselines(_architecture_check()).rooms
    assert sorted(name for name in found if name.endswith(check.BASELINE_SUFFIX)) == [
        "COMPOSITION_ROOT_BEHAVIOR_BASELINE",
        "COMPOSITION_ROOT_SIZE_BASELINE",
        "DOMAIN_NAME_BASELINE",
        "FLAT_MODULE_BASELINE",
        "LISTENER_DEFAULT_BASELINE",
        "NON_DOMAIN_SERVICE_IMPORT_BASELINE",
        "ROUTE_DATABASE_IMPORT_BASELINE",
        "SAVEPOINT_BASELINE",
        "SERVICE_DATABASE_IMPORT_BASELINE",
        "SERVICE_ROOT_EXPORT_BASELINE",
        "TRANSACTION_CONTROL_BASELINE",
    ]


def test_every_per_entry_baseline_the_real_check_records_is_read() -> None:
    found = check.baselines(_architecture_check()).rooms
    nested = [name for name in found if not name.endswith(check.BASELINE_SUFFIX)]
    assert nested, "the model access map records a baseline per model"
    assert all(name.startswith("MODEL_ACCESS[") and name.endswith("['baseline']") for name in nested)


def test_every_constant_the_real_check_records_a_baseline_in_is_read_in_full() -> None:
    found = check.baselines(_architecture_check())
    assert sorted(set(found.owners.values()) & found.unread) == []


def test_no_baseline_the_real_check_records_is_unreadable() -> None:
    found = check.baselines(_architecture_check()).rooms
    assert [name for name, room in found.items() if isinstance(room, check.Unreadable)] == []


def test_a_count_that_rises_is_flagged() -> None:
    assert check.check_baselines("SIZE_BASELINE = 3\n", "SIZE_BASELINE = 4\n") == [
        "SIZE_BASELINE rises from 3 to 4; a baseline records room already given up, so it may only fall"
    ]


def test_a_count_that_falls_is_clean() -> None:
    assert check.check_baselines("SIZE_BASELINE = 3\n", "SIZE_BASELINE = 2\n") == []


def test_an_entry_added_to_a_baseline_is_flagged() -> None:
    before = 'LEAK_BASELINE = ("one.py",)\n'
    after = 'LEAK_BASELINE = ("one.py", "two.py")\n'
    assert check.check_baselines(before, after) == [
        "LEAK_BASELINE gains 'two.py'; a baseline records room already given up, so it may only lose entries"
    ]


def test_an_entry_removed_from_a_baseline_is_clean() -> None:
    before = 'LEAK_BASELINE = ("one.py", "two.py")\n'
    after = 'LEAK_BASELINE = ("one.py",)\n'
    assert check.check_baselines(before, after) == []


def test_a_nested_entry_added_to_a_baseline_is_flagged() -> None:
    before = 'PAIR_BASELINE = (("one.py", "A"),)\n'
    after = 'PAIR_BASELINE = (("one.py", "A"), ("two.py", "B"))\n'
    assert check.check_baselines(before, after) == [
        "PAIR_BASELINE gains ('two.py', 'B'); a baseline records room already given up, so it may only lose entries"
    ]


def test_a_key_added_to_a_baseline_is_flagged() -> None:
    before = 'MAP_BASELINE = {"one": 1}\n'
    after = 'MAP_BASELINE = {"one": 1, "two": 2}\n'
    assert check.check_baselines(before, after) == [
        "MAP_BASELINE gains the entry 'two'; a baseline records room already given up, so it may only lose entries"
    ]


def test_a_key_removed_from_a_baseline_is_clean() -> None:
    before = 'MAP_BASELINE = {"one": 1, "two": 2}\n'
    after = 'MAP_BASELINE = {"one": 1}\n'
    assert check.check_baselines(before, after) == []


def test_a_count_under_a_key_that_rises_is_flagged() -> None:
    before = 'MAP_BASELINE = {"one": 1}\n'
    after = 'MAP_BASELINE = {"one": 2}\n'
    assert check.check_baselines(before, after) == [
        "MAP_BASELINE['one'] rises from 1 to 2; a baseline records room already given up, so it may only fall"
    ]


def test_an_entry_under_a_key_that_is_added_is_flagged() -> None:
    before = 'MAP_BASELINE = {"one": ("A",)}\n'
    after = 'MAP_BASELINE = {"one": ("A", "B")}\n'
    assert check.check_baselines(before, after) == [
        "MAP_BASELINE['one'] gains 'B'; a baseline records room already given up, so it may only lose entries"
    ]


def test_a_baseline_that_is_new_is_clean() -> None:
    assert check.check_baselines("SIZE_BASELINE = 3\n", "SIZE_BASELINE = 3\nLEAK_BASELINE = ()\n") == []


def test_a_baseline_that_is_gone_while_it_still_holds_room_is_flagged() -> None:
    before = 'LEAK_BASELINE = ("one.py",)\nSIZE_BASELINE = 3\n'
    after = "SIZE_BASELINE = 3\n"
    assert check.check_baselines(before, after) == [
        "LEAK_BASELINE is gone while it still held 1 entry; empty it before the rule that reads it goes"
    ]


def test_a_baseline_that_is_gone_having_been_emptied_is_clean() -> None:
    assert check.check_baselines("LEAK_BASELINE = ()\nSIZE_BASELINE = 3\n", "SIZE_BASELINE = 3\n") == []


def test_a_count_that_is_gone_having_reached_zero_is_clean() -> None:
    assert check.check_baselines("SIZE_BASELINE = 0\nLEAK_BASELINE = ()\n", "LEAK_BASELINE = ()\n") == []


def test_a_baseline_that_changes_shape_is_flagged() -> None:
    assert check.check_baselines("SIZE_BASELINE = 3\n", 'SIZE_BASELINE = ("one.py",)\n') == [
        "SIZE_BASELINE changes from a count to a list of entries; a reader cannot tell what room it gives back"
    ]


def test_a_constant_that_is_not_a_baseline_is_not_read() -> None:
    assert check.check_baselines("SIZE_BASELINE = 3\nLIMIT = 3\n", "SIZE_BASELINE = 3\nLIMIT = 4\n") == []


def test_an_annotated_baseline_is_read() -> None:
    before = 'MAP_BASELINE: dict[str, int] = {"one": 1}\n'
    after = 'MAP_BASELINE: dict[str, int] = {"one": 2}\n'
    assert check.check_baselines(before, after) == [
        "MAP_BASELINE['one'] rises from 1 to 2; a baseline records room already given up, so it may only fall"
    ]


def test_a_baseline_that_is_not_a_literal_is_reported() -> None:
    assert check.check_baselines("SIZE_BASELINE = 3\n", "SIZE_BASELINE = len(OTHER)\n") == [
        "SIZE_BASELINE is not written out in full; a baseline is read without running the check"
    ]


def test_a_baseline_written_again_at_module_scope_is_reported() -> None:
    after = "SIZE_BASELINE = 3\nSIZE_BASELINE += 3\n"
    assert check.check_baselines("SIZE_BASELINE = 3\n", after) == [
        "SIZE_BASELINE is written at module scope more than once, so the last value decides; "
        "a baseline is read without running the check"
    ]


def test_a_baseline_written_again_under_module_level_control_flow_is_reported() -> None:
    after = "SIZE_BASELINE = 3\nif OTHER:\n    SIZE_BASELINE = 90\n"
    assert check.check_baselines("SIZE_BASELINE = 3\n", after) == [
        "SIZE_BASELINE is written at module scope more than once, so the last value decides; "
        "a baseline is read without running the check"
    ]


def test_a_baseline_recording_a_kind_this_check_does_not_read_is_reported() -> None:
    assert check.check_baselines("SIZE_BASELINE = 3\n", "SIZE_BASELINE = None\n") == [
        "SIZE_BASELINE records a kind of value this check does not read; a baseline is read without running the check"
    ]


def test_a_baseline_recording_a_kind_this_check_does_not_read_is_reported_when_gone() -> None:
    before = "SIZE_BASELINE = None\nLEAK_BASELINE = ()\n"
    assert check.check_baselines(before, "LEAK_BASELINE = ()\n") == [
        "SIZE_BASELINE is gone, and whether it was empty cannot be read, because it records a kind of value "
        "this check does not read"
    ]


def test_a_baseline_that_arrives_unreadable_is_reported() -> None:
    after = "SIZE_BASELINE = 3\nLEAK_BASELINE = len(OTHER)\n"
    assert check.check_baselines("SIZE_BASELINE = 3\n", after) == [
        "LEAK_BASELINE arrives and is not written out in full; a baseline is read without running the check"
    ]


def test_a_baseline_that_arrives_written_out_in_full_is_clean() -> None:
    assert check.check_baselines("SIZE_BASELINE = 3\n", 'SIZE_BASELINE = 3\nLEAK_BASELINE = ("one.py",)\n') == []


def test_an_empty_earlier_revision_is_reported() -> None:
    assert check.check_baselines("", "SIZE_BASELINE = 3\n") == [
        "the earlier revision is empty, so there is nothing to measure against"
    ]


def test_a_whitespace_only_earlier_revision_is_reported() -> None:
    assert check.check_baselines("\n  \n", "SIZE_BASELINE = 3\n") == [
        "the earlier revision is empty, so there is nothing to measure against"
    ]


def test_a_first_baseline_after_the_last_one_retired_is_clean() -> None:
    assert check.check_baselines("LIMIT = 3\n", 'LIMIT = 3\nLEAK_BASELINE = ("one.py",)\n') == []


def test_a_baseline_recorded_per_entry_is_read() -> None:
    before = 'SIZE_BASELINE = 3\nACCESS = {"a.B": {"repository": "r.py", "baseline": ("one.py",)}}\n'
    after = 'SIZE_BASELINE = 3\nACCESS = {"a.B": {"repository": "r.py", "baseline": ("one.py", "two.py")}}\n'
    assert check.check_baselines(before, after) == [
        "ACCESS['a.B']['baseline'] gains 'two.py'; a baseline records room already given up, "
        "so it may only lose entries"
    ]


def test_a_repository_beside_a_baseline_recorded_per_entry_is_not_read() -> None:
    before = 'SIZE_BASELINE = 3\nACCESS = {"a.B": {"repository": "one.py", "baseline": ()}}\n'
    after = 'SIZE_BASELINE = 3\nACCESS = {"a.B": {"repository": "two.py", "baseline": ()}}\n'
    assert check.check_baselines(before, after) == []


def test_every_way_one_change_gives_back_room_is_reported() -> None:
    before = 'SIZE_BASELINE = 3\nLEAK_BASELINE = ("one.py",)\n'
    after = 'SIZE_BASELINE = 4\nLEAK_BASELINE = ("one.py", "two.py")\n'
    assert check.check_baselines(before, after) == [
        "LEAK_BASELINE gains 'two.py'; a baseline records room already given up, so it may only lose entries",
        "SIZE_BASELINE rises from 3 to 4; a baseline records room already given up, so it may only fall",
    ]


def test_a_tree_whose_check_does_not_parse_is_reported() -> None:
    assert check.check_baselines("SIZE_BASELINE = 3\n", "SIZE_BASELINE = (\n") == [
        "the check does not parse, so no baseline can be read from it"
    ]


def _write_pair(tmp_path: Path, before: str, after: str) -> tuple[Path, Path]:
    base = tmp_path / "base.py"
    head = tmp_path / "head.py"
    base.write_text(before, encoding="utf-8")
    head.write_text(after, encoding="utf-8")
    return base, head


def test_the_command_passes_a_change_that_gives_back_no_room(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    base, head = _write_pair(tmp_path, "SIZE_BASELINE = 3\n", "SIZE_BASELINE = 2\n")
    assert check.main(["--base", str(base), "--head", str(head)]) == 0
    assert "No baseline gives back room" in capsys.readouterr().out


def test_the_command_fails_a_change_that_gives_back_room(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    base, head = _write_pair(tmp_path, "SIZE_BASELINE = 3\n", "SIZE_BASELINE = 4\n")
    assert check.main(["--base", str(base), "--head", str(head)]) == 1
    assert "SIZE_BASELINE rises from 3 to 4" in capsys.readouterr().out


def test_the_command_reports_a_missing_file(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    base, head = _write_pair(tmp_path, "SIZE_BASELINE = 3\n", "SIZE_BASELINE = 3\n")
    base.unlink()
    assert check.main(["--base", str(base), "--head", str(head)]) == 1
    assert str(base) in capsys.readouterr().out


def test_a_count_that_is_gone_while_it_still_stands_above_zero_is_flagged() -> None:
    before = "SIZE_BASELINE = 3\nLEAK_BASELINE = ()\n"
    assert check.check_baselines(before, "LEAK_BASELINE = ()\n") == [
        "SIZE_BASELINE is gone while it still stood at 3; take it to zero before the rule that reads it"
    ]


def test_a_baseline_written_out_for_the_first_time_cannot_be_measured() -> None:
    assert check.check_baselines("SIZE_BASELINE = len(OTHER)\n", "SIZE_BASELINE = 99\n") == [
        "SIZE_BASELINE cannot be measured, because at the branch point it is not written out in full"
    ]


def test_a_baseline_that_was_never_written_out_and_is_gone_cannot_be_measured() -> None:
    before = "SIZE_BASELINE = len(OTHER)\nLEAK_BASELINE = ()\n"
    assert check.check_baselines(before, "LEAK_BASELINE = ()\n") == [
        "SIZE_BASELINE is gone, and whether it was empty cannot be read, because it is not written out in full"
    ]


def test_the_command_reads_a_revision_from_standard_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    head = tmp_path / "head.py"
    head.write_text("SIZE_BASELINE = 4\n", encoding="utf-8")
    monkeypatch.setattr("sys.stdin", io.StringIO("SIZE_BASELINE = 3\n"))
    assert check.main(["--base", "-", "--head", str(head)]) == 1
    assert "SIZE_BASELINE rises from 3 to 4" in capsys.readouterr().out


def test_a_bare_annotation_beside_a_declaration_is_not_a_second_write() -> None:
    assert check.check_baselines("SIZE_BASELINE = 3\n", "SIZE_BASELINE: int\nSIZE_BASELINE = 2\n") == []


def test_a_write_to_one_field_of_a_map_that_owns_baselines_is_reported() -> None:
    before = 'SIZE_BASELINE = 3\nACCESS = {"a.B": {"baseline": ("one.py",)}}\n'
    after = before + 'ACCESS["a.B"]["baseline"] = ("one.py", "two.py")\n'
    assert check.check_baselines(before, after) == [
        "ACCESS['a.B']['baseline'] is written at module scope more than once, so the last value decides; "
        "a baseline is read without running the check"
    ]


def test_a_map_that_owns_baselines_assigned_again_is_reported() -> None:
    before = 'SIZE_BASELINE = 3\nACCESS = {"a.B": {"baseline": ("one.py",)}}\n'
    after = before + "ACCESS = {}\n"
    assert check.check_baselines(before, after) == [
        "ACCESS['a.B']['baseline'] is written at module scope more than once, so the last value decides; "
        "a baseline is read without running the check"
    ]


def test_a_count_replaced_by_a_boolean_is_reported() -> None:
    assert check.check_baselines("SIZE_BASELINE = 1\n", "SIZE_BASELINE = True\n") == [
        "SIZE_BASELINE records a kind of value this check does not read; a baseline is read without running the check"
    ]


def test_a_boolean_under_a_key_is_reported() -> None:
    assert check.check_baselines('MAP_BASELINE = {"one": 1}\n', 'MAP_BASELINE = {"one": True}\n') == [
        "MAP_BASELINE records a kind of value this check does not read; a baseline is read without running the check"
    ]


def test_a_baseline_written_only_under_module_level_control_flow_is_reported() -> None:
    assert check.check_baselines("SIZE_BASELINE = 0\n", "if OTHER:\n    SIZE_BASELINE = 99\n") == [
        "SIZE_BASELINE is written at module scope without a declaration this check can read; "
        "a baseline is read without running the check"
    ]


def test_one_computed_baseline_does_not_hide_the_rest_of_a_map() -> None:
    before = 'SIZE_BASELINE = 3\nACCESS = {"a.B": {"baseline": ()}}\n'
    after = 'SIZE_BASELINE = 3\nACCESS = {"a.B": {"baseline": ("grew.py",)}, "c.D": {"baseline": tuple(["new.py"])}}\n'
    assert check.check_baselines(before, after) == [
        "ACCESS['a.B']['baseline'] gains 'grew.py'; "
        "a baseline records room already given up, so it may only lose entries",
        "ACCESS['c.D']['baseline'] arrives and is not written out in full; "
        "a baseline is read without running the check",
    ]


_MAY_BE_REPLACED = (
    "sits in a mapping whose keys are not all written out, so the value written here may be replaced; "
    "a baseline is read without running the check"
)
_HIDES = "ACCESS records a baseline and is not written out in full, so a baseline written in it may be missed"


def test_a_baseline_an_unpacking_can_replace_is_reported() -> None:
    before = 'ACCESS = {"a.B": {"baseline": ("one.py",)}}\n'
    after = 'ACCESS = {"a.B": {"baseline": ("one.py",), **EXTRA}}\n'
    assert check.check_baselines(before, after) == [f"ACCESS['a.B']['baseline'] {_MAY_BE_REPLACED}", _HIDES]


def test_an_unpacking_in_the_map_itself_is_reported() -> None:
    before = 'ACCESS = {"a.B": {"baseline": ("one.py",)}}\n'
    after = 'ACCESS = {"a.B": {"baseline": ("one.py",)}, **EXTRA}\n'
    assert check.check_baselines(before, after) == [f"ACCESS['a.B']['baseline'] {_MAY_BE_REPLACED}", _HIDES]


def test_a_baseline_a_key_not_written_out_can_replace_is_reported() -> None:
    before = 'ACCESS = {"a.B": {"baseline": ()}}\n'
    after = 'MODEL = "a.B"\nACCESS = {"a.B": {"baseline": ()}, MODEL: {"baseline": ("new.py",)}}\n'
    assert check.check_baselines(before, after) == [f"ACCESS['a.B']['baseline'] {_MAY_BE_REPLACED}", _HIDES]


def test_an_entry_that_arrives_unreadable_beside_a_readable_one_is_reported() -> None:
    before = 'ACCESS = {"a.B": {"baseline": ()}}\n'
    after = 'ACCESS = {"a.B": {"baseline": ()}, "c.D": dict(baseline=("new.py",))}\n'
    assert check.check_baselines(before, after) == [_HIDES]


def test_a_key_that_cannot_collide_with_an_entry_leaves_a_map_readable() -> None:
    before = 'ACCESS = {"a.B": {"baseline": ()}}\n'
    after = 'ACCESS = {1: "unrelated", "a.B": {"baseline": ()}}\n'
    assert check.check_baselines(before, after) == []


def test_an_empty_baseline_gone_into_a_map_that_cannot_be_read_is_reported() -> None:
    before = 'ACCESS = {"a.B": {"baseline": ()}}\n'
    after = 'ACCESS = dict({"a.B": {"baseline": ("new.py",)}})\n'
    assert check.check_baselines(before, after) == [
        "ACCESS['a.B']['baseline'] is gone from ACCESS, which is not written out in full, so its removal cannot be read"
    ]


def test_an_empty_baseline_gone_from_a_map_that_can_be_read_is_clean() -> None:
    before = 'ACCESS = {"a.B": {"baseline": ()}, "c.D": {"baseline": ()}}\n'
    after = 'ACCESS = {"c.D": {"baseline": ()}}\n'
    assert check.check_baselines(before, after) == []


def test_a_baseline_gone_with_its_whole_map_is_clean_once_empty() -> None:
    before = 'SIZE_BASELINE = 0\nACCESS = {"a.B": {"baseline": ()}}\n'
    after = "SIZE_BASELINE = 0\n"
    assert check.check_baselines(before, after) == []
