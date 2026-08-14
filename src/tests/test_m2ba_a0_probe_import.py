import importlib.util
import sys
import types
import copy
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
PROBE = REPO / "tools/preflight/m2ba_a0/probe_a0.py"


def _load_a0_probe():
    spec = importlib.util.spec_from_file_location("_test_m2ba_a0_probe", PROBE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_fixture_helper_resolves_from_repo_when_tools_is_shadowed(monkeypatch):
    conflict = types.ModuleType("tools")
    conflict.__file__ = "/external/site-packages/tools/__init__.py"
    monkeypatch.setitem(sys.modules, "tools", conflict)
    module = _load_a0_probe()
    helper = module.load_fixture_probe_module(REPO)
    expected = (
        REPO / "tools/preflight/m2ba_a01_fixture_freeze/probe_wan_runtime.py"
    ).resolve()
    assert Path(helper.__file__).resolve() == expected
    assert callable(helper.build_conditioning_only)
    assert callable(helper.direct_forward_conditioning)
    assert callable(helper.state_inventory)


def _valid_train_fixture():
    plans = []
    for segment_start, segment_end, support in ((0, 280, False), (280, 405, True)):
        cursor = segment_start
        while cursor < segment_end:
            owned_end = min(segment_end, cursor + 65)
            source_start = min(max(0, cursor - 8), 324)
            plans.append({
                "window_index": len(plans),
                "source_frame_index": list(range(source_start, source_start + 81)),
                "owned_half_open": [cursor, owned_end],
                "support": support,
            })
            cursor = owned_end
    rows = []
    for full_frame in range(280, 405):
        plan = next(
            item for item in plans
            if item["owned_half_open"][0] <= full_frame < item["owned_half_open"][1]
        )
        local = plan["source_frame_index"].index(full_frame)
        latent_t = (local + 3) // 4
        rows.append({
            "full_frame": full_frame, "window_id": plan["window_index"],
            "local_frame": local, "latent_t": latent_t, "patch_t": latent_t,
            "token_start": latent_t * 1508, "token_end": (latent_t + 1) * 1508,
            "token_count": 1508,
        })
    grouped = {}
    for row in rows:
        key = (row["window_id"], row["latent_t"], row["token_start"], row["token_end"])
        grouped.setdefault(key, []).append(row["full_frame"])
    groups = [
        {
            "window_id": key[0], "latent_t": key[1], "token_start": key[2],
            "token_end": key[3], "full_frames": frames,
            "full_frame_half_open": [min(frames), max(frames) + 1],
        }
        for key, frames in sorted(grouped.items())
    ]
    return {
        "role": "TRAIN",
        "case_id": "Ep000027_p0007_77s_86s_two_windows_revisit",
        "event_id": "side_alley_return",
        "total_frames": 405,
        "support_full_half_open": [280, 405],
        "target_full_frame": 342,
        "target_window_id": 5,
        "target_window_local_indices": [70],
        "memory_selected_full_frames": [96, 128, 176],
        "planner_windows": plans,
        "frame_to_token_mapping": {
            "status": "PASS", "complete": True,
            "target": next(row for row in rows if row["full_frame"] == 342),
            "per_frame": rows,
            "deduplicated_many_to_one_groups": groups,
            "group_boundaries": [group["full_frame_half_open"] for group in groups],
        },
    }


def test_train_fixture_schema_accepts_only_authoritative_memory_field():
    module = _load_a0_probe()
    assert module.validate_train_fixture_schema(_valid_train_fixture()) == [96, 128, 176]


def test_train_fixture_schema_rejects_missing_field_and_wrong_alias():
    module = _load_a0_probe()
    missing = _valid_train_fixture()
    missing.pop("memory_selected_full_frames")
    missing["memory_selected_set"] = [96, 128, 176]
    try:
        module.validate_train_fixture_schema(missing)
    except RuntimeError as exc:
        assert "forbidden summary alias" in str(exc)
    else:
        raise AssertionError("summary alias was silently accepted")

    drift = _valid_train_fixture()
    drift["memory_selected_full_frames"] = [96, 128, 175]
    try:
        module.validate_train_fixture_schema(drift)
    except RuntimeError as exc:
        assert "identity/memory frames drift" in str(exc)
    else:
        raise AssertionError("wrong frozen memory tuple was silently accepted")


def test_train_fixture_schema_rejects_alias_even_when_authority_exists():
    module = _load_a0_probe()
    fixture = copy.deepcopy(_valid_train_fixture())
    fixture["memory_selected_set"] = fixture["memory_selected_full_frames"]
    try:
        module.validate_train_fixture_schema(fixture)
    except RuntimeError as exc:
        assert "forbidden summary alias" in str(exc)
    else:
        raise AssertionError("ambiguous alias was silently ignored")


def test_train_fixture_schema_rejects_target_slice_drift():
    module = _load_a0_probe()
    fixture = copy.deepcopy(_valid_train_fixture())
    fixture["frame_to_token_mapping"]["target"]["token_start"] += 1508
    try:
        module.validate_train_fixture_schema(fixture)
    except RuntimeError as exc:
        assert "canonical frame/token mapping drift" in str(exc)
    else:
        raise AssertionError("target slice drift was silently accepted")


def test_train_fixture_schema_rejects_deleted_or_duplicate_group():
    module = _load_a0_probe()
    deleted = copy.deepcopy(_valid_train_fixture())
    deleted["frame_to_token_mapping"]["deduplicated_many_to_one_groups"].pop()
    try:
        module.validate_train_fixture_schema(deleted)
    except RuntimeError as exc:
        assert "canonical frame/token mapping drift" in str(exc)
    else:
        raise AssertionError("deleted group was silently accepted")

    duplicate = copy.deepcopy(_valid_train_fixture())
    duplicate["frame_to_token_mapping"]["deduplicated_many_to_one_groups"].append(
        copy.deepcopy(duplicate["frame_to_token_mapping"]["deduplicated_many_to_one_groups"][0])
    )
    try:
        module.validate_train_fixture_schema(duplicate)
    except RuntimeError as exc:
        assert "canonical frame/token mapping drift" in str(exc)
    else:
        raise AssertionError("duplicate group was silently accepted")


def test_train_fixture_schema_rejects_support_frame_coverage_drift():
    module = _load_a0_probe()
    fixture = copy.deepcopy(_valid_train_fixture())
    fixture["frame_to_token_mapping"]["per_frame"].pop(17)
    try:
        module.validate_train_fixture_schema(fixture)
    except RuntimeError as exc:
        assert "canonical frame/token mapping drift" in str(exc)
    else:
        raise AssertionError("support frame coverage drift was silently accepted")


def test_frozen_fixture_set_requires_exactly_one_train_and_classifies_schema_error():
    module = _load_a0_probe()
    for fixtures, expected_count in (([], 0), ([_valid_train_fixture(), _valid_train_fixture()], 2)):
        try:
            module.select_unique_train_fixture({"fixtures": fixtures})
        except RuntimeError as exc:
            assert f"got {expected_count}" in str(exc)
            assert module.classify_probe_error(exc) == "BLOCKED_FIXTURE_SCHEMA"
        else:
            raise AssertionError("non-unique TRAIN fixture set was silently accepted")


def test_frozen_fixtures_must_be_list_and_unique_train_returns_authority():
    module = _load_a0_probe()
    try:
        module.select_unique_train_fixture({"fixtures": {"role": "TRAIN"}})
    except RuntimeError as exc:
        assert "fixtures must be a list" in str(exc)
        assert module.classify_probe_error(exc) == "BLOCKED_FIXTURE_SCHEMA"
    else:
        raise AssertionError("non-list fixtures was silently accepted")
    fixture = _valid_train_fixture()
    selected, memory = module.select_unique_train_fixture({"fixtures": [fixture]})
    assert selected is fixture
    assert memory == [96, 128, 176]
