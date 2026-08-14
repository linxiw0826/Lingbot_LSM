import importlib.util
import sys
import types
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
