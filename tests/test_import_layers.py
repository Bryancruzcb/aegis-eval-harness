"""Fail if aegis_eval packages import across the layout DAG."""
import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PKG = ROOT / "aegis_eval"


def _module_name(path: Path) -> str:
    rel = path.relative_to(ROOT).with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _resolve_from(current: str, module: str | None, level: int) -> str | None:
    if level == 0:
        return module
    parts = current.split(".")
    if current.endswith(".__init__") or (ROOT.joinpath(*parts).is_dir()):
        parent = parts
    else:
        parent = parts[:-1]
    if level > len(parent):
        return None
    ascents = level - 1
    prefix = parent[:-ascents] if ascents else parent
    if module:
        prefix = prefix + module.split(".")
    return ".".join(prefix) if prefix else None


def _imported_names(path: Path, current: str) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            resolved = _resolve_from(current, node.module, node.level)
            if resolved:
                names.append(resolved)
                names.extend(
                    f"{resolved}.{alias.name}"
                    for alias in node.names
                    if alias.name != "*"
                )
    return names


def _bucket(mod: str) -> str:
    if mod == "aegis_eval":
        return "aegis_eval"
    if mod == "aegis_eval.reporter" or mod.startswith("aegis_eval.reporter."):
        return "reporter"
    if mod == "aegis_eval.core" or mod.startswith("aegis_eval.core."):
        return "core"
    if mod == "aegis_eval.harness" or mod.startswith("aegis_eval.harness."):
        return "harness"
    if mod == "aegis_eval.benchmarks" or mod.startswith("aegis_eval.benchmarks."):
        return "benchmarks"
    if (mod == "aegis_eval.workflows.grader_quality"
            or mod.startswith("aegis_eval.workflows.grader_quality.")):
        return "workflows.grader_quality"
    if mod == "aegis_eval.workflows.hosted" or mod.startswith("aegis_eval.workflows.hosted."):
        return "workflows.hosted"
    if mod == "aegis_eval.workflows":
        return "workflows"
    if mod == "aegis_eval.cli" or mod.startswith("aegis_eval.cli."):
        return "cli"
    if mod.startswith("aegis_eval."):
        return "unknown_internal"
    return mod.split(".", 1)[0]


ALLOWED_INTERNAL = {
    "aegis_eval": {"aegis_eval"},
    "core": {"core"},
    "benchmarks": {"benchmarks", "core"},
    "harness": {"harness", "core", "benchmarks"},
    "reporter": {"reporter", "core"},
    "workflows": {"workflows"},
    "workflows.grader_quality": {
        "workflows", "workflows.grader_quality", "core", "harness", "benchmarks",
    },
    "workflows.hosted": {
        "workflows", "workflows.hosted", "workflows.grader_quality",
        "core", "harness", "benchmarks",
    },
    "cli": {
        "cli", "core", "harness", "reporter", "benchmarks", "workflows",
        "workflows.grader_quality", "workflows.hosted",
    },
}


def _violates_layer(source: str, imported: str) -> bool:
    top = imported.split(".", 1)[0]
    if top in {"tools", "experiments"}:
        return True
    if imported != "aegis_eval" and not imported.startswith("aegis_eval."):
        return False
    destination = _bucket(imported)
    if destination == "aegis_eval":
        return False
    return destination not in ALLOWED_INTERNAL.get(source, set())


def test_parent_relative_import_resolves_to_package_layer(tmp_path):
    module = tmp_path / "sample.py"
    module.write_text("from ..harness import runner\n", encoding="utf-8")

    assert "aegis_eval.harness" in _imported_names(module, "aegis_eval.core.sample")


def test_from_package_import_tracks_the_imported_submodule(tmp_path):
    module = tmp_path / "sample.py"
    module.write_text("from aegis_eval import reporter\n", encoding="utf-8")

    assert "aegis_eval.reporter" in _imported_names(module, "aegis_eval.core.sample")


def test_relative_from_import_without_module_tracks_the_imported_submodule(tmp_path):
    module = tmp_path / "sample.py"
    module.write_text("from .. import harness\n", encoding="utf-8")

    assert "aegis_eval.harness" in _imported_names(module, "aegis_eval.core.sample")


@pytest.mark.parametrize(
    ("source", "imported"),
    [
        ("core", "aegis_eval.benchmarks"),
        ("reporter", "aegis_eval.benchmarks"),
        ("workflows.grader_quality", "aegis_eval.reporter"),
        ("workflows.hosted", "aegis_eval.reporter"),
        ("workflows", "aegis_eval.harness"),
        ("core", "aegis_eval.unexpected"),
        ("core", "aegis_eval.coreevil"),
        ("workflows.grader_quality", "aegis_eval.workflows.unexpected"),
        ("workflows.hosted", "aegis_eval.workflows.unexpected"),
    ],
)
def test_undocumented_internal_edges_are_rejected(source, imported):
    assert _violates_layer(source, imported)


@pytest.mark.parametrize(
    ("source", "imported"),
    [
        ("core", "aegis_eval.core.config"),
        ("benchmarks", "aegis_eval.core.config"),
        ("harness", "aegis_eval.benchmarks.jbb"),
        ("reporter", "aegis_eval.core.config"),
        ("workflows.grader_quality", "aegis_eval.harness.refusal_grader"),
        ("workflows.hosted", "aegis_eval.workflows.grader_quality.compare_graders"),
        ("cli", "aegis_eval.workflows.hosted.hosted_comparison"),
    ],
)
def test_documented_internal_edges_are_allowed(source, imported):
    assert not _violates_layer(source, imported)


def test_import_layers_follow_the_layout_dag():
    violations = []
    for path in PKG.rglob("*.py"):
        current = _module_name(path)
        src_bucket = _bucket(current)
        for name in _imported_names(path, current):
            if _violates_layer(src_bucket, name):
                violations.append(f"{current} imports {name}")
    assert violations == []
