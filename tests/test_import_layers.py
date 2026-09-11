"""Fail if aegis_eval packages import across the layout DAG."""
import ast
from pathlib import Path

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
    prefix = parent[:-level] if level else parent
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
    return names


def _bucket(mod: str) -> str:
    if mod == "aegis_eval.reporter" or mod.startswith("aegis_eval.reporter."):
        return "reporter"
    if mod.startswith("aegis_eval.core"):
        return "core"
    if mod.startswith("aegis_eval.harness"):
        return "harness"
    if mod.startswith("aegis_eval.benchmarks"):
        return "benchmarks"
    if mod.startswith("aegis_eval.workflows.grader_quality"):
        return "workflows.grader_quality"
    if mod.startswith("aegis_eval.workflows.hosted"):
        return "workflows.hosted"
    if mod.startswith("aegis_eval.workflows"):
        return "workflows"
    if mod.startswith("aegis_eval.cli"):
        return "cli"
    if mod == "aegis_eval" or mod.startswith("aegis_eval."):
        return "aegis_eval"
    return mod.split(".", 1)[0]


FORBIDDEN = {
    "core": {
        "harness", "workflows", "workflows.grader_quality", "workflows.hosted",
        "cli", "reporter", "tools", "experiments",
    },
    "reporter": {"harness", "cli", "workflows", "workflows.grader_quality", "workflows.hosted"},
    "benchmarks": {
        "harness", "cli", "workflows", "workflows.grader_quality", "workflows.hosted", "reporter",
    },
    "harness": {"cli", "workflows", "workflows.grader_quality", "workflows.hosted", "reporter"},
    "workflows.grader_quality": {"workflows.hosted", "cli"},
    "workflows.hosted": {"cli"},
}


def test_import_layers_follow_the_layout_dag():
    violations = []
    for path in PKG.rglob("*.py"):
        current = _module_name(path)
        src_bucket = _bucket(current)
        for name in _imported_names(path, current):
            top = name.split(".", 1)[0]
            if top in {"tools", "experiments"}:
                violations.append(f"{current} imports {name}")
                continue
            if not name.startswith("aegis_eval"):
                continue
            dst = _bucket(name)
            if dst in FORBIDDEN.get(src_bucket, ()):
                violations.append(f"{current} imports {name}")
    assert violations == []
