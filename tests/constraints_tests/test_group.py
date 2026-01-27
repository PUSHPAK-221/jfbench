import ast
from pathlib import Path

from jfbench.constraints._competitives import COMPETITIVE_CONSTRAINTS
from jfbench.constraints._group import ConstraintGroupMixin


ROOT = Path(__file__).resolve().parents[2]
CONSTRAINTS_DIR = ROOT / "src" / "jfbench" / "constraints"


def _constraint_nodes() -> list[tuple[Path, ast.ClassDef]]:
    nodes: list[tuple[Path, ast.ClassDef]] = []
    for path in CONSTRAINTS_DIR.rglob("*.py"):
        if path.name == "_group.py":
            continue
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in tree.body:
            if not isinstance(node, ast.ClassDef):
                continue
            if node.name.endswith("Constraint") or node.name.endswith("Constraints"):
                nodes.append((path, node))
    return nodes


def _constraint_names() -> set[str]:
    return {node.name for _, node in _constraint_nodes()}


def test_group_mixin_resolves_simple_directory_name() -> None:
    dummy_cls = type("DummyConstraint", (ConstraintGroupMixin,), {})
    dummy_cls.__module__ = "jfbench.constraints.hoge.sample"
    instance = dummy_cls()
    assert instance.group == "Hoge"


def test_group_mixin_resolves_snake_case_directory_name() -> None:
    dummy_cls = type("DummyConstraint", (ConstraintGroupMixin,), {})
    dummy_cls.__module__ = "jfbench.constraints.meta_output.sample"
    instance = dummy_cls()
    assert instance.group == "MetaOutput"


def test_all_constraints_extend_group_mixin() -> None:
    missing: list[str] = []
    for path, node in _constraint_nodes():
        base_names = {ast.unparse(base) for base in node.bases}
        if "ConstraintGroupMixin" not in base_names:
            missing.append(f"{path}:{node.name}")
    assert not missing, f"Classes missing ConstraintGroupMixin: {missing}"


def test_competitive_mapping_covers_all_constraints() -> None:
    names = _constraint_names()
    assert set(COMPETITIVE_CONSTRAINTS) == names


def test_competitive_entries_reference_valid_constraints() -> None:
    names = _constraint_names()
    invalid: list[str] = []
    for name, competitors in COMPETITIVE_CONSTRAINTS.items():
        for other in competitors:
            if other not in names:
                invalid.append(f"{name}->{other}")
    assert not invalid, f"Unknown competitive references detected: {invalid}"


def test_competitive_relationships_are_symmetric() -> None:
    asymmetric: list[str] = []
    for name, competitors in COMPETITIVE_CONSTRAINTS.items():
        for other in competitors:
            if name not in COMPETITIVE_CONSTRAINTS.get(other, ()):
                asymmetric.append(f"{name}<->{other}")
    assert not asymmetric, f"Asymmetric competitive pairs: {asymmetric}"


def test_selected_constraints_have_expected_competitives() -> None:
    mapping = COMPETITIVE_CONSTRAINTS
    assert "JsonFormatConstraint" in mapping["CsvFormatConstraint"]
    assert "CsvFormatConstraint" in mapping["JsonFormatConstraint"]
    assert "NoExplanationConstraint" in mapping["ReasonContentConstraint"]
    assert "ReasonContentConstraint" in mapping["NoExplanationConstraint"]
    assert "EnglishStyleConstraint" in mapping["JapaneseStyleConstraint"]
    assert "JapaneseStyleConstraint" in mapping["EnglishStyleConstraint"]
    assert "NoReasonContentConstraint" in mapping["ReasonContentConstraint"]
    assert "ReasonContentConstraint" in mapping["NoReasonContentConstraint"]
    assert "NoExplanationConstraint" in mapping["ExplanationConstraint"]
    assert "ExplanationConstraint" in mapping["NoExplanationConstraint"]
    assert "NoGreetingOutputConstraint" in mapping["GreetingOutputConstraint"]
    assert "GreetingOutputConstraint" in mapping["NoGreetingOutputConstraint"]
    assert "NoSelfReferenceConstraint" in mapping["SelfReferenceConstraint"]
    assert "SelfReferenceConstraint" in mapping["NoSelfReferenceConstraint"]
    assert "NoSelfAttestationConstraint" in mapping["SelfAttestationConstraint"]
    assert "SelfAttestationConstraint" in mapping["NoSelfAttestationConstraint"]
    assert "NoEnglishStyleConstraint" in mapping["EnglishStyleConstraint"]
    assert "NoJapaneseStyleConstraint" in mapping["JapaneseStyleConstraint"]
    assert (
        "WithCodeFenceJavascriptFormatConstraint"
        in mapping["NoCodeFenceJavascriptFormatConstraint"]
    )
    assert (
        "NoCodeFenceJavascriptFormatConstraint"
        in mapping["WithCodeFenceJavascriptFormatConstraint"]
    )
    assert "WithCodeFencePythonFormatConstraint" in mapping["NoCodeFencePythonFormatConstraint"]
    assert "NoCodeFencePythonFormatConstraint" in mapping["WithCodeFencePythonFormatConstraint"]
    assert "JsonFormatConstraint" in mapping["DiffFormatConstraint"]
    assert "DiffFormatConstraint" in mapping["JsonFormatConstraint"]
