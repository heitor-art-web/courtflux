from __future__ import annotations

import ast
from pathlib import Path

import core


REPO_ROOT = Path(__file__).resolve().parents[1]
P4_SOURCE_PATHS = tuple(sorted((REPO_ROOT / "core").glob("p4_*.py"))) + tuple(
    sorted((REPO_ROOT / "services" / "p4").glob("*.py"))
)
FORBIDDEN_IMPORT_PREFIXES = (
    "core.p3",
    "services.p2",
    "services.p3",
)
FORBIDDEN_BASKETBALL_IDENTIFIERS = {
    "possession_id",
    "game_player_id",
    "ball_handler_game_player_id",
    "ball_control_state",
    "offense_team_id",
    "defense_team_id",
    "tactical_action",
}
REQUIRED_CORE_EXPORTS = {
    "P4_TRAJECTORY_CONTRACT_VERSION",
    "TrajectoryPolicy",
    "TrajectoryPoint",
    "PlayerTrajectory",
    "build_player_trajectories",
    "P4_KINEMATICS_CONTRACT_VERSION",
    "KinematicsPolicy",
    "KinematicsPoint",
    "PlayerKinematics",
    "build_player_kinematics",
    "P4_SPATIAL_RELATIONSHIP_CONTRACT_VERSION",
    "SpatialActorState",
    "PairwiseSpatialRelationship",
    "P4_TEAM_SHAPE_CONTRACT_VERSION",
    "TeamShapeMetrics",
    "P4_MOTION_PRIMITIVE_CONTRACT_VERSION",
    "MotionPrimitivePolicy",
    "MotionPrimitiveObservation",
    "P4_SPATIAL_SNAPSHOT_CONTRACT_VERSION",
    "SpatialSnapshot",
    "materialize_spatial_snapshot",
}


def _imported_modules(tree: ast.AST) -> tuple[str, ...]:
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            modules.append(node.module)
        elif isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
    return tuple(modules)


def _identifiers(tree: ast.AST) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, ast.arg):
            names.add(node.arg)
    return names


def test_p4_source_tree_has_no_p2_or_p3_imports() -> None:
    assert P4_SOURCE_PATHS
    violations: list[str] = []
    for path in P4_SOURCE_PATHS:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for module in _imported_modules(tree):
            if module.startswith(FORBIDDEN_IMPORT_PREFIXES):
                violations.append(f"{path.relative_to(REPO_ROOT)} -> {module}")
    assert not violations, "forbidden P2/P3 imports:\n" + "\n".join(violations)


def test_p4_source_tree_has_no_basketball_only_identifiers() -> None:
    violations: list[str] = []
    for path in P4_SOURCE_PATHS:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        leaked = sorted(_identifiers(tree) & FORBIDDEN_BASKETBALL_IDENTIFIERS)
        if leaked:
            violations.append(f"{path.relative_to(REPO_ROOT)} -> {', '.join(leaked)}")
    assert not violations, "basketball-only identifiers leaked into P4 Spatial:\n" + "\n".join(violations)


def test_core_exports_cover_current_spatial_foundation_contracts() -> None:
    missing = sorted(name for name in REQUIRED_CORE_EXPORTS if not hasattr(core, name))
    assert not missing, f"core package is missing P4 exports: {missing}"
    missing_all = sorted(name for name in REQUIRED_CORE_EXPORTS if name not in core.__all__)
    assert not missing_all, f"core.__all__ is missing P4 exports: {missing_all}"


def test_p4_contract_versions_are_unique() -> None:
    version_names = sorted(
        name
        for name in dir(core)
        if name.startswith("P4_") and name.endswith("_CONTRACT_VERSION")
    )
    values = [getattr(core, name) for name in version_names]
    assert len(values) == len(set(values)), dict(zip(version_names, values))
