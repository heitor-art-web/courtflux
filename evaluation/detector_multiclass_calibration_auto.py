from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml
from ultralytics import YOLO


SKIP_DIRS = {
    ".git",
    ".venv",
    "venv",
    "node_modules",
    "__pycache__",
    ".next",
    ".cache",
    "Library",
}
YAML_NAMES = {"data.yaml", "dataset.yaml", "data.yml", "dataset.yml"}
IMAGE_SPLIT_NAMES = ("test", "val", "train")


def _arg_value(argv: list[str], flag: str, default: str | None = None) -> str | None:
    try:
        idx = argv.index(flag)
    except ValueError:
        return default
    if idx + 1 >= len(argv):
        raise SystemExit(f"{flag} requires a value")
    return argv[idx + 1]


def _has_arg(argv: list[str], flag: str) -> bool:
    return flag in argv


def _normalise_names(names: Any) -> dict[int, str]:
    if isinstance(names, dict):
        return {int(k): str(v).strip().lower() for k, v in names.items()}
    if isinstance(names, (list, tuple)):
        return {idx: str(v).strip().lower() for idx, v in enumerate(names)}
    return {}


def _model_names(path: Path) -> dict[int, str]:
    model = YOLO(str(path))
    return _normalise_names(getattr(model, "names", {}) or {})


def _search_roots(project_root: Path, argv: list[str]) -> list[Path]:
    explicit = _arg_value(argv, "--search-root")
    roots: list[Path] = []
    if explicit:
        roots.extend(Path(item).expanduser().resolve() for item in explicit.split(os.pathsep) if item)

    env_roots = os.environ.get("COURTFLUX_DATA_ROOTS", "")
    if env_roots:
        roots.extend(Path(item).expanduser().resolve() for item in env_roots.split(os.pathsep) if item)

    roots.extend(
        [
            project_root,
            project_root.parent,
            Path.home() / "Documents",
            Path.home() / "Downloads",
            Path.home() / "Desktop",
        ]
    )

    out: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        try:
            root = root.resolve()
        except OSError:
            continue
        if not root.exists() or root in seen:
            continue
        seen.add(root)
        out.append(root)
    return out


def _walk(root: Path):
    for current, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.startswith(".")]
        yield Path(current), files


def _resolve_root(data_yaml: Path, cfg: dict[str, Any]) -> Path:
    raw = cfg.get("path")
    if not raw:
        return data_yaml.parent.resolve()
    path = Path(str(raw)).expanduser()
    if path.is_absolute() and path.exists():
        return path.resolve()
    if not path.is_absolute():
        candidate = (data_yaml.parent / path).resolve()
        if candidate.exists():
            return candidate
    # Common case: training wrote an absolute temporary path. Fall back to the
    # YAML's own directory only if the split paths are actually present there.
    return data_yaml.parent.resolve()


def _split_exists(data_yaml: Path, cfg: dict[str, Any], split: str) -> bool:
    if split not in cfg:
        return False
    root = _resolve_root(data_yaml, cfg)
    values = cfg[split] if isinstance(cfg[split], list) else [cfg[split]]
    for value in values:
        p = Path(str(value)).expanduser()
        candidates = [p] if p.is_absolute() else [root / p, data_yaml.parent / p]
        if any(candidate.exists() for candidate in candidates):
            return True
    return False


def _yaml_candidate_score(
    path: Path,
    *,
    split: str,
    expected_names: dict[int, str],
) -> tuple[int, dict[str, Any] | None, str]:
    try:
        cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        return -1, None, f"invalid yaml: {exc}"
    if not isinstance(cfg, dict):
        return -1, None, "not a mapping"

    names = _normalise_names(cfg.get("names", {}))
    score = 0
    if names:
        score += 3
        if names == expected_names:
            score += 8
        elif set(names.values()) == set(expected_names.values()):
            score += 5
        elif set(names.values()) & set(expected_names.values()):
            score += 2
    if split in cfg:
        score += 3
    if _split_exists(path, cfg, split):
        score += 8
    if path.name.lower() in YAML_NAMES:
        score += 1
    return score, cfg, "ok"


def _find_yaml_candidates(
    roots: list[Path],
    *,
    split: str,
    expected_names: dict[int, str],
) -> list[tuple[int, Path, dict[str, Any]]]:
    found: list[tuple[int, Path, dict[str, Any]]] = []
    seen: set[Path] = set()
    for root in roots:
        for current, files in _walk(root):
            for filename in files:
                lower = filename.lower()
                if lower not in YAML_NAMES and not (lower.endswith((".yaml", ".yml")) and "dataset" in lower):
                    continue
                path = (current / filename).resolve()
                if path in seen:
                    continue
                seen.add(path)
                score, cfg, _reason = _yaml_candidate_score(
                    path,
                    split=split,
                    expected_names=expected_names,
                )
                if score >= 0 and cfg is not None:
                    found.append((score, path, cfg))
    found.sort(key=lambda item: (item[0], str(item[1])), reverse=True)
    return found


def _find_split_roots(roots: list[Path], split: str) -> list[Path]:
    matches: list[Path] = []
    seen: set[Path] = set()
    target = Path("images") / split
    for root in roots:
        for current, dirs, _files in _walk(root):
            current_path = Path(current)
            if current_path.name != "images":
                continue
            if split not in dirs:
                continue
            dataset_root = current_path.parent.resolve()
            if dataset_root in seen:
                continue
            if (dataset_root / target).is_dir() and (dataset_root / "labels" / split).is_dir():
                seen.add(dataset_root)
                matches.append(dataset_root)
    return matches


def _write_synthetic_yaml(
    project_root: Path,
    dataset_root: Path,
    names: dict[int, str],
    split: str,
) -> Path:
    out_dir = project_root / "evaluation" / ".generated_dataset_configs"
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_name = dataset_root.name.replace(" ", "_") or "dataset"
    out = out_dir / f"{safe_name}_{split}.yaml"
    cfg: dict[str, Any] = {
        "path": str(dataset_root),
        "names": names,
    }
    for name in IMAGE_SPLIT_NAMES:
        image_dir = dataset_root / "images" / name
        label_dir = dataset_root / "labels" / name
        if image_dir.is_dir() and label_dir.is_dir():
            cfg[name] = f"images/{name}"
    if split not in cfg:
        raise FileNotFoundError(f"Dataset root has no paired images/{split} + labels/{split}: {dataset_root}")
    out.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return out.resolve()


def _remove_wrapper_args(argv: list[str]) -> list[str]:
    out: list[str] = []
    i = 0
    while i < len(argv):
        if argv[i] == "--search-root":
            i += 2
            continue
        out.append(argv[i])
        i += 1
    return out


def main() -> None:
    argv = sys.argv[1:]
    project_root = Path(__file__).resolve().parents[1]
    target_script = project_root / "evaluation" / "detector_multiclass_calibration.py"

    if _has_arg(argv, "--data"):
        clean = _remove_wrapper_args(argv)
        raise SystemExit(subprocess.call([sys.executable, str(target_script), *clean]))

    split = _arg_value(argv, "--split", "test") or "test"
    model_v100 = Path(_arg_value(argv, "--model-v100", "weights/model_v1.0.0.pt") or "weights/model_v1.0.0.pt")
    model_v110 = Path(_arg_value(argv, "--model-v110", "weights/model_v1.1.0.pt") or "weights/model_v1.1.0.pt")
    if not model_v100.is_absolute():
        model_v100 = (project_root / model_v100).resolve()
    if not model_v110.is_absolute():
        model_v110 = (project_root / model_v110).resolve()

    names_v100 = _model_names(model_v100)
    names_v110 = _model_names(model_v110)
    if names_v100 != names_v110:
        print("Model class maps differ; refusing to synthesize a shared dataset mapping.")
        print("v1.0.0:", json.dumps(names_v100, indent=2))
        print("v1.1.0:", json.dumps(names_v110, indent=2))
        print("Pass the exact common dataset explicitly with --data /path/to/data.yaml")
        raise SystemExit(2)

    roots = _search_roots(project_root, argv)
    print("Searching for supervised dataset under:")
    for root in roots:
        print(f"  - {root}")

    candidates = _find_yaml_candidates(roots, split=split, expected_names=names_v110)
    usable = [item for item in candidates if item[0] >= 14 and _split_exists(item[1], item[2], split)]

    data_yaml: Path | None = None
    if usable:
        best_score = usable[0][0]
        best = [item for item in usable if item[0] == best_score]
        if len(best) == 1:
            data_yaml = best[0][1]
            print(f"Resolved dataset YAML: {data_yaml} (score={best_score})")
        else:
            print("Multiple equally plausible dataset YAMLs were found:")
            for score, path, _cfg in best[:20]:
                print(f"  score={score}: {path}")
            print("Re-run with --data <exact yaml> to avoid benchmarking the wrong split.")
            raise SystemExit(2)

    if data_yaml is None:
        split_roots = _find_split_roots(roots, split)
        if len(split_roots) == 1:
            data_yaml = _write_synthetic_yaml(project_root, split_roots[0], names_v110, split)
            print(f"No usable YAML found; generated: {data_yaml}")
            print(f"Dataset root: {split_roots[0]}")
        elif len(split_roots) > 1:
            print(f"Found multiple datasets with images/{split} + labels/{split}:")
            for path in split_roots[:30]:
                print(f"  - {path}")
            print("Pass the intended dataset explicitly with --data /path/to/data.yaml")
            raise SystemExit(2)
        else:
            print("No local supervised dataset could be found.")
            print("The checkpoint only contains a stale temporary data path, so GT cannot be reconstructed from the .pt file alone.")
            print("Either restore/download the exact common test dataset, or provide its YAML with --data.")
            print("You can widen discovery with:")
            print("  --search-root /some/root")
            print("or set COURTFLUX_DATA_ROOTS to colon-separated directories.")
            raise SystemExit(2)

    clean = _remove_wrapper_args(argv)
    clean.extend(["--data", str(data_yaml)])
    print("\nLaunching multiclass calibration...\n")
    raise SystemExit(subprocess.call([sys.executable, str(target_script), *clean]))


if __name__ == "__main__":
    main()
