from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def resolve_project_path(path: str | Path, base_dir: str | Path | None = None) -> Path:
    """Resolve a project-relative artifact path without forcing it to exist."""
    resolved = Path(path)
    if not resolved.is_absolute() and base_dir is not None:
        resolved = Path(base_dir) / resolved
    return resolved


def resolve_sb3_model_path(path: str | Path, base_dir: str | Path | None = None) -> Path:
    """Resolve an SB3 model path across both `.zip` and legacy no-extension saves.

    Legacy checkpoints in this project were saved without an explicit `.zip`
    suffix because the slug contains decimal points, so `Path.suffix` was not
    empty and SB3 did not auto-append `.zip` on write.
    """
    requested = resolve_project_path(path, base_dir=base_dir)
    candidates = [requested]

    requested_str = str(requested)
    if requested_str.lower().endswith(".zip"):
        candidates.append(Path(requested_str[:-4]))
    else:
        candidates.append(Path(f"{requested}.zip"))

    for candidate in candidates:
        if candidate.exists():
            return candidate

    tried = "\n".join(f"  - {candidate}" for candidate in candidates)
    raise FileNotFoundError(f"Stable-Baselines3 model artifact not found. Tried:\n{tried}")


def load_run_info(models_dir: str | Path = "models") -> tuple[Path, dict[str, Any]]:
    models_path = Path(models_dir)
    info_path = models_path / "run_info.json"
    if not info_path.exists():
        raise FileNotFoundError(
            f"{info_path} not found.\nRun train_ppo.py first, then retry."
        )
    return info_path, json.loads(info_path.read_text())


class HoldoutContaminationError(RuntimeError):
    """The model has already seen (trained on / been selected on) the holdout."""


def assert_holdout_is_unseen(run_info: dict[str, Any], test_start, allow_unverified: bool = False) -> None:
    """Refuse to "reveal" a holdout that overlaps data the model has already seen.

    ``run_info`` records the train/val windows used by train_ppo.train().  A
    sliding walk-forward promotes its LAST fold, whose train/val windows run up
    to the end of the dataset — i.e. straight through the single-split test
    segment — so its "holdout" result would be in-sample.  Legacy run_info files
    without the window metadata cannot be verified and are refused unless
    ``allow_unverified`` is set.
    """
    import pandas as pd

    seen_ends = [run_info.get(k) for k in ("train_end", "val_end")]
    if any(v is None for v in seen_ends):
        msg = (
            "run_info.json has no train_end/val_end metadata (trained before the "
            "holdout guard existed), so it cannot be verified that the test split "
            "is unseen."
        )
        if run_info.get("validation_scheme") == "sliding" or "promoted_from_fold" in run_info:
            msg += (" This model was promoted from a walk-forward fold; a sliding "
                    "fold's train/val windows normally overlap the test split.")
        if not allow_unverified:
            raise HoldoutContaminationError(
                msg + " Retrain, or pass allow_unverified=True (--allow-unverified) "
                "if you are sure the model never saw the test period."
            )
        print(f"WARNING: {msg} Proceeding because allow_unverified=True.")
        return

    last_seen = max(pd.Timestamp(v) for v in seen_ends)
    test_start = pd.Timestamp(test_start)
    if last_seen >= test_start:
        raise HoldoutContaminationError(
            f"Holdout is contaminated: the model saw data up to {last_seen} "
            f"(train/val), but the test split starts at {test_start}. Its test "
            "metrics would be in-sample. For a sliding walk-forward run, use the "
            "per-fold TEST results (models/sliding_walk_forward_summary.csv) and "
            "the stitched curve (models/sliding_oos_equity.csv) instead."
        )
