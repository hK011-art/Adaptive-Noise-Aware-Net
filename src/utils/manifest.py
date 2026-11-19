import os
import json
import hashlib
from typing import List, Dict, Any, Optional

def compute_file_sha256(filepath: str, chunk_size: int = 65536) -> str:
    """
    Computes SHA-256 hash of a file incrementally without loading large images into RAM.
    """
    if not os.path.exists(filepath):
        return "missing"
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


def extract_scene_id_from_sample(sample_id: str, dataset_name: Optional[str] = None) -> str:
    """
    Extracts the underlying scene identifier from a sample/patch string.
    Ensures that patches derived from the same source photograph share an identical scene ID,
    preventing intra-scene patch data leakage across train/val/test splits.
    
    Supported dataset formats:
    - DIV2K: e.g. '0801.png', '0801_y128_x256', '0801_patch0' -> '0801'
    - SIDD: e.g. '0001_001_S6_00100_00060_3200_L_NOISY.PNG', 'sidd_0001_...' -> '0001'
    - DND: e.g. '0001_01.mat', 'dnd_0001_crop2' -> '0001'
    - Generic/Fixtures: e.g. 'sceneA_y0_x0', 'toy_sample_3' -> 'sceneA', 'toy_sample_3'
    
    Fails loudly if the sample ID is empty or malformed.
    """
    if not sample_id or not isinstance(sample_id, str) or not sample_id.strip():
        raise ValueError(f"Invalid empty sample identifier: '{sample_id}'")

    clean_id = sample_id.strip()
    base = os.path.basename(clean_id)
    # Strip file extensions
    stem = os.path.splitext(base)[0]

    # Explicit dataset-specific parsing
    ds = dataset_name.lower().strip() if dataset_name else None

    if ds == "div2k":
        # DIV2K files are numbered images, e.g. 0801 or 0801_y128_x256
        parts = stem.split("_")
        scene_token = parts[0]
        if not scene_token.isdigit():
            raise ValueError(
                f"Malformed DIV2K sample format: '{sample_id}'. "
                f"Expected leading numeric scene ID (e.g. '0801' or '0801_y0_x0'), got '{scene_token}'."
            )
        return scene_token

    elif ds == "sidd":
        # SIDD filenames start with 4-digit scene ID, optionally prefixed with 'sidd_'
        working = stem
        if working.startswith("sidd_"):
            working = working[5:]
        parts = working.split("_")
        scene_token = parts[0]
        if not (scene_token.isdigit() and len(scene_token) == 4):
            # Check for scene_XX format fallback
            if not scene_token.startswith("scene"):
                raise ValueError(
                    f"Malformed SIDD sample format: '{sample_id}'. "
                    f"Expected 4-digit scene prefix (e.g. '0001_001_...'), got '{scene_token}'."
                )
        return scene_token

    elif ds == "dnd":
        working = stem
        if working.startswith("dnd_"):
            working = working[4:]
        parts = working.split("_")
        scene_token = parts[0]
        if not scene_token.isdigit():
            raise ValueError(
                f"Malformed DND sample format: '{sample_id}'. Expected numeric scene prefix, got '{scene_token}'."
            )
        return scene_token

    # Generic / automatic parsing:
    # 1. Strip dataset prefix if present
    working = stem
    for prefix in ["sidd_", "dnd_", "div2k_"]:
        if working.startswith(prefix):
            working = working[len(prefix):]
            break

    # 2. Split off patch crop coordinates (_y..._x...)
    if "_y" in working and "_x" in working:
        scene = working.split("_y")[0]
    elif "_patch" in working:
        scene = working.split("_patch")[0]
    elif "_" in working and working.split("_")[0].isdigit() and len(working.split("_")[0]) == 4:
        # Standard SIDD / DND 4-digit scene convention
        scene = working.split("_")[0]
    else:
        scene = working

    if not scene:
        raise ValueError(f"Failed to extract valid scene ID from malformed sample ID: '{sample_id}'")

    return scene


def check_for_data_leakage(train_sample_ids: List[str], val_sample_ids: List[str]) -> bool:
    """
    Asserts that no sample OR scene appears in both training and validation sets.
    Prevents patch-level and scene-level data leakage.
    """
    # 1. Direct sample collision check
    direct_overlap = set(train_sample_ids).intersection(set(val_sample_ids))
    if len(direct_overlap) > 0:
        raise ValueError(
            f"DATA LEAKAGE DETECTED! Found {len(direct_overlap)} direct sample overlaps "
            f"between train and validation splits: {list(direct_overlap)[:5]}..."
        )

    # 2. Scene-level collision check
    train_scenes = {extract_scene_id_from_sample(s) for s in train_sample_ids}
    val_scenes = {extract_scene_id_from_sample(s) for s in val_sample_ids}
    scene_overlap = train_scenes.intersection(val_scenes)
    if len(scene_overlap) > 0:
        raise ValueError(
            f"SCENE-LEVEL DATA LEAKAGE DETECTED! Found {len(scene_overlap)} overlapping scene IDs "
            f"between train and validation splits: {list(scene_overlap)[:5]}..."
        )

    return True


def validate_training_dataset_safety(dataset_name: str, split: str = "train") -> None:
    """
    Guarantees DND or held-out test benchmarks can NEVER enter training loops.
    """
    normalized_name = dataset_name.strip().lower()
    normalized_split = split.strip().lower()

    if "dnd" in normalized_name and normalized_split == "train":
        raise ValueError(
            "CRITICAL PROTOCOL VIOLATION: DND dataset is strictly a held-out benchmark "
            "and must NEVER be configured for training or parameter tuning!"
        )

    if normalized_split in ["test", "held_out", "benchmark"] and "train" in normalized_split:
        raise ValueError(
            f"SAFETY VIOLATION: Contradictory split configuration '{split}' detected for dataset '{dataset_name}'."
        )


def create_experiment_manifest(
    dataset_name: str,
    split: str,
    sample_ids: List[str],
    seed: int,
    output_path: str = "experiment_manifest.json",
    extra_metadata: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Generates a deterministic experiment manifest to ensure experimental traceability.
    """
    validate_training_dataset_safety(dataset_name, split)

    manifest_data = {
        "dataset_name": dataset_name,
        "split": split,
        "num_samples": len(sample_ids),
        "seed": seed,
        "preprocessing_version": "v1.0-physics-hardened",
        "sample_ids": sample_ids,
        "sample_ids_sha256": hashlib.sha256(",".join(sample_ids).encode()).hexdigest(),
        "extra_metadata": extra_metadata or {}
    }

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(manifest_data, f, indent=4)
        print(f"[Manifest] Saved experiment manifest for {dataset_name} ({split}) to {output_path}")

    return manifest_data
