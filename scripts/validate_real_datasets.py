import os
import sys
import json
import argparse
from typing import Dict, Any, List

# Ensure repository root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.manifest import (
    compute_file_sha256,
    extract_scene_id_from_sample,
    check_for_data_leakage,
    validate_training_dataset_safety
)

def parse_args():
    parser = argparse.ArgumentParser(description="Pre-Flight Real Dataset Validation Tool for Cloud Environments")
    parser.add_argument("--div2k_dir", type=str, default="data/div2k", help="Path to DIV2K directory")
    parser.add_argument("--sidd_dir", type=str, default="data/sidd", help="Path to SIDD directory")
    parser.add_argument("--dnd_dir", type=str, default="data/dnd", help="Path to DND directory")
    parser.add_argument("--output_report", type=str, default="dataset_validation_report.json", help="Output path for validation report")
    return parser.parse_args()

def validate_image_file(filepath: str) -> Dict[str, Any]:
    """Lightweight metadata inspection using Pillow without holding image array in RAM."""
    from PIL import Image
    try:
        with Image.open(filepath) as img:
            width, height = img.size
            mode = img.mode
            channels = len(img.getbands())
        return {
            "valid": True,
            "width": width,
            "height": height,
            "mode": mode,
            "channels": channels,
            "size_bytes": os.path.getsize(filepath)
        }
    except Exception as e:
        return {
            "valid": False,
            "error": str(e),
            "size_bytes": os.path.getsize(filepath) if os.path.exists(filepath) else 0
        }

def validate_real_datasets():
    args = parse_args()
    print("============================================================")
    print("       PRE-FLIGHT REAL DATASET VALIDATION PROTOCOL          ")
    print("============================================================")
    print(f"DIV2K Target: {args.div2k_dir}")
    print(f"SIDD Target:  {args.sidd_dir}")
    print(f"DND Target:   {args.dnd_dir}")
    print("============================================================\n")

    report: Dict[str, Any] = {
        "status": "PASS",
        "datasets": {},
        "leakage_checks": {},
        "safety_checks": {},
        "errors": []
    }

    # 1. Safety Check: Verify DND training prohibition guard
    print("[1/4] Auditing DND Training Prohibition Guard...")
    try:
        validate_training_dataset_safety(args.dnd_dir, split="train")
        report["errors"].append("CRITICAL: validate_training_dataset_safety failed to reject DND training!")
        report["status"] = "FAIL"
    except ValueError as e:
        report["safety_checks"]["dnd_training_prohibited"] = "PASS: DND correctly rejected from training loop"
        print(f"  [PASS] {e}")

    # 2. DIV2K Validation
    print("\n[2/4] Validating DIV2K Photographic Dataset...")
    div2k_report: Dict[str, Any] = {"path": args.div2k_dir, "exists": os.path.exists(args.div2k_dir)}
    if not os.path.exists(args.div2k_dir):
        msg = f"DIV2K directory not found: '{args.div2k_dir}'"
        print(f"  [NOTICE] {msg} (Must mount in Kaggle/Colab before full runs)")
        div2k_report["status"] = "NOT_MOUNTED"
        report["datasets"]["div2k"] = div2k_report
    else:
        files = sorted([f for f in os.listdir(args.div2k_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        div2k_report["file_count"] = len(files)
        div2k_report["sample_ids"] = []
        div2k_report["scene_ids"] = []
        div2k_hashes = set()
        duplicate_hashes = []
        malformed_files = []

        for f in files:
            full_path = os.path.join(args.div2k_dir, f)
            try:
                scene_id = extract_scene_id_from_sample(f, dataset_name="div2k")
                div2k_report["scene_ids"].append(scene_id)
            except ValueError as ve:
                malformed_files.append({"file": f, "error": str(ve)})

            f_hash = compute_file_sha256(full_path)
            if f_hash in div2k_hashes:
                duplicate_hashes.append({"file": f, "sha256": f_hash})
            else:
                div2k_hashes.add(f_hash)
            div2k_report["sample_ids"].append(f)

        div2k_report["unique_scenes"] = len(set(div2k_report["scene_ids"]))
        div2k_report["malformed_count"] = len(malformed_files)
        div2k_report["duplicate_hash_count"] = len(duplicate_hashes)
        if malformed_files:
            div2k_report["malformed_samples"] = malformed_files[:5]
            report["errors"].append(f"DIV2K contains {len(malformed_files)} malformed filenames.")
        if len(files) == 0:
            div2k_report["status"] = "EMPTY"
            report["errors"].append("DIV2K directory is empty.")
        else:
            div2k_report["status"] = "VALID"
            # Sample check first image
            meta = validate_image_file(os.path.join(args.div2k_dir, files[0]))
            div2k_report["sample_image_meta"] = meta
            print(f"  [PASS] DIV2K: Found {len(files)} valid images across {div2k_report['unique_scenes']} unique scenes.")
        report["datasets"]["div2k"] = div2k_report

    # 3. SIDD Validation
    print("\n[3/4] Validating SIDD Genuine Camera Benchmark...")
    sidd_report: Dict[str, Any] = {"path": args.sidd_dir, "exists": os.path.exists(args.sidd_dir)}
    if not os.path.exists(args.sidd_dir):
        msg = f"SIDD directory not found: '{args.sidd_dir}'"
        print(f"  [NOTICE] {msg} (Must mount in Kaggle/Colab before full runs)")
        sidd_report["status"] = "NOT_MOUNTED"
        report["datasets"]["sidd"] = sidd_report
    else:
        # Check subfolders or root
        noisy_dir = os.path.join(args.sidd_dir, "noisy") if os.path.exists(os.path.join(args.sidd_dir, "noisy")) else args.sidd_dir
        gt_dir = os.path.join(args.sidd_dir, "gt") if os.path.exists(os.path.join(args.sidd_dir, "gt")) else args.sidd_dir

        noisy_files = sorted([f for f in os.listdir(noisy_dir) if f.lower().endswith(('.png', '.jpg'))])
        gt_files = sorted([f for f in os.listdir(gt_dir) if f.lower().endswith(('.png', '.jpg'))])
        sidd_report["noisy_count"] = len(noisy_files)
        sidd_report["gt_count"] = len(gt_files)
        sidd_scenes = []
        malformed_sidd = []

        for nf in noisy_files:
            try:
                sid = extract_scene_id_from_sample(nf, dataset_name="sidd")
                sidd_scenes.append(sid)
            except ValueError as ve:
                malformed_sidd.append({"file": nf, "error": str(ve)})

        sidd_report["unique_scenes"] = len(set(sidd_scenes))
        sidd_report["malformed_count"] = len(malformed_sidd)
        if len(noisy_files) == 0:
            sidd_report["status"] = "EMPTY"
            report["errors"].append("SIDD noisy directory is empty.")
        else:
            sidd_report["status"] = "VALID"
            meta = validate_image_file(os.path.join(noisy_dir, noisy_files[0]))
            sidd_report["sample_image_meta"] = meta
            print(f"  [PASS] SIDD: Found {len(noisy_files)} noisy crops across {sidd_report['unique_scenes']} scenes.")
        report["datasets"]["sidd"] = sidd_report

    # 4. DND Validation
    print("\n[4/4] Validating DND Held-Out Benchmark...")
    dnd_report: Dict[str, Any] = {"path": args.dnd_dir, "exists": os.path.exists(args.dnd_dir)}
    if not os.path.exists(args.dnd_dir):
        msg = f"DND directory not found: '{args.dnd_dir}'"
        print(f"  [NOTICE] {msg} (Must mount in Kaggle/Colab before final evaluation)")
        dnd_report["status"] = "NOT_MOUNTED"
        report["datasets"]["dnd"] = dnd_report
    else:
        dnd_files = sorted([f for f in os.listdir(args.dnd_dir) if f.lower().endswith(('.mat', '.png', '.jpg', '.h5'))])
        dnd_report["file_count"] = len(dnd_files)
        if len(dnd_files) == 0:
            dnd_report["status"] = "EMPTY"
        else:
            dnd_report["status"] = "VALID"
            print(f"  [PASS] DND: Found {len(dnd_files)} benchmark raw/mat files.")
        report["datasets"]["dnd"] = dnd_report

    # Cross-dataset scene collision check if both DIV2K and SIDD are present
    if div2k_report.get("status") == "VALID" and sidd_report.get("status") == "VALID":
        print("\nVerifying Cross-Dataset Split Isolation...")
        div2k_scenes = set(div2k_report.get("scene_ids", []))
        sidd_scenes_set = set(sidd_scenes)
        overlap = div2k_scenes.intersection(sidd_scenes_set)
        if len(overlap) > 0:
            report["leakage_checks"]["cross_dataset_collision"] = f"FAIL: Overlap found: {list(overlap)[:5]}"
            report["status"] = "FAIL"
            report["errors"].append("Cross-dataset scene collision detected between DIV2K and SIDD!")
        else:
            report["leakage_checks"]["cross_dataset_collision"] = "PASS: Strict scene isolation between DIV2K and SIDD"
            print("  [PASS] Zero scene collisions between DIV2K and SIDD.")

    if report["errors"]:
        report["status"] = "FAIL"

    # Export report
    with open(args.output_report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)
    print(f"\n[Report] Saved complete pre-flight dataset validation report to '{args.output_report}'")

    print("\n============================================================")
    print(f"  PRE-FLIGHT DATASET VALIDATION: {report['status']}")
    print("============================================================\n")
    return report

if __name__ == "__main__":
    validate_real_datasets()

