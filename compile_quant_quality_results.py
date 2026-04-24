#!/usr/bin/env python3
"""
Script to compile evaluation results from multiple methods into a pandas table.
"""

import json
import pandas as pd
from pathlib import Path

BASE_DIR = Path("/root/intelpa-2/cagegaussian")

METHODS = {
    "deforming_nerf": BASE_DIR / "existing_methods/deforming_nerf/output",
    "frosting": BASE_DIR / "existing_methods/frosting/output",
    "games": BASE_DIR / "existing_methods/gaussian_mesh_splatting/output",
    "sugar": BASE_DIR / "existing_methods/sugar/output",
    "ours": BASE_DIR / "gs3d/output",
}

METRICS = ["SSIM", "PSNR", "LPIPS"]


def discover_scenes():
    """Discover all scene names from existing eval directories."""
    scenes = set()
    for method_name, output_dir in METHODS.items():
        for eval_dir in output_dir.glob("eval_*"):
            if eval_dir.is_dir():
                scene_name = eval_dir.name.replace("eval_", "")
                scenes.add(scene_name)
    return sorted(scenes)


def load_results(method_name, output_dir, scene_name):
    """Load results.json for a given method and scene."""
    results_path = output_dir / f"eval_{scene_name}" / "results.json"
    if not results_path.exists():
        return None

    with open(results_path, "r") as f:
        data = json.load(f)

    # The JSON has a nested structure with a method-specific key
    # Extract the first (and typically only) entry
    if data:
        inner_data = next(iter(data.values()))
        return inner_data
    return None


def compile_results():
    """Compile all results into a pandas DataFrame."""
    scenes = discover_scenes()
    print(f"Discovered scenes: {scenes}")

    rows = []
    for method_name, output_dir in METHODS.items():
        for scene_name in scenes:
            results = load_results(method_name, output_dir, scene_name)
            if results:
                row = {
                    "method": method_name,
                    "scene": scene_name,
                }
                for metric in METRICS:
                    row[metric] = results.get(metric)
                rows.append(row)
            else:
                print(f"Warning: No results found for {method_name}/{scene_name}")

    df = pd.DataFrame(rows)
    return df


def get_best_method(scene_df, metric):
    """Get the best method for a metric (higher is better for SSIM/PSNR, lower for LPIPS)."""
    if metric == "LPIPS":
        best_idx = scene_df[metric].idxmin()
    else:
        best_idx = scene_df[metric].idxmax()
    return scene_df.loc[best_idx, "method"]


def main():
    df = compile_results()

    print("\n=== Full Results Table ===")
    print(df.to_string(index=False))

    # Print per-scene results with best method highlighted
    scenes = df["scene"].unique()
    for scene in sorted(scenes):
        print(f"\n{'='*60}")
        print(f"Scene: {scene}")
        print("=" * 60)
        scene_df = df[df["scene"] == scene].copy()
        scene_df = scene_df.set_index("method")[METRICS]
        print(scene_df.to_string())

        # Find best method for each metric
        print("\nBest methods:")
        for metric in METRICS:
            if metric == "LPIPS":
                best_method = scene_df[metric].idxmin()
                best_value = scene_df[metric].min()
            else:
                best_method = scene_df[metric].idxmax()
                best_value = scene_df[metric].max()
            print(f"  {metric}: {best_method} ({best_value:.4f})")

    # Compute mean across scenes for each method
    print(f"\n{'='*60}")
    print("Average Metrics by Method (across all scenes)")
    print("=" * 60)
    avg_df = df.groupby("method")[METRICS].mean()
    print(avg_df.to_string())

    # Find best method for each metric on average
    print("\nBest methods (average):")
    for metric in METRICS:
        if metric == "LPIPS":
            best_method = avg_df[metric].idxmin()
            best_value = avg_df[metric].min()
        else:
            best_method = avg_df[metric].idxmax()
            best_value = avg_df[metric].max()
        print(f"  {metric}: {best_method} ({best_value:.4f})")

    # Save to CSV
    output_path = BASE_DIR / "compiled_results_quant_quality.csv"
    df.to_csv(output_path, index=False)
    print(f"\nResults saved to {output_path}")

    return df


if __name__ == "__main__":
    main()
