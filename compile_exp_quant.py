import json
from pathlib import Path
from typing import List

import pandas as pd

ROOT_FOLDER = Path(__file__).parent
METHODS_FOLDER = ROOT_FOLDER / "existing_methods"

DEFORM_SCENES = [
    "nerf_lego",
    "nerf_chair",
    "nerf_ficus",
    "nerf_hotdog"
]

DEFORMING_NERF_SCENES = [
    "nerf_lego",
    "nerf_chair",
    "nsvf_robot",
    "nsvf_toad"
]

CAGE_TYPE = ["broxy", "deforming_nerf"]

def read_deforming_nerf_training_stats(scenes: List[str]) -> pd.DataFrame:
    table = []
    for scene in scenes:
        p = METHODS_FOLDER / "deforming_nerf" / "opt" / "ckpt" / scene / "time_secs.txt"
        time_sec = float(p.read_text(encoding="UTF-8"))
        table.append(("deforming_nerf", scene, time_sec))
    return pd.DataFrame(table, columns=['method', 'scene', 'time_sec'])

def read_deforming_nerf_deformation_stats(scenes: List[str], cage_type: str) -> pd.DataFrame:
    table = []
    ctype = cage_type
    for scene in scenes:
        p = METHODS_FOLDER / "results" / "deforming_nerf" / f"benchmark_{scene}_{ctype}.json"
        d = json.loads(p.read_text(encoding="UTF-8"))
        table.append(("deforming_nerf", ctype, scene, "preprocess", d["preprocess_ms_avg"]))
        table.append(("deforming_nerf", ctype, scene, "deform", d["deform_ms_avg"]))
        table.append(("deforming_nerf", ctype, scene, "render", d["render_ms_avg"]))

    return pd.DataFrame(table, columns=['method', 'cage_type', 'scene', 'stage', 'time_ms'])

def read_games_training_stats(scenes: List[str]) -> pd.DataFrame:
    table = []
    for scene in scenes:
        p = METHODS_FOLDER / "gaussian_mesh_splatting" / "output" / scene / "training_time.json"
        time_sec = json.loads(p.read_text(encoding="UTF-8"))["training_time_sec"]
        table.append(("games", scene, time_sec))
    return pd.DataFrame(table, columns=['method', 'scene', 'time_sec'])

def read_games_deformation_stats(scenes: List[str], cage_type: str) -> pd.DataFrame:
    table = []
    ctype = cage_type
    for scene in scenes:
        p = METHODS_FOLDER / "results" / "games" / f"benchmark_{scene}_{ctype}.json"
        d = json.loads(p.read_text(encoding="UTF-8"))
        preprocess = d["avg_preprocess_time_ms"] + d["avg_preprocess_time2_ms"]
        table.append(("games", ctype, scene, "preprocess", preprocess))
        table.append(("games", ctype, scene, "deform", d["avg_deform_time_ms"]))
        table.append(("games", ctype, scene, "render", d["avg_render_time_ms"]))

    return pd.DataFrame(table, columns=['method', 'cage_type', 'scene', 'stage', 'time_ms'])

def read_sugar_training_stats(scenes: List[str]) -> pd.DataFrame:
    table = []
    for scene in scenes:
        base = METHODS_FOLDER / "sugar" / "output" / scene
        p = base / "training_time_coarse.json"
        coarse_time_sec = json.loads(p.read_text(encoding="UTF-8"))["training_time_sec"]
        p = base / "fine_benchmark.json"
        fine_time_sec = json.loads(p.read_text(encoding="UTF-8"))["training_time_sec"]
        table.append(("sugar", scene, coarse_time_sec + fine_time_sec))
    return pd.DataFrame(table, columns=['method', 'scene', 'time_sec'])

def read_sugar_deformation_stats(scenes: List[str], cage_type: str) -> pd.DataFrame:
    table = []
    ctype = cage_type
    for scene in scenes:
        p = METHODS_FOLDER / "results" / "sugar" / f"benchmark_{scene}_{ctype}.json"
        d = json.loads(p.read_text(encoding="UTF-8"))
        table.append(("sugar", ctype, scene, "preprocess", d["avg_time_preproc_ms"]))
        table.append(("sugar", ctype, scene, "deform", d["avg_time_deform_ms"]))
        table.append(("sugar", ctype, scene, "render", d["avg_time_render_ms"]))

    return pd.DataFrame(table, columns=['method', 'cage_type', 'scene', 'stage', 'time_ms'])

def read_vanilla_training_stats(scenes: List[str]) -> pd.DataFrame:
    table = []
    for scene in scenes:
        base = ROOT_FOLDER / "gs3d" / "output" / scene
        p = base / "training_time.json"
        time_sec = json.loads(p.read_text(encoding="UTF-8"))["training_time_sec"]
        table.append(("vanilla", scene, time_sec))
    return pd.DataFrame(table, columns=['method', 'scene', 'time_sec'])

def read_ours_deformation_stats(scenes: List[str], cage_type: str) -> pd.DataFrame:
    table = []
    ctype = cage_type
    for scene in scenes:
        p = ROOT_FOLDER / f"exp-quant-benchmark-{ctype}" / f"{scene}_all_benchmark.json"
        d = json.loads(p.read_text(encoding="UTF-8"))
        table.append(("ours", ctype, scene, "preprocess", d["preprocess_time_ms_avg"]))
        table.append(("ours", ctype, scene, "deform", d["deform_time_ms_avg"]))
        table.append(("ours", ctype, scene, "render", d["render_time_ms_avg"]))

    return pd.DataFrame(table, columns=['method', 'cage_type', 'scene', 'stage', 'time_ms'])

def read_frosting_training_stats(scenes: List[str]) -> pd.DataFrame:
    table = []
    for scene in scenes:
        base = METHODS_FOLDER / "frosting" / "output" / "vanilla_gs" / scene
        p = base / "training_time_coarse.json"
        coarse_time_sec = json.loads(p.read_text(encoding="UTF-8"))["training_time_sec"]
        p = base / "fine_benchmark.json"
        fine_time_sec = json.loads(p.read_text(encoding="UTF-8"))["training_time_sec"]
        table.append(("frosting", scene, coarse_time_sec + fine_time_sec))
    return pd.DataFrame(table, columns=['method', 'scene', 'time_sec'])

def read_frosting_deformation_stats(scenes: List[str], cage_type: str) -> pd.DataFrame:
    table = []
    ctype = cage_type
    for scene in scenes:
        p = METHODS_FOLDER / "results" / "frosting" / f"benchmark_{scene}_{ctype}.json"
        d = json.loads(p.read_text(encoding="UTF-8"))
        table.append(("frosting", ctype, scene, "preprocess", d["avg_time_preproc_ms"]))
        table.append(("frosting", ctype, scene, "deform", d["avg_time_deform_ms"]))
        table.append(("frosting", ctype, scene, "render", d["avg_time_render_ms"]))

    return pd.DataFrame(table, columns=['method', 'cage_type', 'scene', 'stage', 'time_ms'])

def main():
    for cage_type in ["broxy", "deforming_nerf"]:
        if cage_type == "broxy":
            training_scenes = DEFORM_SCENES.copy()
            benchmark_scenes = DEFORM_SCENES
        elif cage_type == "deforming_nerf":
            training_scenes = DEFORMING_NERF_SCENES
            benchmark_scenes = DEFORMING_NERF_SCENES
        else:
            assert False

        method_ordering = ["ours", "vanilla", "deforming_nerf", "sugar", "games", "frosting"]

        all_train_stats = pd.concat([
            read_deforming_nerf_training_stats(training_scenes),
            read_games_training_stats(training_scenes),
            read_sugar_training_stats(training_scenes),
            read_frosting_training_stats(training_scenes),
            read_vanilla_training_stats(training_scenes)
        ], ignore_index=True)
        all_train_stats['method'] = pd.Categorical(all_train_stats['method'], categories=method_ordering, ordered=True)

        print("--- writing training stats & averages ---")
        all_train_stats["time_sec"] = all_train_stats["time_sec"].round(2)
        all_train_stats.to_csv(f"exp_{cage_type}_all_train_stats.csv")
        avg = all_train_stats.groupby("method").agg({'time_sec': 'mean'}).reset_index()
        avg = avg.sort_values('method')
        for col in avg.columns:
            if col != 'method':
                avg[col] = avg[col].round(2)
        avg.to_csv(f"exp_{cage_type}_all_train_stats_avg.csv")

        all_deform_stats = pd.concat([
            read_deforming_nerf_deformation_stats(benchmark_scenes, cage_type),
            read_games_deformation_stats(benchmark_scenes, cage_type),
            read_sugar_deformation_stats(benchmark_scenes, cage_type),
            read_frosting_deformation_stats(benchmark_scenes, cage_type),
            read_ours_deformation_stats(benchmark_scenes, cage_type)
        ], ignore_index=True)
        all_deform_stats['method'] = pd.Categorical(all_deform_stats['method'], categories=method_ordering, ordered=True)

        print("--- writing deformation stats & averages ---")
        all_deform_stats["time_ms"] = all_deform_stats["time_ms"].round(2)
        all_deform_stats.to_csv(f"exp_{cage_type}_all_deform_stats.csv")
        avg = all_deform_stats.groupby(["method", "cage_type", "stage"]).agg({'time_ms': 'mean'}).reset_index()
        avg = avg.sort_values('method')
        avg.to_csv(f"exp_{cage_type}_all_deform_stats_avg.csv")

        broxy_avg = avg[avg['cage_type'] == cage_type].drop('cage_type', axis=1)
        broxy_avg_pivoted = broxy_avg.pivot(index='method', columns='stage', values='time_ms').reset_index()
        broxy_avg_pivoted.columns = ['method', 'deform_time_ms', 'preprocess_time_ms', 'render_time_ms']
        broxy_avg_pivoted = broxy_avg_pivoted[['method', 'preprocess_time_ms', 'deform_time_ms', 'render_time_ms']]

        for col in broxy_avg_pivoted.columns:
            if col != 'method':
                broxy_avg_pivoted[col] = broxy_avg_pivoted[col].round(2)

        deform_time_fps = (1000 / broxy_avg_pivoted["deform_time_ms"]).round(2)
        broxy_avg_pivoted["deform_time_summary"] = [
            f"{avg:.2f} ({fps:.2f}FPS)"
            for avg, fps in zip(broxy_avg_pivoted["deform_time_ms"], deform_time_fps)
        ]

        render_time_fps = (1000 / broxy_avg_pivoted["render_time_ms"]).round(2)
        broxy_avg_pivoted["render_time_summary"] = [
            f"{avg:.2f} ({fps:.2f}FPS)"
            for avg, fps in zip(broxy_avg_pivoted["render_time_ms"], render_time_fps)
        ]

        broxy_avg_pivoted = broxy_avg_pivoted.sort_values('method')

        broxy_avg_pivoted.to_csv(f"exp_{cage_type}_all_deform_stats_pivoted.csv")

if __name__ == '__main__':
    main()