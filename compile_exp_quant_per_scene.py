from pathlib import Path

import math
import pandas as pd


def render_table(cage_type: str, tbl: pd.DataFrame) -> str:
    scenes = list(tbl['scene'].unique())
    methods = ["deforming_nerf", "sugar", "games", "frosting", "vanilla", "ours"]
    assert set(list(tbl['method'].unique())) == set(methods)

    buffer = r"""\begin{tabular}{cc||*{4}{c}}
    \toprule"""

    if cage_type == "deforming_nerf":
        buffer += """
    \multicolumn{5}{c}{NeRF \cite{nerf} Scenes (Automatic Cages)}  \\\\"""
    elif cage_type == "broxy":
        buffer += """
    \multicolumn{5}{c}{DeformingNeRF \cite{deforming-nerf} Scenes (Manual Cages)} \\\\"""
    else:
        assert False

    buffer += r"""
    \multirow{2}{*}{Scene} & \multirow{2}{*}{Method} & training & preprocess & deform & render  \\
    & & (sec$\downarrow$) & (ms$\downarrow$) & (ms$\downarrow$/FPS$\uparrow$) & (ms$\downarrow$/FPS$\uparrow$) \\
    \midrule
"""

    def decorate_value(col: pd.Series, val: float, nan_default: str, s_factory) -> str:
        if pd.isna(val):
            return nan_default
        else:
            s = s_factory(val)
            col = col.dropna().sort_values()[:2]
            s = f"\\fst{{{s}}}" if val == col.iloc[0] else s
            s = f"\\snd{{{s}}}" if val == col.iloc[1] else s
            return s

    for scene in scenes:
        scene_tbl = tbl[tbl['scene'] == scene]
        for idx, method in enumerate(methods):
            record = scene_tbl[scene_tbl['method'] == method].iloc[0]
            train = decorate_value(
                scene_tbl["train_sec"], record["train_sec"],
                r"\fst{N/A}", lambda v: f"{v:.2f}"
            )

            preproc = decorate_value(
                scene_tbl["preprocess_ms"], record["preprocess_ms"],
                r"--", lambda v: f"{v:.2f}"
            )

            deform = decorate_value(
                scene_tbl['deform_ms'], record['deform_ms'],
                r"--", lambda v: f"{v:.2f} / {1000/v:.2f}FPS"
            )

            render = decorate_value(
                scene_tbl['render_ms'], record['render_ms'],
                r"--", lambda v: f"{v:.2f} / {1000/v:.2f}FPS"
            )

            scene_name = scene.replace("nerf_", "")
            method_name = {
                "deforming_nerf": "DeformingNeRF",
                "sugar": "SuGaR",
                "games": "GaMeS",
                "frosting": "Frosting",
                "vanilla": "Vanilla 3DGS",
                "ours": "Ours"
            }[method]

            first_col = f"\\midrule \\multirow{{{len(methods)}}}{{*}}{{{scene_name}}}" if idx == 0 else ""

            buffer += f"    {first_col} & {method_name} & {train} & {preproc} & {deform} & {render} \\\\ \n"

    buffer += "    \\bottomrule\n"
    buffer += "\\end{tabular}"

    return buffer

def main():
    for cage_type in ["broxy", "deforming_nerf"]:
        train_stat = pd.read_csv(f"exp_{cage_type}_all_train_stats.csv")
        train_stat = train_stat.drop('Unnamed: 0', axis=1)
        for scene in train_stat['scene'].unique():
            new_row = pd.DataFrame({'method': ['ours'], 'scene': [scene], 'time_sec': [math.nan]})
            train_stat = pd.concat([train_stat, new_row], ignore_index=True)
        # train_stat['cage_type'] = cage_type
        # print(train_stat)

        deform_stat = pd.read_csv(f"exp_{cage_type}_all_deform_stats.csv")
        deform_stat = deform_stat.drop('Unnamed: 0', axis=1)
        deform_stat_pivoted = deform_stat.pivot(
            index=['method', 'scene'],
            columns='stage',
            values='time_ms'
        ).reset_index()
        deform_stat_pivoted = deform_stat_pivoted.rename(columns={
            'preprocess': 'preprocess_ms',
            'deform': 'deform_ms',
            'render': 'render_ms'
        })
        # print(deform_stat)
        # print(deform_stat_pivoted)

        result = pd.merge(
            train_stat,
            deform_stat_pivoted,
            on=['method', 'scene'],
            how='outer'
        )
        result = result.rename(columns={'time_sec': 'train_sec'})

        print(result)

        rendered = render_table(cage_type, result)
        Path(f"compiled_exp_{cage_type}_per_scene_table.tex").write_text(rendered, encoding="UTF-8")


if __name__ == '__main__':
    main()

