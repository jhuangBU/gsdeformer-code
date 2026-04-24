#!/usr/bin/env python3
"""Regenerate GraphDECO cameras.json from transforms_train.json.

This is a lightweight utility used by quantitative evaluation scripts.
It instantiates the bundled GraphDECO `Scene` on a NeRF-synthetic dataset root
(containing `transforms_train.json`) and lets it write `cameras.json` into the
provided `--model_path`.

Note: `Scene` also writes `input.ply` as part of its normal initialization.
"""

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace


class _DummyGaussians:
    def create_from_pcd(self, pcd, spatial_lr_scale):
        return


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Regenerate GraphDECO cameras.json from transforms_train.json via Scene()."
        )
    )
    parser.add_argument(
        "--source_path",
        type=Path,
        required=True,
        help="Dataset root containing transforms_train.json",
    )
    parser.add_argument(
        "--model_path",
        type=Path,
        required=True,
        help="Output folder to write cameras.json into",
    )
    parser.add_argument("--white_background", action="store_true")
    args = parser.parse_args()

    gs_root = Path(__file__).resolve().parent / "gaussian_splatting"
    sys.path.insert(0, str(gs_root))

    from scene import Scene  # noqa: E402

    args.model_path.mkdir(parents=True, exist_ok=True)
    scene_args = SimpleNamespace(
        source_path=str(args.source_path),
        model_path=str(args.model_path),
        images="images",
        eval=False,
        white_background=bool(args.white_background),
        resolution=-1,
        data_device="cuda",
    )

    Scene(
        scene_args,
        _DummyGaussians(),
        load_iteration=None,
        shuffle=False,
        resolution_scales=[],
    )


if __name__ == "__main__":
    main()
