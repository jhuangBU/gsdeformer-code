from pathlib import Path
from typing import List

from loguru import logger as log

# noinspection PyUnresolvedReferences
import gsdeformer.hack_add_gs3d_to_sys_path
from gs3d.scene.dataset_readers import sceneLoadTypeCallbacks, CameraInfo
from gs3d.scene.gaussian_model import GaussianModel


def list_cameras(source_path: Path) -> List[CameraInfo]:
    eval = True
    if (source_path / "sparse").exists():
        scene_info = sceneLoadTypeCallbacks["Colmap"](str(source_path), images=None, eval=eval)
    elif (source_path / "transforms_train.json").exists():
        print("Found transforms_train.json file, assuming Blender data set!")
        scene_info = sceneLoadTypeCallbacks["Blender"](str(source_path), white_background=True, eval=eval)
    else:
        assert False, "Could not recognize scene type!"

    cameras = [*scene_info.train_cameras, *scene_info.test_cameras]

    return cameras


def load_model(model_path: Path, iteration: int = -1) -> GaussianModel:
    if iteration == -1:
        iters = [p for p in (model_path / "point_cloud").glob("iteration_*")]
        iters = [p for p in iters if p.is_dir()]
        iters = [int(p.name.split("_")[1]) for p in iters]
        max_iter = max(iters)
        iteration = max_iter

    ply_path = model_path / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply"
    log.trace("loading model iter {} from {}", iteration, ply_path)
    model = GaussianModel(sh_degree=3)
    model.load_ply(str(ply_path))

    return model
