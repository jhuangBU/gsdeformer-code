import asyncio
import itertools
import time
from dataclasses import field, dataclass
from pathlib import Path
from typing import List, Dict

import anyio
import numpy as np
import open3d.geometry as ogeo
import open3d.visualization.gui as ogui
import open3d.visualization.rendering as ordr
import trimesh
from anyio import TASK_STATUS_IGNORED
from anyio.abc import TaskStatus
from loguru import logger as log
from omegaconf import MISSING, OmegaConf

import gsdeformer.hack_add_gs3d_to_sys_path
from scene.dataset_readers import CameraInfo
from scene.gaussian_model import GaussianModel
from gsdeformer.deform.algorithm_ui import build_deformer
from gsdeformer.deform.model_util import merge_model
from gsdeformer.editorv2.__main__ import GaussianRenderer, CageRenderer, CompositeRenderer
from gsdeformer.editorv2.utils_camera import cvt_camera_3dgs_to_o3d_intrin, cvt_camera_3dgs_to_o3d_extrin, \
    extract_extrinsic
from gsdeformer.editorv2.utils_color import RED, BLUE, WHITE_RGBA
from gsdeformer.editorv2.utils_data import load_model, list_cameras
from gsdeformer.utils_config import parse_args
from gsdeformer.utils_ui import ValueEmitter


@dataclass
class ModelEntry:
    model_path: Path = MISSING
    iteration: int = -1
    src_cage: Path = MISSING
    dst_cage: Path = MISSING


@dataclass
class ViewerArguments:
    cam_source_path: Path = MISSING
    init_cam_idx: int = 0
    models: Dict[str, ModelEntry] = field(default_factory=dict)
    white_bg: bool = False


@dataclass
class CompositedScene:
    composited: GaussianModel
    src_cages: List[ogeo.LineSet]
    dst_cages: List[ogeo.LineSet]


def load_composite_scenes(entries: Dict[str, ModelEntry]) -> CompositedScene:

    def _cage_to_display(cage: trimesh.Trimesh, color):
        s = ogeo.LineSet.create_from_triangle_mesh(cage.as_open3d)
        s = s.paint_uniform_color(color[:, np.newaxis])
        return s

    ret = None
    src, dst = [], []
    for scene in entries.values():
        model = load_model(scene.model_path, scene.iteration)
        src_cage = trimesh.load_mesh(scene.src_cage)
        dst_cage = trimesh.load_mesh(scene.dst_cage)
        src.append(_cage_to_display(src_cage, BLUE))
        dst.append(_cage_to_display(dst_cage, RED))
        deformer = build_deformer("all", model, src_cage, pbar=True)
        deformed = deformer(dst_cage)
        if ret is None:
            ret = deformed
        else:
            ret = merge_model(ret, deformed)
    return CompositedScene(ret, src, dst)


class SceneView:

    def __init__(self, init_cam: CameraInfo, composite: CompositedScene, white_bg=False, target_fps:int=20):
        self._target_fps = target_fps
        # UI state
        self._view = ogui.SceneWidget()
        self._window = None # belonging window cache
        # render init camera
        self._init_cam = init_cam
        self._init_cam_intrin = cvt_camera_3dgs_to_o3d_intrin(self._init_cam)
        self._init_cam_extrin = cvt_camera_3dgs_to_o3d_extrin(self._init_cam)
        # render renderers
        self.gs_renderer = GaussianRenderer(ValueEmitter(composite.composited), white_bg, caching=False)
        self.cage_renderer = CageRenderer(
            [ValueEmitter(a) for a in itertools.chain(composite.src_cages, composite.dst_cages)],
            caching=False
        )
        self._renderer = CompositeRenderer(self.gs_renderer, self.cage_renderer, caching=False)
        # render parameter & result
        self.camera_param = ValueEmitter((self._init_cam_intrin, self._init_cam_extrin))
        self.current_render_camera_param = None
        self.current_render = None # np.ndarray[H,W,3],0-1
        self._current_render_cage_depth = None # np.ndarray[H,W], near-far 0-1 depth image of cage render

    # region UI

    def build_ui(self, window):
        view = self._view
        view.scene = ordr.Open3DScene(window.renderer)
        self._window = window

        self.reset_view()

        accepted_events = {ogui.MouseEvent.DRAG, ogui.MouseEvent.WHEEL}

        def _on_mouse(e):
            if e.type in accepted_events:
                self._ui_update_camera_and_render()
                return ogui.SceneWidget.EventCallbackResult.HANDLED
            return ogui.Widget.EventCallbackResult.IGNORED

        view.set_on_mouse(_on_mouse)

        return view

    def _ui_update_camera_and_render(self):
        intr, ex = self._get_view_intrin_extrin()
        self.camera_param.emit((intr, ex))
        self._update_render()

    # endregion

    # region viewpoint control

    def reset_view(self):
        self.set_view(self._init_cam_extrin)

    def set_view(self, extrin):
        bounds = self._view.scene.bounding_box
        intrin, _ = self._get_view_intrin_extrin()
        self._view.setup_camera(intrin, extrin, bounds)
        self.camera_param.emit((intrin, extrin))
        self._update_render()

    # endregion

    # region render

    def _get_view_intrin_extrin(self):
        view = self._view
        view_height = view.frame.height
        view_width = view.frame.width
        intr = cvt_camera_3dgs_to_o3d_intrin(
            self._init_cam,
            height=view_height if view_height else 512,
            width=view_width if view_width else 512,
        )
        ex = extract_extrinsic(view.scene)
        return intr, ex

    def _update_render(self):
        self.current_render_camera_param = None

    async def run(self, task_status: TaskStatus[None] = TASK_STATUS_IGNORED):
        task_status.started()
        while True:
            start = time.time()
            # TODO: move camera pose caching behaviour into Renderer as well?
            now_intrin, now_extrin = self.camera_param.val
            self.current_render_camera_param = (now_intrin, now_extrin)
            image, c_depth, c_depth_normed = self._renderer.render(self._window, now_intrin, now_extrin)
            image = ogeo.Image(image)
            self.current_render = image
            self._current_render_cage_depth = c_depth_normed
            self._view.scene.set_background(WHITE_RGBA[:, np.newaxis], image)
            end = time.time()

            elapsed = int((end - start) * 1000)
            delay = 1000 // self._target_fps
            await asyncio.sleep(max(0, delay - elapsed) / 1000) # sleep for 20FPS

    # endregion


class Viewer:

    def __init__(self, args: ViewerArguments):
        self._args = args

        self._model = load_composite_scenes(args.models)

        self._cameras = list_cameras(args.cam_source_path)
        init_cam = self._cameras[args.init_cam_idx]

        self._cage_view = SceneView(init_cam, self._model, args.white_bg)

    def _build_ui(self, app):
        window = app.create_window("CompositeViewer", width=512, height=512)
        cage_view = self._cage_view.build_ui(window)
        window.add_child(cage_view)

        def _on_layout(ctx):
            r = window.content_rect

            view_width = r.width
            view_height = max(int(view_width * 3/4), r.height)

            rframe = ogui.Rect(r.x, r.y, view_width, view_height)
            cage_view.frame = rframe

        window.set_on_layout(_on_layout)

    async def run(self):
        app: ogui.Application = ogui.Application.instance
        app.initialize()

        async def _app_event_loop(task_status: TaskStatus[None] = TASK_STATUS_IGNORED):
            task_status.started()
            running = True
            while running:
                running = app.run_one_tick()
                await asyncio.sleep(0)

        async def _main_window(task_status: TaskStatus[None] = TASK_STATUS_IGNORED):
            self._build_ui(app)
            async with anyio.create_task_group() as tg:
                await tg.start(self._cage_view.run)
                task_status.started()
                await asyncio.Future()

        async with anyio.create_task_group() as tg:
            await tg.start(_app_event_loop)
            await tg.start(_main_window)


async def main(args=None):
    import taichi as ti
    ti.init(arch=ti.gpu)
    args = parse_args(ViewerArguments, args)
    log.debug("running wih configuration: \n{}", OmegaConf.to_container(args))
    await Viewer(args).run()



if __name__ == '__main__':
    asyncio.run(main())