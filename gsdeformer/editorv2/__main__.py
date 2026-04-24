import abc
import argparse
import asyncio
import contextlib
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, List, Callable, Optional, Set, TypeVar, Generic

import anyio
import dill
import numpy as np
import open3d.geometry as ogeo
import open3d.io as oio
import open3d.utility as outil
import open3d.visualization as oviz
import open3d.visualization.gui as ogui
import open3d.visualization.rendering as ordr
import torch
import trimesh
from anyio import TASK_STATUS_IGNORED
from anyio.abc import TaskStatus

from loguru import logger as log
from omegaconf import OmegaConf, MISSING

# noinspection PyUnresolvedReferences
import gsdeformer.hack_add_gs3d_to_sys_path
from scene.dataset_readers import CameraInfo
from scene.gaussian_model import GaussianModel
from arguments import PipelineParams
from depth_gaussian_renderer import render

from arguments import PipelineParams
from scene.cameras import Camera
from gsdeformer.deform.algorithm_ui import build_deformer
from gsdeformer.editor.utils_camera import cvt_camera_info_to_camera
from gsdeformer.editorv2.utils_camera import cvt_camera_3dgs_to_o3d_extrin, cvt_camera_3dgs_to_o3d_intrin, \
    cvt_camera_o3d_to_3dgs, picklize_camera_param, unpicklize_camera_param, extract_extrinsic, camera_to_camera_info
from gsdeformer.editorv2.utils_data import load_model, list_cameras_full
from gsdeformer.utils_config import parse_args
from gsdeformer.utils_ui import ValueEmitter, EventEmitter
from gsdeformer.editorv2.utils_color import RED, GREEN, BLUE, WHITE_RGBA, WHITE, BLACK


def _build_point_cloud(model_mean, color):
    cloud = ogeo.PointCloud()
    cloud.points = outil.Vector3dVector(model_mean)
    colors = np.repeat(color[np.newaxis, :], model_mean.shape[0], axis=0)
    cloud.colors = outil.Vector3dVector(colors)
    return cloud


@dataclass
class EditorArguments:
    algorithm: str = "all" # ablation switch, use full algorithm or "mean_only" algorithm
    cage_path: Path = MISSING # path to cage location
    init_anchors: str = "" # comma separated list of anchor vertices - vertices that should not change during ARAP

    init_cam_idx: int = 0 # init camera index
    source_path: Path = MISSING # path to dataset folder
    model_path: Path = MISSING # path to trained GS3D folder
    iteration: int = -1 # the iter of model to load, -1 means to pick the latest

    white_bg: bool = False # white background in rendering?


class ModelDeformController:
    """
    object for handling GaussianModel deformation
    """

    def __init__(self, algorithm: str, model: GaussianModel, src_cage: trimesh.Trimesh, target_fps:int=20):
        self._target_fps = target_fps

        self._deformer = build_deformer(algorithm, model, src_cage, pbar=False)

        self._pending_deform = None
        self._dst_cage = src_cage.copy()

        self.model = ValueEmitter(model)
        self.time_sec = ValueEmitter(0.0)

    def on_dst_cage(self, ls: ogeo.LineSet):
        self._dst_cage.vertices[:,:] = np.asarray(ls.points)

    def on_deform(self):
        if self._pending_deform is None:
            self._pending_deform = asyncio.Future()
        return self._pending_deform

    async def run(self, task_status: TaskStatus[None] = TASK_STATUS_IGNORED):
        task_status.started()
        while True:
            start = time.time()
            deformed = False
            if self._pending_deform is not None:
                deformed = True
                self._pending_deform.set_result(None)
                self._pending_deform = None
                model = self._deformer(self._dst_cage)
                self.model.emit(model)
            end = time.time()

            if deformed:
                self.time_sec.emit(end - start)

            elapsed = int((end - start) * 1000)
            delay = 1000 // self._target_fps
            await asyncio.sleep(max(0, delay - elapsed) / 1000) # sleep for 20FPS


class MeshEditController:
    """
    object for handling mesh deformation logic
    """

    def __init__(self, cage: trimesh.Trimesh, debug:bool=False, init_anchors:Optional[Set[int]]=None):
        src_cage = ogeo.LineSet.create_from_triangle_mesh(cage.as_open3d)
        dst_cage = ogeo.LineSet(src_cage)
        src_cage = src_cage.paint_uniform_color(BLUE[:, np.newaxis])
        dst_cage = dst_cage.paint_uniform_color(RED[:, np.newaxis])
        vertex_cloud = _build_point_cloud(cage.vertices, BLUE)
        for a in init_anchors:
            np.asarray(vertex_cloud.colors)[a] = GREEN

        self._debug = debug
        self._init_mesh = cage
        self.dst_mesh: ogeo.TriangleMesh = cage.as_open3d # TODO: turn this into ValueEmitter
        self._init_anchors = init_anchors if init_anchors else set()

        self._vertices_selected_idx = None
        self._vertices_anchor_indices = set(self._init_anchors)
        self._vertices_selected_og_coord = None
        self._vertices_cached_tree = None

        self.src_cage = ValueEmitter(src_cage)
        self.dst_cage = ValueEmitter(dst_cage)
        self.vertex_cloud = ValueEmitter(vertex_cloud)

    # region basic cage editing

    def reset_dst_cage(self):
        """
        reset dst cage to src cage
        """
        self.load_dst_cage(self._init_mesh.copy(), level=0.0)

    def load_dst_cage(self, loaded: trimesh.Trimesh, level: float = 1.0):
        """
        set dst_cage to loaded, after interpolating with vertex position in src_cage
        """
        assert 0.0 <= level <= 1.0

        m = np.asarray(self.src_cage.val.points) * (1 - level) + loaded.vertices * level

        self._vertices_selected_idx = None
        self._vertices_anchor_indices = set(self._init_anchors)
        self._vertices_selected_og_coord = None
        self._vertices_cached_tree = None

        self.dst_mesh = loaded
        np.asarray(self.dst_cage.val.points)[:, :] = m
        self.dst_cage.notify_inplace_mutation()
        self.vertex_cloud.emit(_build_point_cloud(m, BLUE))

    # endregion

    # region vertex selection & ARAP deform

    def get_selected_vertex_og_coord(self) -> Optional[np.ndarray]:
        """
        get the position of the currently selected vertex
        """
        return self._vertices_selected_og_coord

    def on_try_select_vertex(self, coord: Optional[np.array]):
        """
        given unprojected coordinate of a UI click:
        * if clicking somewhere else / nothing(None), unselect current selection
        * if clicking on a vertex, select it
        * if double-clicking on a vertex, toggle it between ARAP anchor state
        """
        # find vertices (if needed)
        idx = None
        if coord is not None:
            # get nearest cage vertex to select
            if self._vertices_cached_tree is None:
                self._vertices_cached_tree = ogeo.KDTreeFlann(self.vertex_cloud.val)
            [k, idx, dist] = self._vertices_cached_tree.search_knn_vector_3d(coord, 1)

            idx = idx[0]
            dist = np.asarray(dist)[0]

            if dist >= 0.01:
                idx = None

        # performs update

        updated = False
        old_idx = self._vertices_selected_idx
        # double click for anchoring selection
        if idx and idx == old_idx:
            if idx in self._vertices_anchor_indices:
                self._vertices_anchor_indices.remove(idx)
                updated = True
            else:
                self._vertices_anchor_indices.add(idx)
                updated = True
            log.debug("updated anchor set: {}", self._vertices_anchor_indices)
        # single click for change of focus
        else:
            if old_idx:
                self._vertices_selected_idx = None
                self._vertices_selected_og_coord = None
                updated = True
            if idx:
                self._vertices_selected_idx = idx
                self._vertices_selected_og_coord = (np.asarray(self.vertex_cloud.val.points)[idx]).copy()
                updated = True

        # update color

        if updated:
            color = np.asarray(self.vertex_cloud.val.colors)
            color[:] = BLUE
            if self._vertices_selected_idx is not None:
                color[self._vertices_selected_idx] = RED
            for a in self._vertices_anchor_indices:
                color[a] = GREEN
            self.vertex_cloud.notify_inplace_mutation()

    def on_drag_selected_vertex(self, delta: np.array):
        """
        adds the delta to the original coord of the selected vertex, consider it as target handle for ARAP deform
        """
        idx = self._vertices_selected_idx
        og_coord = self._vertices_selected_og_coord
        new_coord = og_coord + delta

        self._vertices_cached_tree = None

        ctx = outil.VerbosityContextManager(outil.VerbosityLevel.Debug) if self._debug else contextlib.nullcontext()
        with ctx as cm:
            static_ids = list(self._vertices_anchor_indices)
            static_pos = np.asarray(self.vertex_cloud.val.points)[static_ids]
            constraint_ids = outil.IntVector([*static_ids, idx])
            constraint_pos = outil.Vector3dVector([*static_pos, new_coord])
            self.dst_mesh = self.dst_mesh.deform_as_rigid_as_possible(
                constraint_ids,
                constraint_pos,
                max_iter=10
            )

        dst_cage = ogeo.LineSet.create_from_triangle_mesh(self.dst_mesh)
        dst_cage.colors = self.dst_cage.val.colors
        self.dst_cage.emit(dst_cage)
        vertex_cloud = _build_point_cloud(np.asarray(self.dst_mesh.vertices), BLUE)
        self.vertex_cloud.emit(vertex_cloud)

    # endregion


T = TypeVar("T")


class CachingRenderer(abc.ABC, Generic[T]):

    def __init__(self, caching: bool = True):
        self._caching = caching
        self._geometry_updated = False
        self._last_camera_param = None
        self._last_render = None

    def has_pending_update(self) -> bool:
        return self._geometry_updated or not self._caching

    def _notify_geometry_updated(self):
        self._geometry_updated = True

    def _do_render(self, window: ogui.Window, intrin, extrin) -> T:
        pass

    def render(self, window: ogui.Window, intrin, extrin) -> T:
        last_intrin, last_extrin = (self._last_camera_param if self._last_camera_param else (None, None))
        camera_param_updated = intrin != last_intrin or (extrin != last_extrin).any()

        if camera_param_updated or self.has_pending_update():
            self._geometry_updated = False
            self._last_camera_param = (intrin, extrin)
            self._last_render = self._do_render(window, intrin, extrin)

        return self._last_render


class GaussianRenderer(CachingRenderer[Tuple[np.ndarray, np.ndarray]]):

    def __init__(self, model: EventEmitter[GaussianModel], white_bg: bool = False, caching: bool = True):
        super().__init__(caching)
        self._model = model
        # render params
        self._pipe = PipelineParams(argparse.ArgumentParser())  # mock pipeline params
        self._bg_color = torch.tensor(WHITE if white_bg else BLACK, dtype=torch.float32, device="cuda")
        self._visible = True

        def mon(_): self._notify_geometry_updated()
        self._model.register(mon)

    def set_visible(self, v:bool):
        self._visible = v
        self._notify_geometry_updated()

    @torch.no_grad()
    def _do_render(self, window, intrin, extrin) -> Tuple[np.ndarray, np.ndarray]:
        cam = cvt_camera_o3d_to_3dgs((intrin, extrin))

        if not self._visible:
            width, height = intrin.width, intrin.height
            image = np.full((height, width, 3), fill_value=255, dtype=np.uint8)
            image[:,:] = (self._bg_color.cpu().numpy() * 255).astype(np.uint8)
            depth = np.full((height, width), fill_value=cam.zfar)
        else:
            m = self._model.val
            new_cov = getattr(m, "precomp_cov", None)

            rendered = render(cam, m, self._pipe, self._bg_color, cov3d=new_cov)

            depth = rendered["depth_3dgs"][0].cpu().numpy()
            # we consider all < z_near as zero depth, and are not actually zero depth but is just unfilled
            depth[depth < cam.znear] = cam.zfar

            image = rendered["render"]
            image = torch.clamp(image, 0.0, 1.0)
            image = (image * 255).to(torch.uint8).permute(1, 2, 0)
            image = image.cpu().detach().numpy()
            image = np.ascontiguousarray(image)

        return image, depth


class CageRenderer(CachingRenderer[Tuple[np.ndarray, np.ndarray, np.ndarray]]):
    """
    FIXME: causes segfault on resizing
    """

    def __init__(
            self,
            cages: List[ValueEmitter[ogeo.LineSet]],
            vcloud: Optional[ValueEmitter[ogeo.PointCloud]]=None,
            caching: bool = True
    ):
        super().__init__(caching=caching)
        self._cages = cages
        self._vcloud = vcloud
        # renderer cache
        self._renderer_built_width_height = None
        self._renderer = None
        self._destory_renderer = lambda: None
        # render options
        self._cage_names = [f"cage_{idx}" for idx, _ in enumerate(self._cages)]
        self._cage_visible = [True for _ in self._cages]
        self._vertex_cloud_name = "Vertex Cloud"
        self._vertex_cloud_visible = True

    # region renderer building

    def _build_renderer(self, window, width, height):
        renderer = ordr.OffscreenRenderer(width, height)
        scene = renderer.scene

        ln_mat = self._build_lineset_material(1)
        cbs = []
        for idx, (name, cage) in enumerate(zip(self._cage_names, self._cages)):
            scene.add_geometry(name, cage.val, ln_mat)
            def on_updated(e: ogeo.LineSet):
                scene.remove_geometry(name)
                scene.add_geometry(name, e, ln_mat)
                scene.show_geometry(name, self._cage_visible[idx])
                self._notify_geometry_updated()
            cbs.append(cage.register(on_updated))

        # add vertex cloud to view
        if self._vcloud is not None:
            vc_mat = self._build_point_cloud_material(10, window)
            vertex_cloud_name = self._vertex_cloud_name
            scene.add_geometry(vertex_cloud_name, self._vcloud.val, vc_mat)
            scene.show_geometry(vertex_cloud_name, self._vertex_cloud_visible)
            def on_vertex_cloud_updated(e: ogeo.PointCloud):
                scene.remove_geometry(vertex_cloud_name)
                scene.add_geometry(vertex_cloud_name, e, vc_mat)
                scene.show_geometry(vertex_cloud_name, self._vertex_cloud_visible)
                self._notify_geometry_updated()
            d3 = self._vcloud.register(on_vertex_cloud_updated)
        else:
            d3 = lambda: None

        def _destroy():
            for cb in cbs:
                cb()
            d3()

        return renderer, _destroy

    def _build_lineset_material(self, line_width=1):
        ln_mat = oviz.rendering.MaterialRecord()
        ln_mat.shader = "unlitLine"
        ln_mat.line_width = line_width
        return ln_mat

    def _build_point_cloud_material(self, factor_scaling, window):
        mat = ordr.MaterialRecord()
        mat.shader = "defaultUnlit"
        # Point size is in native pixels, but "pixel" means different things to
        # different platforms (macOS, in particular), so multiply by Window scale
        # factor.
        mat.point_size = factor_scaling * window.scaling
        return mat

    # endregion

    # region visibility setter

    def set_cage_visible(self, idx, vis):
        assert self._renderer
        self._cage_visible[idx] = vis
        self._renderer.scene.show_geometry(self._cage_names[idx], self._cage_visible[idx])
        self._notify_geometry_updated()

    def set_vertex_cloud_visible(self, vis):
        assert self._renderer
        self._vertex_cloud_visible = vis
        self._renderer.scene.show_geometry(self._vertex_cloud_name, self._vertex_cloud_visible)
        self._notify_geometry_updated()

    # endregion

    # region render

    def _do_render(self, window: ogui.Window, intrin, extrin):
        width_height = intrin.width, intrin.height
        if width_height != self._renderer_built_width_height:
            self._renderer_built_width_height = width_height
            self._destory_renderer()
            self._renderer = None
            width, height = width_height
            rdr, destroy = self._build_renderer(window, width, height)
            self._renderer = rdr
            self._destory_renderer = destroy

        self._renderer.setup_camera(intrin, extrin)
        depth_normed = self._renderer.render_to_depth_image()
        depth_normed = np.ascontiguousarray(depth_normed)
        # depth was self.cage_renderer.render_to_depth_image(z_in_view_space=True)
        # faster formula inferred from:
        # https://github.com/isl-org/Open3D/blob/b2d1f78b971030f460a0bf9b7ec33587fba1d5b0/cpp/open3d/visualization/rendering/Renderer.cpp#L94
        depth = np.zeros_like(depth_normed)
        depth[depth_normed == 1.0] = np.inf
        depth[depth_normed != 1.0] = self._renderer.scene.camera.get_near() / (1.0 - depth_normed[depth_normed != 1.0])
        image = self._renderer.render_to_image()
        image = np.ascontiguousarray(image)

        return image, depth, depth_normed

    # endregion


class CompositeRenderer(CachingRenderer[Tuple[np.ndarray, np.ndarray, np.ndarray]]):

    def __init__(self, gr: GaussianRenderer, cr: CageRenderer, caching: bool = True):
        super().__init__(caching=caching)
        self._cage_renderer = cr
        self._gs_renderer = gr

    def has_pending_update(self) -> bool:
        return self._cage_renderer.has_pending_update() or self._gs_renderer.has_pending_update() or super().has_pending_update()

    @torch.no_grad()
    def _do_render(self, window, intrin, extrin):
        c_image, c_depth, c_depth_normed = self._cage_renderer.render(window, intrin, extrin)
        gs_image, gs_depth = self._gs_renderer.render(window, intrin, extrin)

        depth = np.minimum(c_depth, gs_depth)

        image = gs_image
        image[depth == c_depth] = c_image[depth == c_depth]

        return image, depth, c_depth_normed


class SceneCageView:

    def __init__(self, init_cam: CameraInfo, edit: MeshEditController, deform: ModelDeformController, white_bg=False, target_fps:int=20, exit_event: asyncio.Event = None):
        self._ctrl = edit
        self._deform = deform
        self._target_fps = target_fps
        # UI state
        self._view = ogui.SceneWidget()
        self._window = None # belonging window cache
        self._exit = exit_event if exit_event else asyncio.Event()
        # render init camera
        self._init_cam = init_cam
        self._init_cam_intrin = cvt_camera_3dgs_to_o3d_intrin(self._init_cam)
        self._init_cam_extrin = cvt_camera_3dgs_to_o3d_extrin(self._init_cam)
        # render renderers
        self.gs_renderer = GaussianRenderer(deform.model, white_bg, caching=False)
        self.cage_renderer = CageRenderer([edit.src_cage, edit.dst_cage], edit.vertex_cloud, caching=False)
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
            is_down = e.type == ogui.MouseEvent.BUTTON_DOWN
            is_shift_pressed = e.is_modifier_down(ogui.KeyModifier.SHIFT)
            if is_down and is_shift_pressed:
                self._ui_select_vertex(view, e)
                return ogui.Widget.EventCallbackResult.HANDLED

            is_drag = e.type == ogui.MouseEvent.DRAG
            is_ctrl_pressed = e.is_modifier_down(ogui.KeyModifier.CTRL)
            if is_drag and is_shift_pressed and is_ctrl_pressed:
                self._ui_drag_vertex(view, e)
                return ogui.Widget.EventCallbackResult.HANDLED

            if e.type in accepted_events:
                self._ui_update_camera_and_render()
                return ogui.SceneWidget.EventCallbackResult.HANDLED

            return ogui.Widget.EventCallbackResult.IGNORED

        view.set_on_mouse(_on_mouse)

        return view

    def _ui_drag_vertex(self, view: ogui.SceneWidget, e: ogui.MouseEvent):
        # get width, height & pixels
        view_frame = view.frame
        width = view_frame.width
        height = view_frame.height
        now_x = e.x - view_frame.x
        now_y = e.y - view_frame.y

        # get ndc now
        ndc_now_x = 2 * (now_x / width) - 1.0
        ndc_now_y = 2 * ((height - now_y) / height) - 1.0

        # get view og
        og_coord = self._ctrl.get_selected_vertex_og_coord()
        if og_coord is None:
            return
        og_coord = np.append(og_coord, 1.0)
        view_og = view.scene.camera.get_view_matrix() @ og_coord

        # work out view now
        # inverse from ndc to view, math here should help: https://www.songho.ca/opengl/gl_projectionmatrix.html
        project = view.scene.camera.get_projection_matrix()
        view_now_z = view_og_z = view_og[2] # view_now & view_og should fall on the same +Z plane
        view_now_x = (ndc_now_x - view_now_z * project[0,2]) / project[0,0]
        view_now_y = (ndc_now_y - view_now_z * project[1,2]) / project[1,1]
        view_now = np.array([view_now_x, view_now_y, view_now_z, 1.0])

        # build diff vector in view, transform it back into world, no translation because we are offset
        view_diff = view_now - view_og
        view_diff = view_diff[:3]
        inv_world_rot = view.scene.camera.get_view_matrix()[:3,:3].T
        world_diff = np.dot(inv_world_rot, view_diff)

        self._ctrl.on_drag_selected_vertex(world_diff)

    def _ui_select_vertex(self, view: ogui.SceneWidget, e: ogui.MouseEvent):
        # get selected pixel
        view_frame = view.frame
        x = e.x - view_frame.x
        y = e.y - view_frame.y

        depth = self._current_render_cage_depth[y, x]
        if depth == 1.0:
            # just clear selection
            coord = None
        else:
            # back-project
            world = view.scene.camera.unproject(x, y, depth, view_frame.width, view_frame.height)
            coord = np.array([world[0], world[1], world[2]])

        self._ctrl.on_try_select_vertex(coord)

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

    # region visible

    def set_gs_visible(self, vis):
        self.gs_renderer.set_visible(vis)

    def set_src_cage_visible(self, vis):
        self.cage_renderer.set_cage_visible(0, vis)

    def set_dst_cage_visible(self, vis):
        self.cage_renderer.set_cage_visible(1, vis)

    def set_vertex_cloud_visible(self, vis):
        self.cage_renderer.set_vertex_cloud_visible(vis)

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
        while not self._exit.is_set():
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


def add_btn(l, name: str, lam: Callable[[], None]):
    btn = ogui.Button(name)
    btn.set_on_clicked(lam)
    l.add_child(btn)


class ControlMenu:

    def __init__(
            self,
            view: SceneCageView, mesh_ctrl: MeshEditController, deform: ModelDeformController,
            cameras: Tuple[List[CameraInfo], int, int], init_path: Path
    ):
        self._view = view
        self._mesh = mesh_ctrl
        self._deform = deform
        self._cameras, self._train_cnt, self._eval_cnt = cameras
        self._init_path = init_path

    def build_ui(self, window):
        v = ogui.Margins(left=5, right=5)
        v = ogui.Vert(margins=v)

        v.add_child(ogui.Label("Operations"))

        v0 = ogui.CollapsableVert("Deform")
        self._build_ui_cage_interpolate(v0, window)
        self._build_ui_deform(v0)
        v.add_child(v0)

        v0 = ogui.CollapsableVert("View")
        self._build_ui_view_control(v0, window)
        self._build_ui_display_control(v0)
        v.add_child(v0)

        return v

    def _build_ui_cage_interpolate(self, v, window):
        interpolate_level = 0.0
        dst = None

        v.add_child(ogui.Label("Interpolate Cage"))
        level = ogui.Label(f"Interpolate Level - {interpolate_level}")
        v.add_child(level)

        def mod_level(delta):
            nonlocal interpolate_level, dst
            interpolate_level = max(min(interpolate_level + delta, 1.0), 0.0)
            if dst:
                self._mesh.load_dst_cage(dst, interpolate_level)
            level.text = f"Interpolate Level - {interpolate_level}"

        h = ogui.Horiz()
        btn = ogui.Button("Load")
        def _on_cancel():
            window.close_dialog()
        def _on_load(p: str):
            nonlocal dst
            window.close_dialog()
            p = Path(p)
            dst = trimesh.load_mesh(p)
            mod_level(1.0)
            btn.text = "Load(*)"
        def _on_btn_clicked():
            d = ogui.FileDialog(ogui.FileDialog.OPEN, "Select Result Cage File", window.theme)
            d.add_filter("", "All files")
            d.set_path(str(self._init_path.absolute()))
            d.set_on_cancel(_on_cancel)
            d.set_on_done(_on_load)
            window.show_dialog(d)
        btn.set_on_clicked(_on_btn_clicked)
        h.add_child(btn)

        add_btn(h, "+0.1", lambda: mod_level(0.1))
        add_btn(h, "-0.1", lambda: mod_level(-0.1))
        v.add_child(h)

        v.add_child(ogui.Label("Cage"))

        h = ogui.Horiz()
        btn = ogui.Button("Reset")
        btn.set_on_clicked(self._mesh.reset_dst_cage)
        h.add_child(btn)

        def _save_cage():
            v = self._mesh.dst_mesh
            save_path = str(f"dst_cage_{time.strftime('%Y%m%d-%H%M%S')}.ply")
            oio.write_triangle_mesh(save_path, v)
            log.info(f"Saved mesh to {save_path}")

        btn = ogui.Button("Save")
        btn.set_on_clicked(_save_cage)
        h.add_child(btn)

        v.add_child(h)

    def _build_ui_deform(self, v):

        cb = ogui.Checkbox("Auto Deform")

        def _on_dst_cage(c):
            if cb.checked:
                self._deform.on_deform()
        self._mesh.dst_cage.register(_on_dst_cage)
        v.add_child(cb)

        btn = ogui.Button("Deform")
        btn.set_on_clicked(self._deform.on_deform)
        v.add_child(btn)

        lbl = ogui.Label("Deform Time: N/A")
        v.add_child(lbl)
        lbl_fps = ogui.Label("Deform FPS: N/A")
        v.add_child(lbl_fps)

        def _on_deform_time(v: float):
            if v == 0.0:
                text = "Deform Time: N/A"
                fps_text = "Deform FPS: N/A"
            else:
                fps = 1.0 / v
                v_text = ">1000" if v >= 1.0 else f"{v * 1000:.2f}"
                text = f"Deform Time: {v_text}ms"
                fps_text = f"Deform FPS: {fps:.2f}"
            lbl.text = text
            lbl_fps.text = fps_text
        self._deform.time_sec.register(_on_deform_time)

    def _build_ui_view_control(self, v, window):
        v.add_child(ogui.Label("Views"))
        capture_dir = Path(".").absolute() / "captures"
        capture_dir.mkdir(parents=True, exist_ok=True)
        def _save_params():
            t = time.strftime("%Y%m%d-%H%M%S")
            camera = cvt_camera_o3d_to_3dgs(self._view.camera_param.val)
            camera_info = camera_to_camera_info(camera)
            torch.save(camera_info, capture_dir / f"{t}_camera_info.pt", pickle_module=dill)
            oio.write_image(str(capture_dir / f"{t}_ref_render.png"), self._view.current_render)
        def _on_cancel():
            window.close_dialog()
        def _on_load(p: str):
            window.close_dialog()
            p = torch.load(Path(p), pickle_module=dill)
            assert type(p) in {Camera, CameraInfo}, f"must be loading a Camera or CameraInfo object, got {type(p)} instead!"
            if isinstance(p, CameraInfo):
                p = cvt_camera_info_to_camera(p)
            extrin = cvt_camera_3dgs_to_o3d_extrin(p)
            self._view.set_view(extrin)
        def _load_params():
            d = ogui.FileDialog(ogui.FileDialog.OPEN, "Select Saved Parameter", window.theme)
            d.add_filter(".pt", "Saved Camera Parameter")
            d.set_path(str(capture_dir.absolute()))
            d.set_on_cancel(_on_cancel)
            d.set_on_done(_on_load)
            window.show_dialog(d)

        v.add_child(ogui.Label("Camera Params"))
        h = ogui.Horiz()
        add_btn(h, "Save", _save_params)
        add_btn(h, "Load", _load_params)
        v.add_child(h)

        v.add_child(ogui.Label("Set to Dataset Pose"))
        v.add_child(ogui.Label(f"train 0-{self._train_cnt-1}"))
        v.add_child(ogui.Label(f"eval {self._train_cnt}-{self._train_cnt+self._eval_cnt-1}"))
        num = ogui.NumberEdit(ogui.NumberEdit.INT)
        @num.set_on_value_changed
        def _on_value_changed(v):
            v = int(v)
            if v < 0:
                num.set_value(0)
                return
            if v >= len(self._cameras):
                num.set_value(len(self._cameras) - 1)
                return
            cam = self._cameras[v]
            extrin = cvt_camera_3dgs_to_o3d_extrin(cam)
            self._view.set_view(extrin)
        v.add_child(num)

    def _build_ui_display_control(self, v):
        v.add_child(ogui.Label("Display"))

        gs = ogui.Checkbox("Gaussian")
        gs.checked = True
        gs.set_on_checked(self._view.set_gs_visible)
        v.add_child(gs)

        src = ogui.Checkbox("Src Cage")
        src.checked = True
        src.set_on_checked(self._view.set_src_cage_visible)
        v.add_child(src)

        dst = ogui.Checkbox("Dst Cage")
        dst.checked = True
        dst.set_on_checked(self._view.set_dst_cage_visible)
        v.add_child(dst)

        vs = ogui.Checkbox("Vertices")
        vs.checked = True
        vs.set_on_checked(self._view.set_vertex_cloud_visible)
        v.add_child(vs)


class EditorViewer:

    def __init__(self, args: EditorArguments):
        self._args = args

        self._closing_event = asyncio.Event()

        self._model = load_model(args.model_path, iteration=args.iteration)
        self._cage: trimesh.Trimesh = trimesh.load_mesh(args.cage_path)
        self._cameras, self._train_cnt, self._eval_cnt = list_cameras_full(args.source_path)

        init_cam = self._cameras[args.init_cam_idx]
        self._deform_ctrl = ModelDeformController(args.algorithm, self._model, self._cage)
        init_anchors = {int(s) for s in args.init_anchors.split(",")} if args.init_anchors else set()
        self._mesh_ctrl = MeshEditController(self._cage, init_anchors=init_anchors)
        self._mesh_ctrl.dst_cage.register(self._deform_ctrl.on_dst_cage)
        self._cage_view = SceneCageView(init_cam, self._mesh_ctrl, self._deform_ctrl, args.white_bg, exit_event=self._closing_event)
        self._menu = ControlMenu(
            self._cage_view, self._mesh_ctrl, self._deform_ctrl,
            (self._cameras, self._train_cnt, self._eval_cnt), args.cage_path
        )

    def _build_ui(self, app):
        window = app.create_window("EditorViewer", width=512+160, height=512)
        cage_view = self._cage_view.build_ui(window)
        window.add_child(cage_view)
        menu = self._menu.build_ui(window)
        window.add_child(menu)

        def _on_layout(ctx):
            r = window.content_rect

            menu_width = 10 * ctx.theme.font_size
            view_width = (r.width - menu_width)
            view_height = max(int(view_width * 3/4), r.height)

            rframe = ogui.Rect(r.x, r.y, view_width, view_height)
            cage_view.frame = rframe

            mframe = ogui.Rect(rframe.x + rframe.width, r.y, menu_width, view_height)
            menu.frame = mframe
            menu.preferred_width = menu_width

        window.set_on_layout(_on_layout)

        def _on_close():
            self._closing_event.set()
            return True

        window.set_on_close(_on_close)

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
                await tg.start(self._deform_ctrl.run)
                task_status.started()
                await asyncio.Future()

        async with anyio.create_task_group() as tg:
            await tg.start(_app_event_loop)
            await tg.start(_main_window)
            await self._closing_event.wait()
            tg.cancel_scope.cancel()

async def main(args=None):
    import taichi as ti
    ti.init(ti.gpu)
    args = parse_args(EditorArguments, args)
    log.debug("running wih configuration: \n{}", OmegaConf.to_container(args))
    await EditorViewer(args).run()


if __name__ == '__main__':
    asyncio.run(main())