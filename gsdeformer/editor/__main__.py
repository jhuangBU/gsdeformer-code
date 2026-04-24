import argparse
import asyncio
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal, Tuple, Set, NamedTuple, List

import anyio
import math
import numpy as np
import open3d.utility as outil
import open3d.visualization.gui as ogui
import open3d.visualization.rendering as ordr
import open3d.visualization as oviz
import open3d.geometry as ogeo
import open3d.io as oio
import open3d
import torch
import trimesh
from anyio import TASK_STATUS_IGNORED
from anyio.abc import TaskStatus
from loguru import logger as log
from omegaconf import MISSING, OmegaConf

# noinspection PyUnresolvedReferences
import gsdeformer.hack_add_gs3d_to_sys_path
from scene.dataset_readers import CameraInfo
from scene.gaussian_model import GaussianModel
from gaussian_renderer import render
from arguments import PipelineParams

from gsdeformer.deform.algorithm_ui import build_deformer
from gsdeformer.editor.utils_camera import cvt_camera_3dgs_to_o3d_intrin, cvt_camera_3dgs_to_o3d_extrin, \
    cvt_camera_o3d_to_3dgs
from gsdeformer.editor.utils_data import list_cameras, load_model
from gsdeformer.utils_config import parse_args
from gsdeformer.utils_ui import EventEmitter, ValueEmitter


@dataclass
class EditorArguments:
    algorithm: str = "all" # ablation switch, use full algorithm or "mean_only" algorithm
    cage_path: Path = MISSING # path to cage location

    init_cam_idx: int = 0 # init camera index
    source_path: Path = MISSING # path to dataset folder
    model_path: Path = MISSING # path to trained GS3D folder
    iteration: int = -1 # the iter of model to load, -1 means to pick the latest

    white_bg: bool = False # white background in rendering?


KEY_SHIFT = 256


WHITE_RGBA = np.array([1, 1, 1, 1], dtype=np.float32)
WHITE = np.array([1, 1, 1], dtype=np.float32)
BLACK = np.array([0, 0, 0], dtype=np.float32)
RED = np.array([1, 0, 0], dtype=np.float32)
BLUE = np.array([0, 0, 1], dtype=np.float32)


def _build_point_cloud(model_mean, color):
    cloud = ogeo.PointCloud()
    cloud.points = outil.Vector3dVector(model_mean)
    colors = np.repeat(color[np.newaxis, :], model_mean.shape[0], axis=0)
    cloud.colors = outil.Vector3dVector(colors)
    return cloud


def add_btn(l, name: str, lam: Callable[[], None]):
    btn = ogui.Button(name)
    btn.set_on_clicked(lam)
    l.add_child(btn)


class MeshEditController:

    class Translate(NamedTuple):
        axis: Literal["X", "Y", "Z"]
        val: float

    class Rotation(NamedTuple):
        axis: Literal["X", "Y", "Z"]
        radian: float

    def __init__(self, cage: trimesh.Trimesh):
        src_cage = ogeo.LineSet.create_from_triangle_mesh(cage.as_open3d)
        dst_cage = ogeo.LineSet(src_cage)
        src_cage = src_cage.paint_uniform_color(BLUE[:, np.newaxis])
        dst_cage = dst_cage.paint_uniform_color(RED[:, np.newaxis])
        vertex_cloud = _build_point_cloud(cage.vertices, BLUE)

        self.src_cage = ValueEmitter(src_cage)
        self.dst_cage = ValueEmitter(dst_cage)
        self.vertex_cloud = ValueEmitter(vertex_cloud)
        self._vertex_cloud_selected_idx = set()
        self._vertex_cloud_tree_cached = None

    def load_dst_cage(self, loaded: trimesh.Trimesh, level: float = 1.0):
        assert 0.0 <= level <= 1.0

        m = np.asarray(self.src_cage.val.points) * (1 - level) + loaded.vertices * level

        np.asarray(self.dst_cage.val.points)[:, :] = m
        self.dst_cage.notify_inplace_mutation()
        self.vertex_cloud.emit(_build_point_cloud(m, BLUE))
        self._vertex_cloud_selected_idx = set()
        self._vertex_cloud_tree_cached = None

    # standard select & move #

    def on_toggle_vertex(self, coord: np.array):
        # get nearest cage vertex to select
        if self._vertex_cloud_tree_cached is None:
            self._vertex_cloud_tree_cached = ogeo.KDTreeFlann(self.vertex_cloud.val)
        [k, idx, dist] = self._vertex_cloud_tree_cached.search_knn_vector_3d(coord, 1)

        idx = idx[0]
        dist = np.asarray(dist)[0]
        # TODO: perhaps introduces distance threshold?

        # apply selection
        if idx is not None:
            if idx not in self._vertex_cloud_selected_idx:
                color = RED
                self._vertex_cloud_selected_idx.add(idx)
            else:
                color = BLUE
                self._vertex_cloud_selected_idx.remove(idx)
            # apply color changes
            np.asarray(self.vertex_cloud.val.colors)[idx, :] = color
            self.vertex_cloud.notify_inplace_mutation()

    def on_translate(self, t: Translate):
        idxes = np.array(list(self._vertex_cloud_selected_idx))

        move = self._get_axis_vector(t.axis)
        move *= t.val

        np.asarray(self.dst_cage.val.points)[idxes] += move
        np.asarray(self.vertex_cloud.val.points)[idxes] += move
        self.dst_cage.notify_inplace_mutation()
        self.vertex_cloud.notify_inplace_mutation()
        self._vertex_cloud_tree_cached = None

    def on_rotation(self, r: Rotation):
        idxes = np.array(list(self._vertex_cloud_selected_idx))

        axis = self._get_axis_vector(r.axis)
        center = self.src_cage.val.get_center()

        rot = trimesh.transformations.rotation_matrix(r.radian, axis, center)
        rot = rot[:3, :3]

        modified = np.matmul(rot, np.asarray(self.dst_cage.val.points)[idxes].T).T

        np.asarray(self.dst_cage.val.points)[idxes] = modified
        np.asarray(self.vertex_cloud.val.points)[idxes] = modified
        self.dst_cage.notify_inplace_mutation()
        self.vertex_cloud.notify_inplace_mutation()
        self._vertex_cloud_tree_cached = None

    def _get_axis_vector(self, axis):
        move = np.array([0.0, 0.0, 0.0])
        if axis == "X":
            move[0] = 1.0
        elif axis == "Y":
            move[1] = 1.0
        elif axis == "Z":
            move[2] = 1.0
        else:
            assert False
        return move


class ModelDeformController:

    def __init__(self, args: EditorArguments, model: GaussianModel, src_cage: trimesh.Trimesh):
        self._ls = ogeo.LineSet.create_from_triangle_mesh(src_cage.as_open3d)
        self._dst_cage = src_cage.copy()

        self._deformer = build_deformer(args.algorithm, model, src_cage, pbar=True)

        self.model = ValueEmitter(model)
        self.mean_cloud = ValueEmitter(self._build_mean_cloud(model))

    def _build_mean_cloud(self, model: GaussianModel):
        mean_cloud = model.get_xyz.detach().cpu().numpy()[::10]
        mean_cloud = _build_point_cloud(mean_cloud, BLACK)
        return mean_cloud

    def on_dst_cage(self, ls: ogeo.LineSet):
        self._ls = ls

    def on_deform(self):
        self._dst_cage.vertices[:,:] = np.asarray(self._ls.points)
        model = self._deformer(self._dst_cage)
        self.model.emit(model)
        self.mean_cloud.emit(self._build_mean_cloud(self.model.val))


class CageView:

    def __init__(self, model: GaussianModel, init_cam: CameraInfo, ctrl: MeshEditController, deform: ModelDeformController):
        self._model = model
        self._init_cam = init_cam
        self._init_cam_intrin = cvt_camera_3dgs_to_o3d_intrin(self._init_cam)
        self._init_cam_extrin = cvt_camera_3dgs_to_o3d_extrin(self._init_cam)
        self._ctrl = ctrl
        self._deform = deform

        # create UI
        self._view = ogui.SceneWidget()

        # signals
        self.camera_param = ValueEmitter((self._init_cam_intrin, self._init_cam_extrin))

    def build_ui(self, window):
        cage_view = self._view
        cage_view.scene = ordr.Open3DScene(window.renderer)

        # add cages to view
        ln_mat = self._build_lineset_material()

        src_cage_name = "Src Cage"
        cage_view.scene.add_geometry(src_cage_name, self._ctrl.src_cage.val, ln_mat)
        def on_src_updated(e: ogeo.LineSet):
            cage_view.scene.remove_geometry(src_cage_name)
            cage_view.scene.add_geometry(src_cage_name, e, ln_mat)
        self._ctrl.src_cage.register(on_src_updated)

        dst_cage_name = "Dst Cage"
        cage_view.scene.add_geometry(dst_cage_name, self._ctrl.dst_cage.val, ln_mat)
        def on_dst_updated(e: ogeo.LineSet):
            cage_view.scene.remove_geometry(dst_cage_name)
            cage_view.scene.add_geometry(dst_cage_name, e, ln_mat)
        self._ctrl.dst_cage.register(on_dst_updated)

        # add vertex cloud to view
        vc_mat = self._build_point_cloud_material(10, window)
        vertex_cloud_name = "Vertex Cloud"
        vertex_cloud_visible = False
        cage_view.scene.add_geometry(vertex_cloud_name, self._ctrl.vertex_cloud.val, vc_mat)
        cage_view.scene.show_geometry(vertex_cloud_name, vertex_cloud_visible)
        def on_vertex_cloud_updated(e: ogeo.PointCloud):
            cage_view.scene.remove_geometry(vertex_cloud_name)
            cage_view.scene.add_geometry(vertex_cloud_name, e, vc_mat)
            cage_view.scene.show_geometry(vertex_cloud_name, vertex_cloud_visible)
        self._ctrl.vertex_cloud.register(on_vertex_cloud_updated)

        def _on_key(ke: ogui.KeyEvent):
            nonlocal vertex_cloud_visible
            is_shift = ke.key == KEY_SHIFT
            if is_shift and ke.type == ogui.KeyEvent.DOWN:
                vertex_cloud_visible = True
                cage_view.scene.show_geometry(vertex_cloud_name, vertex_cloud_visible)
            elif is_shift and ke.type == ogui.KeyEvent.UP:
                vertex_cloud_visible = False
                cage_view.scene.show_geometry(vertex_cloud_name, vertex_cloud_visible)
            return ogui.Widget.HANDLED

        cage_view.set_on_key(_on_key)

        accepted_events = {ogui.MouseEvent.DRAG, ogui.MouseEvent.WHEEL}

        def _on_mouse(me: ogui.MouseEvent):
            if me.type in accepted_events:
                ex = cage_view.scene.camera.get_model_matrix()
                ex = np.linalg.inv(ex)  # w2c to c2w
                ex[1:3, :] *= -1  # OpenGL to OpenCV
                self.camera_param.emit((self._init_cam_intrin, ex))
                return ogui.Widget.EventCallbackResult.HANDLED

            is_down = me.type == ogui.MouseEvent.Type.BUTTON_DOWN
            is_shift_pressed = me.is_modifier_down(ogui.KeyModifier.SHIFT)
            if is_down and is_shift_pressed:

                def depth_callback(depth_image):
                    # get world coordinate of selection
                    view_frame = cage_view.frame
                    x = me.x - view_frame.x
                    y = me.y - view_frame.y
                    depth = np.asarray(depth_image)[y, x]

                    if depth == 1.0:
                        coord = None
                    else:
                        world = cage_view.scene.camera.unproject(x, y, depth, view_frame.width, view_frame.height)
                        coord = np.array([world[0], world[1], world[2]])

                    if coord is not None:
                        ogui.Application.instance.post_to_main_thread(window, lambda: self._ctrl.on_toggle_vertex(coord))

                cage_view.scene.scene.render_to_depth_image(depth_callback)

                return ogui.Widget.EventCallbackResult.HANDLED

            return ogui.Widget.EventCallbackResult.IGNORED

        cage_view.set_on_mouse(_on_mouse)

        # add means to view
        # mat = self._build_point_cloud_material(5, window)
        # mean_cloud_name = "Mean Cloud"
        # cage_view.scene.add_geometry(mean_cloud_name, self._deform.mean_cloud.val, mat)
        # def on_mean_cloud_updated(e: ogeo.PointCloud):
        #     cage_view.scene.remove_geometry(mean_cloud_name)
        #     cage_view.scene.add_geometry(mean_cloud_name, e, mat)
        # self._deform.mean_cloud.register(on_mean_cloud_updated)

        # position the cameras
        self.reset_view()

        return cage_view

    def reset_view(self):
        self.set_view(self._init_cam_extrin)

    def set_view(self, extrin):
        bounds = self._view.scene.bounding_box
        self._view.setup_camera(self._init_cam_intrin, extrin, bounds)
        self.camera_param.emit((self._init_cam_intrin, extrin))

    def _build_lineset_material(self):
        ln_mat = oviz.rendering.MaterialRecord()
        ln_mat.shader = "unlitLine"
        ln_mat.line_width = 1
        return ln_mat

    def _build_point_cloud_material(self, factor_scaling, window):
        mat = ordr.MaterialRecord()
        mat.shader = "defaultUnlit"
        # Point size is in native pixels, but "pixel" means different things to
        # different platforms (macOS, in particular), so multiply by Window scale
        # factor.
        mat.point_size = factor_scaling * window.scaling
        return mat

    async def run(self, task_status: TaskStatus[None] = TASK_STATUS_IGNORED):
        task_status.started()
        await asyncio.Future()


def extract_extrinsic(scene):
    ex = scene.camera.get_model_matrix()
    ex = np.linalg.inv(ex)  # w2c to c2w
    ex[1:3, :] *= -1  # OpenGL to OpenCV
    return ex


class RenderView:

    def __init__(self, init_cam: CameraInfo, deform: ModelDeformController, white_bg=False):
        self._init_cam = init_cam
        self._init_cam_intrin = cvt_camera_3dgs_to_o3d_intrin(self._init_cam)
        self._init_cam_extrin = cvt_camera_3dgs_to_o3d_extrin(self._init_cam)
        self._deform = deform
        # create UI
        self._view = ogui.SceneWidget()
        # render params
        self._pipe = PipelineParams(argparse.ArgumentParser())  # mock pipeline params
        self._bg_color = torch.tensor(WHITE, dtype=torch.float32, device="cuda") if white_bg else torch.tensor(BLACK, dtype=torch.float32, device="cuda")
        # signals
        self.camera_param = ValueEmitter((self._init_cam_intrin, self._init_cam_extrin))
        # members
        self.current_render = None

    def build_ui(self, window):
        view = self._view
        view.scene = ordr.Open3DScene(window.renderer)

        self.reset_view()

        accepted_events = {ogui.MouseEvent.DRAG, ogui.MouseEvent.WHEEL}

        def _on_mouse(e):
            if e.type in accepted_events:
                intr, ex = self._get_view_intrin_extrin()
                self.camera_param.emit((intr, ex))
                self._update_render()
                return ogui.SceneWidget.EventCallbackResult.HANDLED
            return ogui.Widget.EventCallbackResult.IGNORED

        view.set_on_mouse(_on_mouse)

        self._deform.model.register(lambda e: self._update_render())

        return view

    def _get_view_intrin_extrin(self):
        view = self._view
        intr = cvt_camera_3dgs_to_o3d_intrin(self._init_cam, height=view.frame.height, width=view.frame.width)
        ex = extract_extrinsic(view.scene)
        return intr, ex

    @torch.no_grad()
    def _update_render(self):
        view = self._view
        cam = cvt_camera_o3d_to_3dgs(self.camera_param.val)

        m = self._deform.model.val
        new_cov = getattr(m, "precomp_cov", None)
        image = render(cam, m, self._pipe, self._bg_color, cov3d=new_cov)["render"]
        image = torch.clamp(image, 0.0, 1.0)
        image = (image * 255).to(torch.uint8).permute(1, 2, 0)
        image = image.cpu().detach().numpy()
        image = np.ascontiguousarray(image)
        image = ogeo.Image(image)
        self.current_render = image

        view.scene.set_background(WHITE_RGBA[:, np.newaxis], image)

    def reset_view(self):
        self.set_view(self._init_cam_extrin)

    def set_view(self, extrin):
        bounds = self._view.scene.bounding_box
        intrin, _ = self._get_view_intrin_extrin()
        self._view.setup_camera(intrin, extrin, bounds)
        self.camera_param.emit((intrin, extrin))
        self._update_render()

    async def run(self, task_status: TaskStatus[None] = TASK_STATUS_IGNORED):
        task_status.started()
        await asyncio.Future()


def picklize_camera_param(camera_param):
    intrin, extrin = camera_param
    intrin = (intrin.width, intrin.height, intrin.intrinsic_matrix)
    camera_param = (intrin, extrin)
    return camera_param


def unpicklize_camera_param(camera_param):
    intrin, extrin = camera_param
    width, height, intrinsic_matrix = intrin
    intrin = open3d.camera.PinholeCameraIntrinsic(width, height, intrinsic_matrix)
    return (intrin, extrin)


class ControlMenu:

    def __init__(
            self,
            render: RenderView, cage: CageView, mesh_ctrl: MeshEditController, deform: ModelDeformController,
            cameras: List[CameraInfo], init_path: Path
    ):
        self._render = render
        self._cage = cage
        self._mesh_ctrl = mesh_ctrl
        self._deform = deform
        self._cameras = cameras
        self._init_path = init_path

    def build_ui(self, window):
        v = ogui.Margins(left=5, right=5)
        v = ogui.Vert(margins=v)

        v.add_child(ogui.Label("Operations"))

        # self._build_ui_translate(v)

        # self._build_ui_rotate(v)

        self._build_ui_interpolate(v, window)

        self._build_ui_deform(v)

        self._build_ui_reset_view(v, window)

        return v

    def _build_ui_interpolate(self, v, window):
        interpolate_level = 0.0
        dst = None

        v.add_child(ogui.Label("Interpolate Cage"))
        level = ogui.Label(f"Interpolate Level - {interpolate_level}")
        v.add_child(level)

        def mod_level(delta):
            nonlocal interpolate_level, dst
            interpolate_level = max(min(interpolate_level + delta, 1.0), 0.0)
            if dst:
                self._mesh_ctrl.load_dst_cage(dst, interpolate_level)
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

    def _build_ui_deform(self, v):

        cb = ogui.Checkbox("Auto Deform")

        def _on_dst_cage(c):
            if cb.checked:
                self._deform.on_deform()
        self._mesh_ctrl.dst_cage.register(_on_dst_cage)
        v.add_child(cb)

        btn = ogui.Button("Deform")

        def _on_deform():
            btn.text = "Deform (Processing)"
            self._deform.on_deform()
            btn.text = "Deform (Done)"

        btn.set_on_clicked(_on_deform)
        v.add_child(btn)

    def _build_ui_translate(self, v):
        step = 0.1
        def action(axis, sign): self._mesh_ctrl.on_translate(MeshEditController.Translate(axis, sign * step))

        v.add_child(ogui.Label("Translate Cage"))
        h = ogui.Horiz()
        add_btn(h, "+X", lambda: action("X", 1.0))
        add_btn(h, "-X", lambda: action("X", -1.0))
        v.add_child(h)
        h = ogui.Horiz()
        add_btn(h, "+Y", lambda: action("Y", 1.0))
        add_btn(h, "-Y", lambda: action("Y", -1.0))
        v.add_child(h)
        h = ogui.Horiz()
        add_btn(h, "+Z", lambda: action("Z", 1.0))
        add_btn(h, "-Z", lambda: action("Z", -1.0))
        v.add_child(h)

    def _build_ui_rotate(self, v):
        step = 10 * (math.pi / 180)
        def action(axis, sign): self._mesh_ctrl.on_rotation(MeshEditController.Rotation(axis, sign * step))

        v.add_child(ogui.Label("Rotate Cage"))
        h = ogui.Horiz()
        add_btn(h, "+X", lambda: action("X", 1.0))
        add_btn(h, "-X", lambda: action("X", -1.0))
        v.add_child(h)
        h = ogui.Horiz()
        add_btn(h, "+Y", lambda: action("Y", 1.0))
        add_btn(h, "-Y", lambda: action("Y", -1.0))
        v.add_child(h)
        h = ogui.Horiz()
        add_btn(h, "+Z", lambda: action("Z", 1.0))
        add_btn(h, "-Z", lambda: action("Z", -1.0))
        v.add_child(h)

    def _build_ui_reset_view(self, v, window):
        v.add_child(ogui.Label("Views"))
        capture_dir = Path(".").absolute() / "captures"
        capture_dir.mkdir(parents=True, exist_ok=True)
        def _save_params():
            t = int(time.time())
            camera_param = picklize_camera_param(self._render.camera_param.val)
            torch.save(camera_param, capture_dir / f"{t}_camera_params.pt")
            oio.write_image(str(capture_dir / f"{t}_ref_render.png"), self._render.current_render)
        def _on_cancel():
            window.close_dialog()
        def _on_load(p: str):
            window.close_dialog()
            p = torch.load(Path(p))
            p = unpicklize_camera_param(p)
            intrin, extrin = p
            self._render.set_view(extrin)
            self._cage.set_view(extrin)
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

        cb = ogui.Checkbox("Auto Sync Views")
        def _sync_extrin(t):
            _, extrin = t
            if cb.checked:
                _, cextrin = self._cage.camera_param.val
                if (cextrin != extrin).any():
                    self._cage.set_view(extrin)
                _, rextrin = self._render.camera_param.val
                if (rextrin != extrin).any():
                    self._render.set_view(extrin)
        self._render.camera_param.register(_sync_extrin)
        self._cage.camera_param.register(_sync_extrin)
        v.add_child(cb)

        v.add_child(ogui.Label("Set to Dataset Pose"))
        num = ogui.NumberEdit(ogui.NumberEdit.INT)
        num.set_limits(0, len(self._cameras)-1)
        v.add_child(num)
        h = ogui.Horiz()
        def _set_view(view):
            cam = self._cameras[num.int_value]
            extrin = cvt_camera_3dgs_to_o3d_extrin(cam)
            view.set_view(extrin)
        add_btn(h, "Render", lambda: _set_view(self._render))
        add_btn(h, "Cage", lambda: _set_view(self._cage))
        v.add_child(h)


class EditorViewer:

    def __init__(self, args: EditorArguments):
        self._args = args

        self._model = load_model(args.model_path, iteration=args.iteration)
        self._cage: trimesh.Trimesh = trimesh.load_mesh(args.cage_path)
        self._cameras = list_cameras(args.source_path)

        init_cam = self._cameras[args.init_cam_idx]
        self._deform_ctrl = ModelDeformController(args, self._model, self._cage)
        self._mesh_ctrl = MeshEditController(self._cage)
        self._mesh_ctrl.dst_cage.register(self._deform_ctrl.on_dst_cage)
        self._cage_view = CageView(self._model, init_cam, self._mesh_ctrl, self._deform_ctrl)
        self._render_view = RenderView(init_cam, self._deform_ctrl, args.white_bg)
        self._menu = ControlMenu(
            self._render_view, self._cage_view, self._mesh_ctrl, self._deform_ctrl,
            self._cameras, args.cage_path
        )

    def _build_ui(self, app):
        window = app.create_window("EditorViewer", width=512*2+160, height=512)
        cage_view = self._cage_view.build_ui(window)
        window.add_child(cage_view)
        render_view = self._render_view.build_ui(window)
        window.add_child(render_view)
        menu = self._menu.build_ui(window)
        window.add_child(menu)

        def _on_layout(ctx):
            r = window.content_rect

            menu_width = 10 * ctx.theme.font_size
            view_width = (r.width - menu_width) // 2
            view_height = max(int(view_width * 3/4), r.height)

            rframe = ogui.Rect(r.x, r.y, view_width, view_height)
            render_view.frame = rframe

            lframe = ogui.Rect(r.x + rframe.width, r.y, view_width, view_height)
            cage_view.frame = lframe

            mframe = ogui.Rect(lframe.x + lframe.width, r.y, menu_width, view_height)
            menu.frame = mframe
            menu.preferred_width = menu_width

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
                await tg.start(self._render_view.run)
                task_status.started()
                await asyncio.Future()

        async with anyio.create_task_group() as tg:
            await tg.start(_app_event_loop)
            await tg.start(_main_window)


async def main(args=None):
    import taichi as ti
    ti.init(ti.gpu)
    args = parse_args(EditorArguments, args)
    log.debug("running wih configuration: \n{}", OmegaConf.to_container(args))
    await EditorViewer(args).run()


if __name__ == '__main__':
    asyncio.run(main())