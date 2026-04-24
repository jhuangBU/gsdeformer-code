import numpy as np
import torch
import tqdm
import trimesh
from scipy.spatial import Delaunay
import open3d.geometry as ogeo
import open3d.utility as outil

from coords_util import mean_value_coordinates_3D


def _to_tensor(x, device):
    return torch.from_numpy(x.astype(np.float32)).clone().to(device)


def euclidean_to_mvc(src_mesh: trimesh.Trimesh, coords: torch.Tensor, batch_size=25_000, pbar=False) -> torch.Tensor:
    batch_size = batch_size if batch_size else len(coords)
    faces, vertices = _to_tensor(src_mesh.faces, coords.device), _to_tensor(src_mesh.vertices, coords.device)
    faces, vertices = faces.long()[None, ...], vertices[None, ...]

    with torch.no_grad():
        wjs = []
        it = range(0, len(coords), batch_size)
        it = tqdm.tqdm(it, desc="Deform - Euclidean -> Cage Coord") if pbar else it
        for i in it:
            query = coords[i:i + batch_size][None, ...]
            wj = mean_value_coordinates_3D(query, vertices, faces)
            wjs.append(wj)
        wjs = torch.cat(wjs, 1).to(coords.device)
        wjs = wjs[0]

    return wjs


def deform_mvc_to_euclidean(dst_mesh: trimesh.Trimesh, mvc_coords: torch.Tensor, eu_coords: torch.Tensor, batch_size=50_000, pbar=False) -> torch.Tensor:
    cage_verts = dst_mesh.vertices
    weights = mvc_coords

    # only deform those are in cage
    is_inside = weights.sum(-1) > 0.98  # (N,)
    weights = weights[is_inside]

    deformed = []
    it = range(0, len(weights), batch_size)
    it = tqdm.tqdm(it, desc="Deform - Cage Coord -> Euclidean") if pbar else it
    for i in it:
        w = weights[i:i + batch_size]
        d = torch.sum(w.unsqueeze(-1) * _to_tensor(cage_verts, mvc_coords.device).unsqueeze(0), dim=1)
        deformed.append(d)
    deformed = torch.cat(deformed, 0)

    # keep the rest as-is
    ret = eu_coords.clone()
    ret[is_inside] = deformed

    return ret, is_inside


def deform_mesh(mesh: ogeo.TriangleMesh, src_cage: trimesh.Trimesh):
    # calculate deforming mask
    cage_vertices = ogeo.PointCloud(outil.Vector3dVector(src_cage.vertices))
    cage_hull, _ = cage_vertices.compute_convex_hull()
    hull = Delaunay(np.asarray(cage_hull.vertices))
    mesh_vertices = np.asarray(mesh.vertices).copy()
    mesh_vertices_mask = hull.find_simplex(mesh_vertices) >= 0

    deform_vertices = mesh_vertices[mesh_vertices_mask]
    deform_vertices = torch.tensor(deform_vertices).cuda()
    mean_vals = euclidean_to_mvc(src_cage, deform_vertices)

    def _deform(dst_cage: trimesh.Trimesh) -> ogeo.TriangleMesh:
        deformed_vertices, _ = deform_mvc_to_euclidean(dst_cage, mean_vals, deform_vertices)

        ret_vertices = mesh_vertices.copy()
        ret_vertices[mesh_vertices_mask] = deformed_vertices.cpu().numpy()
        ret_vertices = ret_vertices.astype(np.float64)
        ret_mesh = ogeo.TriangleMesh(
            vertices=outil.Vector3dVector(ret_vertices),
            triangles=outil.Vector3iVector(np.asarray(mesh.triangles))
        )
        ret_mesh.vertex_colors=outil.Vector3dVector(np.asarray(mesh.vertex_colors))

        return ret_mesh

    return _deform
