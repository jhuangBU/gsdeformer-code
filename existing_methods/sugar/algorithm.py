import numpy as np
import torch
import tqdm
import trimesh

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