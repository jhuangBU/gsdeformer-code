import argparse
import torch
import trimesh
import numpy as np
from coords_util import mean_value_coordinates_3D
import tqdm
from typing import Tuple

def _to_tensor(x, device):
    return torch.from_numpy(x.astype(np.float32)).clone().to(device)


def euclidean_to_mvc(src_mesh: trimesh.Trimesh, coords: torch.Tensor, batch_size=25_000, pbar=False) -> Tuple[torch.Tensor, torch.Tensor]:
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


    # only deform those are in cage
    is_inside = wjs.sum(-1) > 0.98  # (N,)
    wjs = wjs[is_inside]

    return wjs, is_inside


def deform_mvc_to_euclidean(dst_mesh: trimesh.Trimesh, mvc_coords: torch.Tensor, batch_size=50_000, pbar=False) -> torch.Tensor:
    cage_verts = dst_mesh.vertices
    cvt = _to_tensor(cage_verts, mvc_coords.device).unsqueeze(0)
    weights = mvc_coords

    deformed = []
    it = range(0, len(weights), batch_size)
    it = tqdm.tqdm(it, desc="Deform - Cage Coord -> Euclidean") if pbar else it
    for i in it:
        w = weights[i:i + batch_size]
        d = torch.sum(w.unsqueeze(-1) * cvt, dim=1)
        deformed.append(d)
    deformed = torch.cat(deformed, 0)

    return deformed

def load_tet_mesh(filename):
    with open(filename, 'r') as f:
        n_verts, n_tets = map(int, f.readline().split())
        vertices = np.array([list(map(float, f.readline().split())) for _ in range(n_verts)])
        tets = np.array([list(map(int, f.readline().split())) for _ in range(n_tets)])
    return vertices, tets

def save_tet_mesh(filename, vertices, tets):
    with open(filename, 'w') as f:
        f.write(f"{len(vertices)} {len(tets)}\n")
        for v in vertices:
            f.write(f"{v[0]} {v[1]} {v[2]}\n")
        for t in tets:
            f.write(f"{t[0]} {t[1]} {t[2]} {t[3]}\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src-cage', required=True, help='Source cage mesh file')
    parser.add_argument('--dst-cage', required=True, help='Target cage mesh file')
    parser.add_argument('--tet-mesh', required=True, help='Input tet mesh file')
    parser.add_argument('--output', required=True, help='Output tet mesh file')
    parser.add_argument('--batch-size', type=int, default=25000, help='Batch size for MVC computation')
    args = parser.parse_args()

    # Load meshes
    src_cage = trimesh.load(args.src_cage)
    dst_cage = trimesh.load(args.dst_cage)
    vertices, tets = load_tet_mesh(args.tet_mesh)

    # Convert vertices to tensor
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vertices_tensor = torch.from_numpy(vertices).float().to(device)

    # Compute MVC coordinates
    mvc_coords, is_inside = euclidean_to_mvc(src_cage, vertices_tensor, batch_size=args.batch_size, pbar=True)

    # Deform vertices using MVC
    deformed_vertices = deform_mvc_to_euclidean(dst_cage, mvc_coords, batch_size=args.batch_size, pbar=True)

    # Update vertices with deformed positions
    new_vertices = vertices.copy()
    inside_indices = torch.where(is_inside)[0].cpu().numpy()
    new_vertices[inside_indices] = deformed_vertices.cpu().numpy()

    # Save original and deformed vertices as point clouds
    vertices_pcd = trimesh.PointCloud(vertices)
    vertices_pcd.export(args.output.replace(".txt", '.original.ply'))

    deformed_pcd = trimesh.PointCloud(new_vertices)
    deformed_pcd.export(args.output.replace(".txt", '.deformed.ply'))

    # Save deformed tet mesh
    save_tet_mesh(args.output, new_vertices, tets)

if __name__ == '__main__':
    main()
