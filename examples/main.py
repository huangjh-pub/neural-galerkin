import torch
import jspsr
import open3d as o3d
import numpy as np
from pathlib import Path


def to_o3d_pcd(xyz: torch.Tensor, normal: torch.Tensor):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz.detach().cpu().numpy())
    pcd.normals = o3d.utility.Vector3dVector(normal.detach().cpu().numpy())
    return pcd


def to_o3d_mesh(vertices: torch.Tensor, triangles: torch.Tensor):
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices.detach().cpu().numpy())
    mesh.triangles = o3d.utility.Vector3iVector(triangles.detach().cpu().numpy())
    mesh.compute_vertex_normals()
    return mesh


if __name__ == '__main__':
    horse_npts = np.fromfile(Path(__file__).parent / "horse.bnpts").reshape(-1, 6)
    horse_xyz, horse_normal = horse_npts[:, :3], horse_npts[:, 3:]
    horse_xyz = torch.from_numpy(horse_xyz).cuda().float()
    horse_normal = torch.from_numpy(horse_normal).cuda().float()

    print(f"Start computing the mesh. #pts = {horse_xyz.shape[0]}")
    v, f = jspsr.reconstruct(horse_xyz, horse_normal, depth=4, voxel_size=0.002, screen_alpha=32.0)
    print("Done!")
    o3d.visualization.draw_geometries([to_o3d_pcd(horse_xyz, horse_normal), to_o3d_mesh(v, f)])
