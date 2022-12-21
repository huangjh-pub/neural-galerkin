import open3d as o3d

import torch    # For libraries to work
import jittor as jt

import jspsr
import numpy as np
from pathlib import Path


def to_o3d_pcd(xyz: jt.Var, normal: jt.Var):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz.numpy())
    pcd.normals = o3d.utility.Vector3dVector(normal.numpy())
    return pcd


def to_o3d_mesh(vertices: jt.Var, triangles: jt.Var):
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices.numpy())
    mesh.triangles = o3d.utility.Vector3iVector(triangles.numpy())
    mesh.compute_vertex_normals()
    return mesh


if __name__ == '__main__':
    horse_npts = np.fromfile(Path(__file__).parent / "horse.bnpts").reshape(-1, 6)
    horse_xyz, horse_normal = horse_npts[:, :3], horse_npts[:, 3:]
    horse_xyz = jt.array(horse_xyz)
    horse_normal = jt.array(horse_normal)

    print(f"Start computing the mesh. #pts = {horse_xyz.shape[0]}")
    v, f = jspsr.reconstruct(horse_xyz, horse_normal, depth=4, voxel_size=0.002, screen_alpha=32.0)
    print("Done!")
    o3d.visualization.draw_geometries([to_o3d_pcd(horse_xyz, horse_normal), to_o3d_mesh(v, f)])
