import jittor as jt
from typing import Tuple

__version__ = '1.0.0'
__version_info__ = (1, 0, 0)


def reconstruct(xyz: jt.Var, normal: jt.Var, voxel_size: float, depth: int = 4,
                min_density: float = 32.0, screen_alpha: float = 4.0) -> Tuple[jt.Var, jt.Var]:
    """
    A utility function, that reconstructs a triangle mesh given input points and normals.
    :param xyz: (N, 3) float32 cuda tensor
    :param normal: (N, 3) float32 cuda tensor
    :param voxel_size: float, size of the voxel at the finest level
    :param depth: int, total depth of the tree
    :param min_density: float, density for determine point layering
    :param screen_alpha: float, weight of the screening term
    :return: vertices (V, 3) jt.Var, triangles (T, 3) jt.Var
    """
    import torch

    from jspsr.core.hashtree import HashTree
    from jspsr.core.reconstructor import Reconstructor

    assert xyz.shape[0] == normal.shape[0] and xyz.shape[1] == normal.shape[1] == 3, \
        "Input positions and normals should have same size (N, 3)!"

    from jspsr.bases.bezier_tensor import BezierTensorBasis

    # Get a copy on pth
    xyz_pth = torch.from_numpy(xyz.numpy()).float().cuda()
    normal_pth = torch.from_numpy(normal.numpy()).float().cuda()

    # Build the tree based on given positions
    hash_tree = HashTree(xyz_pth, voxel_size=voxel_size, depth=depth)
    hash_tree.build_encoder_hierarchy_adaptive(min_density=min_density)
    hash_tree.reflect_decoder_coords()

    # Splat point normals onto the tree
    normal_data = {}
    sample_weight = 1.0 / hash_tree.xyz_density
    for d in range(hash_tree.depth):
        depth_mask = hash_tree.xyz_depth == d
        if not torch.any(depth_mask):
            continue
        normal_data_depth = hash_tree.splat_data(
            xyz_pth[depth_mask], hash_tree.DECODER, d, normal_pth[depth_mask] * sample_weight[depth_mask, None])
        normal_data[d] = normal_data_depth / (hash_tree.get_stride(hash_tree.DECODER, d) ** 3)

    # Perform reconstruction
    reconstructor = Reconstructor(hash_tree, BezierTensorBasis())
    reconstructor.sample_weight = sample_weight
    reconstructor.solve_multigrid(hash_tree.depth - 1, 0, normal_data,
                                  screen_alpha=screen_alpha, screen_xyz=xyz_pth, solver="pcg", verbose=False)

    # Mesh extraction (not differentiable for now)
    v, f = reconstructor.extract_multiscale_mesh(n_upsample=2, build_o3d_mesh=False)
    v = jt.array(v.detach().cpu().numpy())
    f = jt.array(f.detach().cpu().numpy())

    return v, f
