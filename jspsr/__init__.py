import torch
from typing import Tuple

from jspsr.core.hashtree import HashTree
from jspsr.core.reconstructor import Reconstructor

__version__ = '1.0.0'
__version_info__ = (1, 0, 0)


def reconstruct(xyz: torch.Tensor, normal: torch.Tensor, voxel_size: float, depth: int = 4,
                min_density: float = 32.0, screen_alpha: float = 4.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    A utility function, that reconstructs a triangle mesh given input points and normals.
    :param xyz: (N, 3) float32 cuda tensor
    :param normal: (N, 3) float32 cuda tensor
    :param voxel_size: float, size of the voxel at the finest level
    :param depth: int, total depth of the tree
    :param min_density: float, density for determine point layering
    :param screen_alpha: float, weight of the screening term
    :return: vertices (V, 3) torch.Tensor, triangles (T, 3) torch.Tensor
    """
    assert xyz.size(0) == normal.size(0) and xyz.size(1) == normal.size(1) == 3, \
        "Input positions and normals should have same size (N, 3)!"
    assert xyz.is_cuda and normal.is_cuda, "Input should be on the device!"
    assert xyz.dtype == normal.dtype == torch.float32, "Input should have dtype == float32"

    from jspsr.bases.bezier_tensor import BezierTensorBasis

    # Build the tree based on given positions
    hash_tree = HashTree(xyz, voxel_size=voxel_size, depth=depth)
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
            xyz[depth_mask], hash_tree.DECODER, d, normal[depth_mask] * sample_weight[depth_mask, None])
        normal_data[d] = normal_data_depth / (hash_tree.get_stride(hash_tree.DECODER, d) ** 3)

    # Perform reconstruction
    reconstructor = Reconstructor(hash_tree, BezierTensorBasis())
    reconstructor.sample_weight = sample_weight
    reconstructor.solve_multigrid(hash_tree.depth - 1, 0, normal_data,
                                  screen_alpha=screen_alpha, screen_xyz=xyz, solver="pcg", verbose=False)

    # Mesh extraction (not differentiable for now)
    final_mesh = reconstructor.extract_multiscale_mesh(n_upsample=2, build_o3d_mesh=False)

    return final_mesh
