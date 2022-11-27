import weakref
from pathlib import Path

import torch
import numpy as np
import torch_scatter
from typing import Union
from enum import Enum
from torch_spsr.core._hash_backend import SparseFeatureHierarchy, compute_density


class VoxelStatus(Enum):
    """
    VoxelStatus.VS_NON_EXIST: This voxel shouldn't exist
    VoxelStatus.VS_EXIST_STOP: This voxel exists and is a leaf node
    VoxelStatus.VS_EXIST_CONTINUE: This voxel exists and has >0 children
    """
    VS_NON_EXIST = 0
    VS_EXIST_STOP = 1
    VS_EXIST_CONTINUE = 2


class HashTree:
    """
    This is the main data structure that stores the trees described in our paper.
    We build separate hierarchies for different network branches:
        - ENCODER: Encoder branch of U-Net
        - DECODER: Decoder branch of U-Net
        - DECODER_TMP: When performing structural prediction, we need a denser structure
    than DECODER after the up-sampling layer, so as to allow voxels to be pruned.
    """

    ENCODER = 0
    DECODER = 1
    DECODER_TMP = 2

    def __init__(self, xyz: torch.Tensor, voxel_size: float, depth: int):
        """
        Build an octree structure, and each layer is indexed as a sparse hash table.
            This only deal with a single instance (batch size = 1)
        :param xyz: [N, 3]
        :param voxel_size: size of the voxel to discrete points
        :param depth: octree depth. [0] is the bottom,
            [-1] is pseudo-root (highest level, may contain multiple nodes but at least one.)
        """
        self.xyz = xyz
        self.xyz_depth = None
        self.xyz_density = None
        self.device = self.xyz.device
        self.is_reflected = False

        range_kernel_func = weakref.WeakMethod(self.get_range_kernel)
        self._enc_branch = SparseFeatureHierarchy(depth, voxel_size, self.device, range_kernel_func)
        self._dec_branch = SparseFeatureHierarchy(depth, voxel_size, self.device, range_kernel_func)
        self._dec_tmp_branch = SparseFeatureHierarchy(depth, voxel_size, self.device, range_kernel_func)

    def reflect_decoder_coords(self):
        """
        Link 2 branches by reference (to reduce computation and memory if they reflect each other)
        """
        self._dec_tmp_branch = self._dec_branch = self._enc_branch
        self.is_reflected = True

    @property
    def voxel_size(self) -> float:
        """
        :return: width of the voxel at the finest level.
        """
        return self._enc_branch.voxel_size

    @property
    def depth(self) -> int:
        """
        :return: total depth of the tree
        """
        return self._enc_branch.depth

    def get_coords(self, branch: int, depth: int) -> torch.Tensor:
        """
        :return: the bottom-left-lower coordinates at depth.
        """
        return self._get_branch(branch).get_coords(depth)

    def get_stride(self, branch: int, depth: int) -> int:
        """
        :return: the stride at depth. Usually this will be 2**depth.
        """
        return self._get_branch(branch).get_stride(depth)

    def get_coords_size(self, branch: int, depth: int):
        """
        :return: get the size of coordinates. res == self.get_coords(branch, depth).size(0)
        """
        return self._get_branch(branch).get_num_voxels(depth)

    def update_coords(self, branch: int, depth: int, coords: Union[torch.Tensor, None]):
        """
        Update the coordinates of branch at depth.
        """
        return self._get_branch(branch).update_coords(depth, coords)

    # Cache the kernel, so this will not change across machines.
    KERNEL_CACHE_PATH = Path(__file__).parent.parent / "data" / "kernel.npz"
    cached_kernels = None

    def get_range_kernel(self, n_range):
        assert n_range % 2 == 1, "target_range must be odd."

        if HashTree.cached_kernels is None:
            data = np.load(HashTree.KERNEL_CACHE_PATH)
            HashTree.cached_kernels = data['kernel']

        kernel = torch.tensor(HashTree.cached_kernels.copy()[:n_range ** 3],
                              dtype=torch.int, device=self.device)
        return kernel

    def _get_branch(self, branch: int):
        if branch == self.ENCODER:
            return self._enc_branch
        elif branch == self.DECODER:
            return self._dec_branch
        elif branch == self.DECODER_TMP:
            return self._dec_tmp_branch
        else:
            raise NotImplementedError

    def evaluate_voxel_status(self, branch: int, coords: torch.Tensor, depth: int):
        """
        Evaluate status in the hierarchy, please refer to core.hashtree.VoxelStatus for numerical values:
            VoxelStatus.VS_NON_EXIST: This voxel shouldn't exist
            VoxelStatus.VS_EXIST_STOP: This voxel exists and is a leaf node
            VoxelStatus.VS_EXIST_CONTINUE: This voxel exists and has >0 children
        :param branch: int, the branch you want to query
        :param coords: (N, 3) torch.Tensor coordinates in the world space
        :param depth: int
        :return: (N, ) long torch.Tensor, indicating voxel status
        """
        return self._get_branch(branch).evaluate_voxel_status(coords, depth)

    def get_neighbours_data(self, source_coords: torch.Tensor, source_stride: int, target_depth: int,
                            nn_kernel: torch.Tensor, branch: int, conv_based: bool = False,
                            transposed: bool = False):
        """
        Please refer to _hash_backend.SparseFeatureHierarchy.get_coords_neighbours
        """
        return self._get_branch(branch).get_coords_neighbours(source_coords, source_stride, target_depth, nn_kernel,
                                                              conv_based, transposed)

    def get_neighbours(self, source_depth: int, target_depth: int, target_range: int, branch: int,
                       conv_based: bool = False):
        """
        Please refer to _hash_backend.SparseFeatureHierarchy.get_self_neighbours
        """
        return self._get_branch(branch).get_self_neighbours(source_depth, target_depth, target_range, conv_based)

    def evaluate_interpolated(self, query_pos: torch.Tensor, basis, basis_feat: Union[torch.Tensor, None],
                              feat_depth: int, feat: torch.Tensor, compute_mask: bool = False,
                              compute_grad: bool = False):
        dec_coords = self._dec_branch.get_coords(feat_depth)
        dec_stride = self._dec_branch.get_stride(feat_depth)

        # Evaluate interpolated value at any position if a basis function is applied.
        assert feat.size(0) == dec_coords.size(0), "Feature size not compatible."
        if basis_feat is not None:
            assert basis_feat.size(0) == dec_coords.size(0), "Feature size not compatible."

        # This minus half is because get_neighbours_data assumes top-left integer coordinates.
        xyz_pos = query_pos / self.voxel_size - 0.5

        base_ids, tgt_ids, tgt_offsets, _ = self.get_neighbours_data(
            xyz_pos, 1, feat_depth, self.get_range_kernel(basis.get_domain_range()), self.DECODER)

        query_coords = -tgt_offsets / dec_stride
        if compute_grad:
            query_val = basis.evaluate_derivative(feat=basis_feat, xyz=query_coords, feat_ids=tgt_ids,
                                                  stride=dec_stride)
            query_val = query_val * feat[tgt_ids, None]
        else:
            query_val = basis.evaluate(feat=basis_feat, xyz=query_coords, feat_ids=tgt_ids)
            query_val = query_val * feat[tgt_ids]

        # This automatically set non-supported chi to 0.
        query_val = torch_scatter.scatter_sum(query_val, base_ids, dim=0, dim_size=xyz_pos.size(0))

        if compute_mask:
            # Whether the query position is covered by any basis in this depth level.
            cover_mask = torch.zeros((query_val.size(0),), device=query_val.device, dtype=bool)
            cover_mask[base_ids] = True
            return query_val, cover_mask

        return query_val

    def get_random_samples(self, sample_per_voxel: int, depth, expand: int = 0):
        """
        Get a uniformly sampled grid (used for supervision).
        :param sample_per_voxel: P
        :param depth: octree-depth
        :param expand: size of expansion of the voxel!
        :return: (NxP, 3) in physical coordinates, where N is the size of base grid expanded by 'expand'
        """
        scale = self._dec_branch.get_stride(depth)
        base_coords = self._dec_branch.get_coords(depth, expand=expand)
        local_coords = torch.rand((base_coords.size(0), sample_per_voxel, 3), device=self.device) * scale
        query_pos = base_coords.unsqueeze(1) + local_coords.unsqueeze(0)
        query_pos = query_pos.view(-1, 3) * self.voxel_size
        return query_pos

    def get_test_grid(self, resolution, depth, expand: int = 0, conforming: bool = False):
        """
        Get a sparse-uniform grid point to evaluate functions over.
        :param resolution: the resolution within each grid, 0 means no sample within grid will be generated
        :param depth: the octree-depth for the basis grid
        :param expand: size of expansion.
        :param conforming: whether to force the tree to be conforming.
        :return:
            (NxRxRxR, 3) grid in physical coordinates, where N is the size of base grid expanded by 'expand'
            (N, 3) in div-voxel-size coordinates.
        """
        scale = self._dec_branch.get_stride(depth)
        base_coords = self._dec_branch.get_coords(depth, expand=expand, conforming=conforming)
        if resolution == 0:
            return None, base_coords
        box_coords = torch.linspace(0.0, scale, resolution, device=self.device)
        box_coords = torch.stack(torch.meshgrid(box_coords, box_coords, box_coords, indexing='ij'), dim=3)
        box_coords = box_coords.view(-1, 3)
        query_pos = base_coords.unsqueeze(1) + box_coords.unsqueeze(0)
        query_pos = query_pos.view(-1, 3) * self.voxel_size
        return query_pos, base_coords

    def get_voxel_centers(self, depth: int, branch: int = DECODER, normalized: bool = False):
        return self._get_branch(branch).get_voxel_centers(depth, normalized)

    def __repr__(self):
        stat = f"HashTree:\n"
        stat += self._enc_branch.__repr__()
        stat += self._dec_branch.__repr__()
        return stat

    def build_encoder_hierarchy_dense(self, expand_range: int = 0, density_depth: int = 2,
                                      uniform_density: bool = False):
        """
        Rebuild the tree structure, based on current xyz, voxel_size and depth.
        """
        if uniform_density:
            self.xyz_density = torch.ones((self.xyz.size(0),), device=self.device)
        else:
            enc_stride = self._enc_branch.get_stride(density_depth)
            self.xyz_density = compute_density(
                self.xyz, enc_stride * self.voxel_size) / (enc_stride ** 2)
        self._enc_branch.build_hierarchy_dense(self.xyz, expand_range=expand_range)
        self.xyz_depth = torch.zeros((self.xyz.size(0),), device=self.device, dtype=torch.int)

    def build_hierarchy_subdivide(self, subdivide_policy, expand: bool = False, density_depth: int = 2,
                                  limit_adaptive_depth: int = 100,
                                  **policy_kwargs):
        """
        Build a hierarchy, based on subdivision policy
        """
        enc_stride = self._enc_branch.get_stride(density_depth)
        self.xyz_density = compute_density(
            self.xyz, enc_stride * self.voxel_size) / (enc_stride ** 2)
        self.xyz_depth = self._enc_branch.build_hierarchy_subdivide(
            self.xyz, subdivide_policy, expand, limit_adaptive_depth, **policy_kwargs)

    def build_encoder_hierarchy_adaptive(self, density_depth: int = 2, log_base: float = 4.0, min_density: float = 8.0,
                                         uniform_density: bool = False, limit_adaptive_depth: int = 100):
        """
        Build the hierarchy by first determine the integer level of each point (based on xyz_density, log_base and
        min_density), then splat the points onto the tree structure.
        :param uniform_density: force uniform density
        :param density_depth: tree depth used to estimate point density.
        :param log_base: float
        :param min_density: float, minimum density in each voxel. If exceed, go to coarser level.
        :param limit_adaptive_depth: int. Maximum adaptive number of levels.
        :return torch.Tensor long. (N, ) level that the point lies in.
        """
        if uniform_density:
            # At least works for the default kwargs, which the users would never change.
            assert log_base == 4.0 and min_density == 8.0 and limit_adaptive_depth == 100
            self.xyz_density = torch.ones((self.xyz.size(0),), device=self.device) * 1e6
        else:
            enc_stride = self._enc_branch.get_stride(density_depth)
            self.xyz_density = compute_density(
                self.xyz, enc_stride * self.voxel_size) / (enc_stride ** 2)
        self.xyz_depth = self._enc_branch.build_hierarchy_adaptive(
            self.xyz, self.xyz_density, log_base, min_density, limit_adaptive_depth)

    def split_data(self, xyz: torch.Tensor, branch: int, data_depth: int, data: torch.Tensor):
        """
        Obtain the tri-linearly interpolated data located at xyz.
        :param branch: int
        :param xyz: torch.Tensor (N, 3)
        :param data_depth: int
        :param data: torch.Tensor (M, K), where K is feature dimension, and M = self.get_num_voxels(data_depth)
        :return: (N, K) torch.Tensor
        """
        return self._get_branch(branch).split_data(xyz, data_depth, data)

    def splat_data(self, xyz: torch.Tensor, branch: int, data_depth: int, data: torch.Tensor = None,
                   check_corr: bool = True, return_nf_mask: bool = False):
        """
        Splat data located at xyz to the tree voxels.
        :param branch: int
        :param xyz: torch.Tensor (N, 3)
        :param data_depth: int
        :param data: torch.Tensor (N, K)
        :param check_corr: if True, check if data is fully supported by its 8 neighbours
        :param return_nf_mask: Legacy, do not use.
        :return: (M, K) or (M,), where M = self.get_num_voxels(data_depth)
        """
        return self._get_branch(branch).splat_data(xyz, data_depth, data, check_corr, return_nf_mask)
