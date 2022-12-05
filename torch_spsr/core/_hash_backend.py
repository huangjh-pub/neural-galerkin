from typing import List, Union

import numpy as np
import torch
import torch_scatter

from torch_spsr.ext import CuckooHashTable
from .ops import torch_unique
from torch_spsr.bases.bezier_tensor import BezierTensorBasis


def compute_density(xyz: torch.Tensor, voxel_size: float) -> torch.Tensor:
    """
    Kernel density estimation using Bezier Tensor.
    :param xyz: (N, 3) input point cloud.
    :param voxel_size: float, Smoothing kernel size.
    :return: (N, ) density. Unit is roughly proportional to pts/voxel.
    """
    from torch_spsr.core.hashtree import HashTree

    hash_tree = HashTree(xyz, voxel_size, 1)
    hash_tree.build_encoder_hierarchy_dense(expand_range=2, uniform_density=True)
    wd = hash_tree.splat_data(xyz, hash_tree.ENCODER, 0, check_corr=False)
    hash_tree.reflect_decoder_coords()
    wd = hash_tree.evaluate_interpolated(xyz, BezierTensorBasis(), None, 0, wd)

    return wd


class NeighbourMaps:
    """
    A cache similar to sparseConv kernel map, but without the need of re-computing
    everything when enlarging neighbourhoods.
    """

    def __init__(self, device):
        # Note: none of the relevant range here is multiplied by strides
        self.cache = {}
        self.device = device

    def get_map(self, source_depth: int, target_depth: int, target_range: int, force_recompute: bool = False):
        """
        Given the query, return the existing part and also the part needed to be queried.
        :return: tuple (src-id, tgt-id, neighbour-types, nbsizes, ranges lacked [a,b] )
        """
        if (source_depth, target_depth) in self.cache.keys():
            if force_recompute:
                del self.cache[(source_depth, target_depth)]
                max_range, exist_src, exist_tgt, exist_nt, exist_nbs = -1, None, None, None, None
            else:
                max_range, exist_src, exist_tgt, exist_nt, exist_nbs = self.cache[(source_depth, target_depth)]
        else:
            max_range, exist_src, exist_tgt, exist_nt, exist_nbs = -1, None, None, None, None

        if target_range == max_range:
            return exist_src, exist_tgt, exist_nt, exist_nbs, None
        elif target_range < max_range:
            tr3 = target_range * target_range * target_range
            n_query = torch.sum(exist_nbs[:tr3])
            return exist_src[:n_query], exist_tgt[:n_query], exist_nt[:n_query], exist_nbs[:tr3], None
        else:
            return exist_src, exist_tgt, exist_nt, exist_nbs, [max_range + 2, target_range]

    def update_map(self, source_depth: int, target_depth: int, target_range: int, res: list):
        self.cache[(source_depth, target_depth)] = [target_range] + res


class SparseFeatureHierarchy:
    """
    An indexing structure, containing multiple layers within a tree.
    """

    CONFORM_OFFSETS = [(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1),
                       (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)]

    def __init__(self, depth: int, voxel_size: float, device, range_kernel):
        """
        Initialize the metadata of the hierarchy.
        :param depth: int, number of layers
        :param voxel_size: float, width of the voxel at the finest level.
        :param device: torch.Device, device where the data structure should reside
        :param range_kernel: a helper function that specifies the relative offsets in the kernel.
            *Note*: The sequence in the kernel only has to be respected when conv=True in get_self_neighbours!
        """
        self._depth = depth
        self._voxel_size = voxel_size
        self._device = device
        self._range_kernel = range_kernel

        # Conv-based include same-level
        self._conv_nmap = NeighbourMaps(self._device)
        # Region-based exclude same-level
        self._region_nmap = NeighbourMaps(self._device)

        self._strides = [2 ** d for d in range(self.depth)]
        # List of torch.Tensor (Nx3)
        self._coords = [None for d in range(self.depth)]
        self._hash_table: List[CuckooHashTable] = [None for d in range(self.depth)]

    @property
    def depth(self) -> int:
        """
        :return: total depth of the tree
        """
        return self._depth

    @property
    def voxel_size(self) -> float:
        """
        :return: width of the voxel at the finest level.
        """
        return self._voxel_size

    def get_stride(self, depth: int) -> int:
        """
        :return: the stride at depth. Usually this would be 2**depth
        """
        return self._strides[depth]

    def get_coords(self, depth: int, expand: int = 0, conforming: bool = False) -> torch.Tensor:
        """
        :param depth:
        :param expand:
        :param conforming:
        :return: (N, 3) float32 torch.Tensor, each row is the bottom-left-near voxel corner coordinate in normalized space.
            Note: This might be called multiple times within a function, so we could possibly cache it
        instead of iterating over the tree multiple times.
        """
        scale = self._strides[depth]
        base_coords = self._coords[depth]

        if expand >= 3:
            mc_offsets = self._range_kernel()(expand) * scale
            base_coords = (base_coords.unsqueeze(dim=1).repeat(1, mc_offsets.size(0), 1) +
                           mc_offsets.unsqueeze(0)).view(-1, 3)
            base_coords = torch_unique(base_coords, dim=0)

        if conforming:
            base_coords = (base_coords / scale / 2.).floor().int() * scale * 2
            base_coords = torch_unique(base_coords, dim=0)
            conform_offsets = torch.tensor(
                self.CONFORM_OFFSETS, dtype=torch.int32, device=base_coords.device) * scale
            base_coords = (base_coords.unsqueeze(dim=1).repeat(1, 8, 1) +
                           conform_offsets.unsqueeze(0)).view(-1, 3)

        return base_coords

    def get_num_voxels(self, depth: int) -> int:
        """
        :return: number of voxels in a given layer
        """
        return self._coords[depth].size(0) if self._coords[depth] is not None else 0

    def get_voxel_centers(self, depth: int, normalized: bool = False):
        """
        Get the centroid coordinates of all existing voxels at depth.
        :param depth: int
        :param normalized: if True, then divide the coordinates by voxel size, so that unit 1 is a single voxel.
        :return: (N, 3) float32 torch.Tensor
        """
        return (self.get_coords(depth) + self._strides[depth] / 2.) * (self._voxel_size if not normalized else 1.0)

    def __repr__(self) -> str:
        stat = f"Depth={self.depth}:\n"
        for stride, coords in zip(self._strides, self._coords):
            if coords is None:
                stat += f" + [{stride}] Empty\n"
                continue
            c_min = torch.min(coords, dim=0).values
            c_max = torch.max(coords, dim=0).values
            stat += f" + [{stride}] #Voxels={coords.size(0)} " \
                    f"Bound=[{c_min[0]},{c_max[0]}]x[{c_min[1]},{c_max[1]}]x[{c_min[2]},{c_max[2]}]\n"
        return stat

    def get_coords_neighbours(self, source_coords: torch.Tensor, source_stride: int, target_depth: int,
                              nn_kernel: torch.Tensor, conv_based: bool = False, transposed: bool = False):
        """
        A generic interface for querying neighbourhood information. (This is without cache)
            For all source (data), find all target whose neighbourhood (in target level) covers it,
        will also return the relative position of the two.
        :param source_coords: (N, 3)
        :param source_stride: int
        :param target_depth: int
        :param conv_based: if True, then used for convolution
        :param nn_kernel: Unit w.r.t. target strides.
        :param transposed: allows efficient per-source handling.
        """
        assert 0 <= target_depth < self._depth

        if not conv_based:
            # Flaw: If the layers are different (source stride < target stride), you may end up with
            #   neighbours that has no overlap support.
            assert source_stride <= self._strides[target_depth], "Data must be deeper and has more nodes."
            # Compute voxel center offsets.
            quantized_source_coords = torch.div(
                source_coords.detach() + 0.5 * source_stride, self._strides[target_depth],
                rounding_mode='floor').int() * self._strides[target_depth]
            c_offset = (quantized_source_coords - source_coords) / source_stride + \
                       (self._strides[target_depth] // source_stride - 1) / 2.
        else:
            assert not source_coords.requires_grad
            assert source_stride >= self._strides[target_depth], "Data must be sparser and shallower."
            quantized_source_coords = source_coords

        hash_res = self._hash_table[target_depth].query(
            quantized_source_coords, nn_kernel * self._strides[target_depth])  # (K, N)

        if transposed:
            hash_res = hash_res.T

        nbsizes = torch.sum(hash_res != -1, dim=1)

        if transposed:
            source_ids, kernel_ids = torch.where(hash_res != -1)
            target_ids = hash_res[source_ids, kernel_ids]
        else:
            kernel_ids, source_ids = torch.where(hash_res != -1)
            target_ids = hash_res[kernel_ids, source_ids]

        neighbour_types = nn_kernel[kernel_ids]

        if not conv_based:
            neighbour_types = neighbour_types.float()
            neighbour_types *= self._strides[target_depth] / source_stride
            neighbour_types += c_offset[source_ids, :3]

        return source_ids, target_ids, neighbour_types, nbsizes

    def get_self_neighbours(self, source_depth: int, target_depth: int, target_range: int,
                            conv_based: bool = False):
        """
        :param source_depth: source depth where you want the coord id to start from
        :param target_depth: target depth where you want the coord id to shoot to
        :param target_range: must be odd, logical neighbourhood range to search for, e.g. 5 for B2 basis.
        :param conv_based: if True, then used for convolution
        :return: [sid, tid]
        """
        assert 0 <= source_depth < self.depth and 0 <= target_depth < self.depth

        tree_coords, tree_strides = self._coords, self._strides

        # conv_based flag will be ignored if source-depth == target-depth, because this is anyway
        #   covered in both situations.
        inv_op = False
        if not conv_based and source_depth != target_depth:
            neighbour_maps = self._region_nmap
            # In the case where source is shallower/fewer than target, we inverse the operation
            if source_depth > target_depth:
                source_depth, target_depth, inv_op = target_depth, source_depth, True
        else:
            neighbour_maps = self._conv_nmap

        def recover_inv_op(inv_src_ids, inv_tgt_ids, inv_nts, inv_nbs):
            if not inv_op:
                return inv_src_ids, inv_tgt_ids, inv_nts, inv_nbs
            else:
                # Filter far away nodes.
                near_mask = torch.all(inv_nts.abs() < target_range / 2. + 1.0e-6, dim=1)
                # Convert back neighbour types.
                inv_nts = -inv_nts / tree_strides[target_depth] * tree_strides[source_depth]
                return inv_tgt_ids[near_mask], inv_src_ids[near_mask], inv_nts[near_mask], None

        exist_src, exist_tgt, exist_nt, exist_nbs, lack_range = \
            neighbour_maps.get_map(source_depth, target_depth, target_range)

        if lack_range is None:
            return recover_inv_op(exist_src, exist_tgt, exist_nt, exist_nbs)

        # Only compute incremental part:
        neighbour_kernel = self._range_kernel()(target_range)
        starting_lap = max(0, lack_range[0] - 2)
        starting_lap = starting_lap ** 3
        neighbour_kernel = neighbour_kernel[starting_lap:]

        source_ids, target_ids, neighbour_types, nbsizes = self.get_coords_neighbours(
            tree_coords[source_depth], tree_strides[source_depth], target_depth, neighbour_kernel, conv_based
        )

        if exist_src is not None:
            source_ids = torch.cat([exist_src, source_ids], dim=0)
            target_ids = torch.cat([exist_tgt, target_ids], dim=0)
            neighbour_types = torch.cat([exist_nt, neighbour_types], dim=0)
            nbsizes = torch.cat([exist_nbs, nbsizes], dim=0)

        # Cache result for future use.
        neighbour_maps.update_map(source_depth, target_depth, target_range,
                                  [source_ids, target_ids, neighbour_types, nbsizes])

        return recover_inv_op(source_ids, target_ids, neighbour_types, nbsizes)

    def evaluate_voxel_status(self, coords: torch.Tensor, depth: int):
        """
        Evaluate status in the hierarchy, please refer to core.hashtree.VoxelStatus for numerical values:
            VoxelStatus.VS_NON_EXIST: This voxel shouldn't exist
            VoxelStatus.VS_EXIST_STOP: This voxel exists and is a leaf node
            VoxelStatus.VS_EXIST_CONTINUE: This voxel exists and has >0 children
        :param coords: (N, 3) torch.Tensor coordinates in the world space
        :param depth: int
        :return: (N, ) long torch.Tensor, indicating voxel status
        """
        from torch_spsr.core.hashtree import VoxelStatus
        status = torch.full((coords.size(0),), VoxelStatus.VS_NON_EXIST.value, dtype=torch.long, device=coords.device)
        sidx, _, _, _ = self.get_coords_neighbours(
            coords, self._strides[depth], depth, self._identity_kernel(), conv_based=True)
        status[sidx] = VoxelStatus.VS_EXIST_STOP.value

        if depth > 0:
            # Next level.
            conform_offsets = torch.tensor(self.CONFORM_OFFSETS, dtype=torch.int32, device=self._device) * \
                              self._strides[depth - 1]
            conform_coords = (coords[sidx].unsqueeze(dim=1).repeat(1, 8, 1) + conform_offsets.unsqueeze(0)).view(-1, 3)
            qidx, _, _, _ = self.get_coords_neighbours(
                conform_coords, self._strides[depth - 1], depth - 1, self._identity_kernel(), conv_based=True)
            qidx = torch.div(qidx, 8, rounding_mode='floor')
            status[sidx[qidx]] = VoxelStatus.VS_EXIST_CONTINUE.value

        return status

    def split_data(self, xyz: torch.Tensor, data_depth: int, data: torch.Tensor):
        """
        Obtain the tri-linearly interpolated data located at xyz.
        :param xyz: torch.Tensor (N, 3)
        :param data_depth: int
        :param data: torch.Tensor (M, K), where K is feature dimension, and M = self.get_num_voxels(data_depth)
        :return: (N, K) torch.Tensor
        """
        tree_stride = self._strides[data_depth]
        assert data.size(0) == self._coords[data_depth].size(0), "Tree data does not agree on size."

        alpha_coords, alpha_weight = self._trilinear_weights(xyz, tree_stride)
        alpha_source, alpha_target, _, _ = self.get_coords_neighbours(
            alpha_coords, tree_stride, data_depth, self._identity_kernel())
        return torch_scatter.scatter_sum(data[alpha_target] * alpha_weight[alpha_source, None],
                                         alpha_source % xyz.size(0), dim=0,
                                         dim_size=xyz.size(0))

    def splat_data(self, xyz: torch.Tensor, data_depth: int, data: torch.Tensor = None,
                   check_corr: bool = True, return_nf_mask: bool = False):
        """
        Splat data located at xyz to the tree voxels.
        :param xyz: torch.Tensor (N, 3)
        :param data_depth: int
        :param data: torch.Tensor (N, K)
        :param check_corr: if True, check if data is fully supported by its 8 neighbours
        :param return_nf_mask: Legacy, do not use.
        :return: (M, K) or (M,), where M = self.get_num_voxels(data_depth)
        """
        if data is not None:
            assert data.size(0) == xyz.size(0), "Input data must agree with xyz in size."
        else:
            data = 1

        tree_stride = self._strides[data_depth]
        alpha_coords, alpha_data = self._trilinear_weights(xyz, tree_stride, data)

        # align normal_coords and tree_coords.
        alpha_source, alpha_target, _, nb_sizes = self.get_coords_neighbours(
            alpha_coords, tree_stride, data_depth, self._identity_kernel(), transposed=True)

        # Make sure that each query coordinates has one correspondent:
        if alpha_source.size(0) < alpha_coords.size(0) and check_corr:
            print("Warning: Some grids that normal should be splatted onto is missing because expansion is too small. "
                  f"# Should = {alpha_coords.size(0)}, Actual = {alpha_source.size(0)}.")
        splat_res = torch_scatter.scatter_sum(alpha_data[alpha_source], alpha_target, dim=0,
                                         dim_size=self._coords[data_depth].size(0))
        if return_nf_mask:
            # If a point can only be splatted on to less than 4 voxels, it is a bad splat.
            return splat_res, nb_sizes.reshape(8, -1).sum(0) < 4
        return splat_res

    def build_hierarchy_dense(self, xyz: torch.Tensor, expand_range: int = 0):
        """
        Rebuild the tree structure, based on current xyz, voxel_size and depth.
        """
        if expand_range == 2:
            unique_coords = self._quantize_coords(xyz, 0)
        else:
            coords = torch.div(xyz, self._voxel_size).floor().int()
            unique_coords = torch_unique(coords, dim=0)
            if expand_range > 0:
                offsets = self._range_kernel()(expand_range)
                my_pad = (unique_coords.unsqueeze(dim=1).repeat(1, offsets.size(0), 1) +
                          offsets.unsqueeze(0)).view(-1, 3)
                unique_coords = torch_unique(my_pad, dim=0)

        self._coords = [unique_coords]
        for d in range(1, self.depth):
            coords = torch.div(self._coords[-1], self._strides[d], rounding_mode='floor') * self._strides[d]
            coords = torch_unique(coords, dim=0)
            self._coords.append(coords)
        self._update_hash_table()

    def build_hierarchy_subdivide(self, xyz: torch.Tensor, subdivide_policy, expand: bool = False,
                                  limit_adaptive_depth: int = 100, **policy_kwargs):
        """
        Build a hierarchy, based on subdivision policy
        """
        current_pts = xyz / self._voxel_size
        inv_mapping = None
        xyz_depth = torch.full((xyz.size(0),), fill_value=self._depth - 1, device=self._device, dtype=torch.int)
        xyz_depth_inds = torch.arange(xyz.size(0), device=self._device, dtype=torch.long)

        for d in range(self._depth - 1, -1, -1):
            if d != self._depth - 1:
                nxt_mask = subdivide_policy(current_pts, inv_mapping, **policy_kwargs)
                current_pts = current_pts[nxt_mask]
                xyz_depth_inds = xyz_depth_inds[nxt_mask]
                policy_kwargs = {k: v[nxt_mask] if isinstance(v, torch.Tensor) else v for k, v in policy_kwargs.items()}
                xyz_depth[xyz_depth_inds] -= 1
            coords = torch.div(current_pts, self.get_stride(d), rounding_mode='floor').int() * self._strides[d]
            unique_coords, inv_mapping = torch_unique(coords, dim=0, return_inverse=True)
            self._coords[d] = unique_coords
        xyz_depth.clamp_(max=limit_adaptive_depth - 1)

        if expand:
            self._coords = []
            for d in range(self.depth):
                depth_samples = xyz[xyz_depth <= d]
                coords = self._quantize_coords(depth_samples, d)
                if depth_samples.size(0) == 0:
                    print(f"-- disregard level {d} due to insufficient samples!")
                self._coords.append(coords)
        self._update_hash_table()

        return xyz_depth

    def build_hierarchy_adaptive(self, xyz: torch.Tensor, xyz_density: torch.Tensor, log_base: float = 4.0,
                                 min_density: float = 8.0,
                                 limit_adaptive_depth: int = 100) -> torch.Tensor:
        """
        Build the hierarchy by first determine the integer level of each point (based on xyz_density, log_base and
        min_density), then splat the points onto the tree structure.
        :param xyz: (N, 3) torch.Tensor
        :param xyz_density: (N, ) float torch.Tensor
        :param log_base: float
        :param min_density: float, minimum density in each voxel. If exceed, go to coarser level.
        :param limit_adaptive_depth: int. Maximum adaptive number of levels.
        :return torch.Tensor long. (N, ) level that the point lies in.
        """
        # Compute expected depth.
        xyz_depth = -(torch.log(xyz_density / min_density) / np.log(log_base)).floor().int().clamp_(max=0)
        xyz_depth.clamp_(max=min(self.depth - 1, limit_adaptive_depth - 1))

        # self.xyz_depth = (self.xyz[:, 0] < 0.0).int()
        # self.xyz_density = torch.ones((self.xyz.size(0),), device=self.device)

        # Determine octants by splatting.
        self._coords = []
        for d in range(self.depth):
            depth_samples = xyz[xyz_depth <= d]
            coords = self._quantize_coords(depth_samples, d)
            # if depth_samples.size(0) == 0:
            #     print(f"-- disregard level {d} due to insufficient samples!")
            self._coords.append(coords)

        self._update_hash_table()
        return xyz_depth

    def update_coords(self, depth: int, coords: Union[torch.Tensor, None]):
        """
        Update the structure of the tree. This is mainly used during decoder's structure building stage.
            For now you could assume that the structure at depth does not exist yet.
            But I think we should have some general function that alters the tree structure.
        :param depth: int
        :param coords: torch.Tensor (N, 3) or None, if None, then this layer would be empty.
        :return:
        """
        if coords is None:
            coords = torch.zeros((0, 3), dtype=torch.int32, device=self._device)
        assert coords.ndim == 2 and coords.size(1) == 3, coords.size()
        self._coords[depth] = coords
        self._hash_table[depth] = CuckooHashTable(data=self._coords[depth])

    def _identity_kernel(self):
        return torch.tensor([[0, 0, 0]], dtype=torch.int32, device=self._device)

    def _quantize_coords(self, xyz: torch.Tensor, data_depth: int):
        # Note this is just splat_data with NEW_BRANCH.
        tree_stride = self._strides[data_depth]
        alpha_coords, _ = self._trilinear_weights(xyz, tree_stride)
        alpha_coords = torch_unique(alpha_coords, dim=0)
        return alpha_coords

    def _update_hash_table(self):
        for d in range(self.depth):
            self._hash_table[d] = CuckooHashTable(data=self._coords[d])
            assert self._hash_table[d].dim == 3

    def _trilinear_weights(self, xyz: torch.Tensor, tree_stride: int, xyz_data: torch.Tensor = 1,
                           compute_grad: bool = False):
        # Gradient is alpha_data w.r.t. xyz.
        q_coords = xyz / self._voxel_size
        d_coords = (q_coords / tree_stride).floor() * tree_stride
        rel_coords = q_coords - d_coords - tree_stride / 2.
        oct_sign = torch.sign(rel_coords)
        oct_local = torch.abs(rel_coords) / tree_stride

        alpha_coords = []
        alpha_data = []
        grad_alpha_data = []
        for nx, ny, nz in self.CONFORM_OFFSETS:
            alpha_coords.append((d_coords + torch.stack([nx * oct_sign[:, 0],
                                                         ny * oct_sign[:, 1],
                                                         nz * oct_sign[:, 2]],
                                                        dim=1) * tree_stride).int())
            alpha_x = oct_local[:, 0] if nx == 1 else 1 - oct_local[:, 0]
            alpha_y = oct_local[:, 1] if ny == 1 else 1 - oct_local[:, 1]
            alpha_z = oct_local[:, 2] if nz == 1 else 1 - oct_local[:, 2]
            alpha_os = alpha_x * alpha_y * alpha_z

            if compute_grad:
                assert xyz_data == 1, "Not supported!"
                d_alpha_x = (oct_sign[:, 0] if nx == 1 else -oct_sign[:, 0]) / (self._voxel_size * tree_stride)
                d_alpha_y = (oct_sign[:, 1] if ny == 1 else -oct_sign[:, 1]) / (self._voxel_size * tree_stride)
                d_alpha_z = (oct_sign[:, 2] if nz == 1 else -oct_sign[:, 2]) / (self._voxel_size * tree_stride)
                grad_alpha_data.append(torch.stack([
                    d_alpha_x * alpha_y * alpha_z,
                    alpha_x * d_alpha_y * alpha_z,
                    alpha_x * alpha_y * d_alpha_z
                ], dim=1))

            alpha_data.append(alpha_os * xyz_data if isinstance(xyz_data, int) or xyz_data.ndim == 1 else
                              alpha_os[:, None] * xyz_data)
        alpha_coords = torch.cat(alpha_coords, dim=0)
        alpha_data = torch.cat(alpha_data, dim=0)

        if compute_grad:
            return alpha_coords, alpha_data, torch.cat(grad_alpha_data, dim=0)

        return alpha_coords, alpha_data