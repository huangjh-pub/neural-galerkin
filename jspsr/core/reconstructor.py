from typing import Union

import jittor as jt
import torch
import torch_scatter
import numpy as np
from jspsr.core.hashtree import HashTree, VoxelStatus
from jspsr.bases.abc import BaseBasis
from jspsr.core.ops import screened_multiplication, marching_cubes_op, marching_cubes, torch_unique
from jspsr.ext import CuckooHashTable
from jspsr.core.solver import solve_sparse


class ScreeningData:
    """
    Sparse matrix representation (Num-pts x Num-vx)
    """
    def __init__(self, pts_ids, vx_ids, values, nb_sizes):
        self.pts_ids = pts_ids
        self.vx_ids = vx_ids
        self.values = values
        self.nb_sizes = nb_sizes


class Reconstructor:
    def __init__(self, hash_tree: HashTree, basis: BaseBasis, feat: dict = None):
        """
        Screened Poisson reconstructor class.
        :param hash_tree: The tree containing the octree structure as well as input points
        :param basis: basis function to use
        :param feat: dict that maps from integer depth to torch features, used only when basis needs features.
        """
        self.hash_tree = hash_tree
        self.fixed_level_set = None
        self.branch = hash_tree.DECODER

        self.basis = basis
        self.solutions = {}
        self.sample_weight = None

        # Initialize basis feature if not provided.
        if feat is None:
            feat = {}
            for d in range(hash_tree.depth):
                feat[d] = torch.zeros(
                    (hash_tree.get_coords_size(self.branch, d),
                     basis.get_feature_size()), device=hash_tree.device)
                basis.initialize_feature_value(feat[d])
        self.grid_features = feat

    def set_fixed_level_set(self, value):
        """
            Originally level set is determined by computing mean(chi(input)), this function allows you to
        manually specify that
        :param value: level set value
        """
        self.fixed_level_set = value

    @classmethod
    def _evaluate_screening_term(cls, data_a: ScreeningData, data_b: ScreeningData,
                                 domain_a, domain_b, pts_weight=None):
        if domain_a.size(0) == 0 or domain_b.size(0) == 0:
            return torch.zeros((0, ), device=domain_a.device)

        if pts_weight is None:
            pts_weight = torch.ones((data_a.nb_sizes.size(0), ), device=domain_a.device)
        elif isinstance(pts_weight, float):
            pts_weight = torch.full((data_a.nb_sizes.size(0), ), fill_value=pts_weight, dtype=torch.float32,
                                    device=domain_a.device)

        domain_table = CuckooHashTable(torch.stack([domain_a, domain_b], dim=1), enlarged=True)
        term_res = screened_multiplication(domain_table.object,
                                           data_a.values, data_b.values,
                                           data_a.vx_ids, data_b.vx_ids,
                                           data_a.nb_sizes, data_b.nb_sizes, pts_weight)

        return term_res

    def solve_multigrid(self, start_depth, end_depth, normal_data: dict,
                        screen_alpha: Union[float, torch.Tensor] = 0.0, screen_xyz: torch.Tensor = None,
                        screen_delta: Union[float, torch.Tensor] = 0.1,
                        solver: str = "pcg", verbose: bool = True):
        """
            Build and solve the linear system L alpha = d, using our coarse-to-fine solver.
        Note that the full V-cycle is not supported in this repo. Normal is however not smoothed
        because empirically we've found no difference.
            The energy function defined in our paper is solver within a truncated domain, with
        explicit dirichlet constraints that the boundary evaluates to 0. We choose not to eliminate
        such constraints because that will introduce many heterogeneous integral computations on
        the boundary.
        :param start_depth: int, the coarsest level for the solver
        :param end_depth: int, the finest level for the solver
        :param normal_data: dictionary that maps from depth to splatted normal data (x, 3)
        :param screen_alpha: float or Tensor. Weight of the screening term
        :param screen_xyz: None or Tensor. positional constraints to the system.
        :param screen_delta: float or Tensor. Target scalar value as described in the paper.
        :param solver: you can choose from 'cholmod' or 'pcg' or 'mixed'.
        :param verbose: Output debug information during solve.
        """
        if isinstance(screen_alpha, torch.Tensor):
            assert screen_xyz is not None, "Must provide points to be screened."
            assert screen_alpha.size(0) == screen_xyz.size(0)
            should_screen = True
        else:
            should_screen = screen_alpha > 0.0

        self.solutions = {}
        neighbour_range = 2 * self.basis.get_domain_range() - 1

        # Basis pre-evaluation for screening term.
        screen_data = {}
        if should_screen:
            base_coords = screen_xyz / self.hash_tree.voxel_size - 0.5
            for d in range(end_depth, start_depth + 1):
                pts_ids, vx_ids, tgt_offsets, nb_sizes = self.hash_tree.get_neighbours_data(
                    base_coords, 1, d, self.hash_tree.get_range_kernel(self.basis.get_domain_range()),
                    self.branch, transposed=True)
                query_coords = -tgt_offsets / self.hash_tree.get_stride(self.branch, d)
                query_val = self.basis.evaluate(feat=self.grid_features[d], xyz=query_coords, feat_ids=vx_ids)
                screen_data[d] = ScreeningData(pts_ids, vx_ids, query_val, nb_sizes)

        for d in range(start_depth, end_depth - 1, -1):
            screen_factor = (1 / 4.) ** d

            # Build RHS:
            rhs_val = 0
            for data_depth, depth_normal_data in normal_data.items():
                normal_ids, tree_ids, normal_offset, _ = self.hash_tree.get_neighbours(
                    data_depth, d,
                    self.basis.get_domain_range(),
                    self.branch)
                partial_sums = self.basis.integrate_const_deriv_product(
                    data=-depth_normal_data[normal_ids],
                    target_feat=self.grid_features[d],
                    rel_pos=normal_offset,
                    data_stride=self.hash_tree.get_stride(self.branch, data_depth),
                    target_stride=self.hash_tree.get_stride(self.branch, d),
                    target_ids=tree_ids
                )
                rhs_val += torch_scatter.scatter_add(
                    partial_sums, tree_ids, dim=0, dim_size=self.hash_tree.get_coords_size(self.branch, d))

            if should_screen:
                if isinstance(screen_alpha, torch.Tensor) or isinstance(screen_delta, torch.Tensor):
                    mult = (screen_delta * screen_alpha)[screen_data[d].pts_ids]
                else:
                    mult = screen_alpha * screen_delta
                rhs_val += screen_factor * torch_scatter.scatter_sum(
                    screen_data[d].values * mult,
                    screen_data[d].vx_ids, dim_size=self.hash_tree.get_coords_size(self.branch, d)
                )

            # Correction:
            for dd in range(start_depth, d, -1):
                src_ids, tgt_ids, rel_pos, _ = self.hash_tree.get_neighbours(
                    d, dd, target_range=neighbour_range, branch=self.branch)
                a_d_dd_val = self.basis.integrate_deriv_deriv_product(
                    source_feat=self.grid_features[d],
                    target_feat=self.grid_features[dd],
                    rel_pos=rel_pos,
                    source_stride=self.hash_tree.get_stride(self.branch, d),
                    target_stride=self.hash_tree.get_stride(self.branch, dd),
                    source_ids=src_ids, target_ids=tgt_ids)
                if should_screen:
                    a_d_dd_val = a_d_dd_val + screen_factor * self._evaluate_screening_term(
                        screen_data[d], screen_data[dd], src_ids, tgt_ids, screen_alpha)
                rhs_val -= torch_scatter.scatter_sum(self.solutions[dd][tgt_ids] * a_d_dd_val,
                                                     src_ids, dim_size=rhs_val.size(0))

            # Build LHS:
            src_ids, tgt_ids, rel_pos, _ = self.hash_tree.get_neighbours(d, d, target_range=neighbour_range,
                                                                         branch=self.branch)
            lhs_val = self.basis.integrate_deriv_deriv_product(
                source_feat=self.grid_features[d],
                target_feat=self.grid_features[d],
                rel_pos=rel_pos,
                source_stride=self.hash_tree.get_stride(self.branch, d),
                target_stride=self.hash_tree.get_stride(self.branch, d),
                source_ids=src_ids, target_ids=tgt_ids)

            if should_screen:
                lhs_val = lhs_val + screen_factor * self._evaluate_screening_term(
                    screen_data[d], screen_data[d], src_ids, tgt_ids, screen_alpha)

            if solver == "mixed":
                cur_solver = "mixed" if d == end_depth else "cholmod"
            else:
                cur_solver = solver
            self.solutions[d] = solve_sparse(src_ids, tgt_ids, lhs_val, rhs_val, cur_solver)

            # Dump residual for comparison
            if verbose:
                residual = torch_scatter.scatter_sum(self.solutions[d][tgt_ids] * lhs_val, src_ids,
                                                     dim_size=rhs_val.size(0)) - rhs_val
                print(f"Solving complete at level {d}, residual = {torch.linalg.norm(residual).item()}.")

    def evaluate_raw_chi(self, xyz: torch.Tensor, compute_mask: bool = False,
                         compute_grad: bool = False, depths: list = None):
        """
        Evaluate the chi value at (x,y,z)
        :param depths: visualize only the depth in the list, default is None
        :param compute_grad: whether to compute gradient of the field
        :param xyz: torch.Tensor (N x 3). metric-space positions
        :param compute_mask: bool for debugging purpose
        :return: (N,) chi value.
        """
        assert len(self.solutions) > 0, "Please run solver before evaluation."

        sdf_vals = 0
        sdf_mask = torch.zeros((xyz.size(0), ), dtype=bool, device=xyz.device) if compute_mask else None
        for level_d, level_solution in self.solutions.items():
            if depths is not None and level_d not in depths:
                continue

            sdf_val = self.hash_tree.evaluate_interpolated(
                xyz, self.basis, self.grid_features[level_d], level_d, level_solution,
                compute_mask, compute_grad=compute_grad)
            if compute_mask:
                sdf_vals += sdf_val[0]
                sdf_mask = torch.logical_or(sdf_mask, sdf_val[1])
            else:
                sdf_vals += sdf_val

        if compute_mask:
            return sdf_vals, sdf_mask
        return sdf_vals

    def get_mean_chi(self):
        if self.fixed_level_set is not None:
            return self.fixed_level_set
        sdf_surface = self.evaluate_raw_chi(self.hash_tree.xyz)
        if self.sample_weight is None:
            print("Warning: Sample weight not set.")
            return torch.mean(sdf_surface)
        else:
            return torch.sum(sdf_surface * self.sample_weight) / torch.sum(self.sample_weight)

    def evaluate_chi(self, xyz: torch.Tensor, compute_mask: bool = False, max_points: int = -1, depths: list = None):
        """
        Evaluate the implicit field value, possibly with chunking
        """
        mean_chi = self.get_mean_chi()

        n_chunks = int(np.ceil(xyz.size(0) / max_points)) if max_points > 0 else 1
        xyz_chunks = torch.chunk(xyz, n_chunks)
        sdf_val_chunks = []

        for xyz in xyz_chunks:
            sdf_val = self.evaluate_raw_chi(xyz, compute_mask=compute_mask, depths=depths)
            sdf_val_chunks.append(sdf_val)

        if compute_mask:
            return torch.cat([t[0] for t in sdf_val_chunks]) - mean_chi, torch.cat([t[1] for t in sdf_val_chunks])

        return torch.cat(sdf_val_chunks) - mean_chi

    def extract_mesh(self, base_coords: torch.Tensor, chi_field: torch.Tensor, chi_depth: int, build_o3d_mesh: bool = True):
        """
        Extract mesh at a specific depth, given densely-evaluated implicit function values.
        :param base_coords: coordinates of the evaluation point
        :param chi_field: sampled implicit function values
        :param chi_depth: int, depth of the mesh extraction
        :param build_o3d_mesh: whether to use Open3D to build TriangleMesh
        :return: o3d.geometry.TriangleMesh or (vertices Vx3, triangles Tx3, normals Vx3)
        """
        scale = self.hash_tree.get_stride(self.branch, chi_depth)

        # Extract mesh.
        num_lif = base_coords.size(0)
        chi_resolution = chi_field.size(0) // num_lif
        chi_resolution = round(chi_resolution ** (1 / 3.))

        chi_field = chi_field.reshape(-1, chi_resolution, chi_resolution, chi_resolution)
        vertices, triangles, normals, _ = marching_cubes_op(
            base_coords.float() / scale, chi_field
        )
        vertices = vertices * (scale * self.hash_tree.voxel_size)

        if build_o3d_mesh:
            import open3d as o3d

            vertices = vertices.cpu().numpy().astype(float)
            triangles = triangles.cpu().numpy().astype(np.int32)
            normals = normals.cpu().numpy().astype(float)

            final_mesh = o3d.geometry.TriangleMesh()
            final_mesh.vertices = o3d.utility.Vector3dVector(vertices)
            final_mesh.vertex_normals = o3d.utility.Vector3dVector(normals)
            final_mesh.triangles = o3d.utility.Vector3iVector(triangles)

            return final_mesh

        else:
            return vertices, triangles, normals

    def extract_multiscale_mesh(self, n_upsample: int = 1, max_depth: int = 100, expand: int = 0, trim: bool = False,
                                build_o3d_mesh: bool = True, max_points: int = -1):
        """
        https://www.cs.rice.edu/~jwarren/papers/dmc.pdf
            (Possible extensions: Use hermite data to compute feature locations & Manifold dual marching cubes)
        :param n_upsample: samples within each primal grid.
        :param max_depth: maximum depth to extract
        :param expand: size of expansion for the tree, set to 3 to guarantee no information is lost.
        :param trim: whether to keep only leaf voxels
        :param build_o3d_mesh: bool whether to build Open3D mesh.
        :param max_points: int, maximum number of points.
        :return: (vertex, triangle) tuple or o3d.geometry.TriangleMesh
        """
        max_depth = min(max_depth, self.hash_tree.depth)

        # Make tree
        conformal_primal_base = {}

        if trim:
            for d in range(max_depth):
                base_coords = self.hash_tree.get_coords(self.branch, d)
                coords_status = self.hash_tree.evaluate_voxel_status(self.branch, base_coords, d)
                conformal_primal_base[d] = base_coords[coords_status == VoxelStatus.VS_EXIST_STOP.value]
        else:
            # Make conformal tree (from fine to coarse) and filter all leaves
            conformal_mask = {}
            for d in range(max_depth):
                _, base_coords = self.hash_tree.get_test_grid(0, d, expand, conforming=d < max_depth - 1)
                # Keep only leaf nodes (by inspecting whether it has children)
                #   Mask has to be applied next round, because the pruning of parents still need a full structure.
                if d > 0:
                    children_table = CuckooHashTable(data=conformal_primal_base[d - 1])
                    ol_mask = children_table.query(base_coords) == -1
                    # No more nodes exist (including this layer)
                    if not torch.any(ol_mask):
                        max_depth = d
                        break
                    conformal_mask[d] = ol_mask
                conformal_primal_base[d] = base_coords
            for d in range(1, max_depth):
                conformal_primal_base[d] = conformal_primal_base[d][conformal_mask[d]]

        # Expand with the sample factor (build primal grids)
        expand_voxel_size = self.hash_tree.voxel_size / n_upsample
        expand_coords = torch.arange(0, n_upsample, dtype=torch.int, device=self.hash_tree.device)
        expand_coords = torch.stack(torch.meshgrid(expand_coords, expand_coords, expand_coords, indexing='ij'), dim=3)
        expand_coords = expand_coords.view(-1, 3)
        expand_primal_base = {}
        for d in range(max_depth):
            scale = self.hash_tree.get_stride(self.branch, d)
            b_d = conformal_primal_base[d] * n_upsample
            b_d = (b_d.unsqueeze(1) + (expand_coords * scale).unsqueeze(0)).view(-1, 3)
            expand_primal_base[d] = b_d  # (N * n_upsample ** 3, 3)

        # Identify dual grids (iterate 8 corners of primal voxels)
        dual_centers = []
        for d in range(max_depth):
            scale = self.hash_tree.get_stride(self.branch, d)
            for offset in [(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1), (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)]:
                dual_centers.append(expand_primal_base[d] + torch.tensor(
                    offset, dtype=torch.int32, device=self.hash_tree.device) * scale)
        dual_centers = torch.cat(dual_centers, 0)
        dual_centers = torch_unique(dual_centers, dim=0)  # (DC, 4)

        # Populate the filter incomplete dual grids. (DC, 8, 2 = depth + inds)
        dual_centers_table = CuckooHashTable(data=dual_centers)
        acc_inds, acc_inds_count = marching_cubes.dual_marching_cubes_indices(
            dual_centers_table.object, expand_primal_base,
            {d: self.hash_tree.get_stride(self.branch, d) for d in range(max_depth)},
            {d: sum([expand_primal_base[dd].size(0) for dd in range(d)]) for d in range(max_depth)})
        acc_inds = acc_inds[acc_inds_count == 8]

        # Obtain dual corners (we assume to be primal centers) and evaluate them
        dual_corners = []
        dual_values = []
        for d in range(max_depth):
            dc_coords = (expand_primal_base[d] + self.hash_tree.get_stride(self.branch, d)) * expand_voxel_size
            dual_corners.append(dc_coords)
            dual_values.append(self.evaluate_chi(dc_coords, max_points=max_points))
        dual_corners = torch.cat(dual_corners, 0)
        dual_values = torch.cat(dual_values, 0)

        # Marching cubes on dual grids.
        tris, vert_ids = marching_cubes.dual_marching_cubes_sparse(acc_inds, dual_corners, dual_values)
        unq, triangles = torch_unique(vert_ids.view(-1, 2), dim=0, return_inverse=True)
        vertices = torch.empty((unq.size(0), 3), device=vert_ids.device)
        vertices[triangles] = tris.view(-1, 3)
        triangles = triangles.view(-1, 3)

        if build_o3d_mesh:
            import open3d as o3d

            vertices = vertices.cpu().numpy().astype(float)
            triangles = triangles.cpu().numpy().astype(np.int32)

            final_mesh = o3d.geometry.TriangleMesh()
            final_mesh.vertices = o3d.utility.Vector3dVector(vertices)
            final_mesh.triangles = o3d.utility.Vector3iVector(triangles)

            return final_mesh
        else:
            return vertices, triangles
