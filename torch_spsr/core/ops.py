import functools
import weakref

import torch
import torch_scatter
from torch_spsr.ext import marching_cubes, sparse_op


class MarchingCubes(torch.autograd.Function):
    """
    Differentiable implementation of sparse marching cubes. (primal version)
        Technique is from MeshSDF.
    """
    @staticmethod
    def forward(ctx, base_coords, sdf_val, bandwidth=1.0):
        tris, vert_ids, src_ids, src_weights = marching_cubes.marching_cubes_sparse(base_coords, sdf_val)

        # Deduplicate vertices.
        unq, triangles = torch_unique(vert_ids.view(-1, 4), dim=0, return_inverse=True)
        vertices = torch.empty((unq.size(0), 3), device=vert_ids.device)
        vertices[triangles] = tris.view(-1, 3)

        # This will 'randomly' pick one solution for overlapping edges.
        vert_src_ids = torch.empty((unq.size(0), src_ids.size(-1)), dtype=src_ids.dtype, device=src_ids.device)
        vert_src_ids[triangles] = src_ids.view(-1, src_ids.size(-1))
        vert_src_weights = torch.empty((unq.size(0), src_weights.size(-1)),
                                       dtype=src_weights.dtype, device=src_weights.device)
        vert_src_weights[triangles] = src_weights.view(-1, src_weights.size(-1))

        triangles = triangles.view(-1, 3)
        normals = compute_normal(vertices, triangles)

        ctx.save_for_backward(vertices, normals, vert_src_ids.long(), vert_src_weights)
        ctx.bandwidth = bandwidth
        ctx.res = sdf_val.size()

        return vertices, triangles, normals, unq

    @staticmethod
    def backward(ctx, grad_vert, grad_face, grad_normal, grad_unq):
        vertices, normals, vert_src_ids, vert_src_weights = ctx.saved_tensors

        # grad_vert (V, 3), normals (V, 3) -> (V,), normal use positive because our chi field is positive inside.
        #   bandwidth defines the width of the band (w.r.t. mesh size) from 0 to 1 in the chi field expected.
        # because we assume extracted mesh has voxel length 1, then usually setting bandwidth=1 will be fine.
        grad_vert_sdf = ctx.bandwidth * (grad_vert * normals).sum(-1)

        # propagate vertex sdf back to grid sdf:
        #   For voxel boundaries do not check neighbours because the contribution will be distributed
        # in the previous blending stage.
        grad_grid_sdf = torch.zeros(ctx.res, device=grad_vert.device).view(-1)

        vert_src_weights = vert_src_weights[..., 0]
        torch_scatter.scatter_add(vert_src_weights * grad_vert_sdf, vert_src_ids[:, 0], out=grad_grid_sdf)
        torch_scatter.scatter_add((1 - vert_src_weights) * grad_vert_sdf, vert_src_ids[:, 1], out=grad_grid_sdf)
        grad_grid_sdf = grad_grid_sdf.view(ctx.res)

        return None, grad_grid_sdf, None, None


marching_cubes_op = MarchingCubes.apply


class ScreenedMultiplication(torch.autograd.Function):
    @staticmethod
    def forward(ctx, domain_hash_data, a_val, b_val, a_inds, b_inds, a_lengths, b_lengths, pt_w):
        res = sparse_op.screened_multiplication(
            domain_hash_data, a_val, b_val, a_inds, b_inds, a_lengths, b_lengths, pt_w)
        ctx.save_for_backward(a_val, b_val, a_inds, b_inds, a_lengths, b_lengths, pt_w)
        ctx.hash_data = domain_hash_data
        return res

    @staticmethod
    def backward(ctx, grad_res):
        a_val, b_val, a_inds, b_inds, a_lengths, b_lengths, pt_w = ctx.saved_tensors
        grad_a, grad_b, grad_w = sparse_op.screened_multiplication_backward(
            ctx.hash_data, a_val, b_val, a_inds, b_inds, a_lengths, b_lengths, pt_w, grad_res
        )
        return None, grad_a, grad_b, None, None, None, None, grad_w


screened_multiplication = ScreenedMultiplication.apply


def compute_normal(vertices: torch.Tensor, triangles: torch.Tensor):
    """
    Compute per-vertex normal by averaging triangle normals, weighted by triangle area.
    :param vertices: (V, 3)
    :param triangles: (T, 3)
    :return: (V, 3)
    """
    v01 = vertices[triangles[:, 1]] - vertices[triangles[:, 0]]
    v02 = vertices[triangles[:, 2]] - vertices[triangles[:, 0]]
    tri_norm = torch.cross(v01, v02)        # w/o normalization

    tri_norm = tri_norm.unsqueeze(1).repeat(1, 3, 1).view(-1, 3)
    vert_norm = torch_scatter.scatter_sum(tri_norm, triangles.view(-1), dim=0, dim_size=vertices.size(0))
    vert_norm = vert_norm / (torch.linalg.norm(vert_norm, dim=-1, keepdim=True) + 1e-6)

    return vert_norm


def torch_unique(input: torch.Tensor, sorted: bool = False, return_inverse: bool = False,
                 return_counts: bool = False, dim: int = None):
    """
    If used with dim, then torch.unique will return a flattened tensor. This fixes that behaviour.
    :param input: (Tensor) – the input tensor
    :param sorted: (bool) – Whether to sort the unique elements in ascending order before returning as output.
    :param return_inverse: (bool) – Whether to also return the indices for where elements in the original input
        ended up in the returned unique list.
    :param return_counts: (bool) – Whether to also return the counts for each unique element.
    :param dim: (int) – the dimension to apply unique. If None, the unique of the flattened input is returned.
        default: None
    :return: output, inverse_indices, counts
    """
    res = torch.unique(input, sorted, return_inverse, return_counts, dim)

    if dim is not None and input.size(dim) == 0:
        output_size = list(input.size())
        output_size[dim] = 0
        if isinstance(res, torch.Tensor):
            res = res.reshape(output_size)
        else:
            res = list(res)
            res[0] = res[0].reshape(output_size)

    return res


def lru_cache_class(*lru_args, **lru_kwargs):
    def decorator(func):
        @functools.wraps(func)
        def wrapped_func(self, *args, **kwargs):
            # We're storing the wrapped method inside the instance. If we had
            # a strong reference to self the instance would never die.
            self_weak = weakref.ref(self)

            @functools.wraps(func)
            @functools.lru_cache(*lru_args, **lru_kwargs)
            def cached_method(*args, **kwargs):
                return func(self_weak(), *args, **kwargs)
            setattr(self, func.__name__, cached_method)
            return cached_method(*args, **kwargs)
        return wrapped_func
    return decorator
