import time
import torch
import torch_scatter
import functools

from omegaconf import OmegaConf
from torch_spsr.core import ops
from torch_spsr.ext import sparse_solve


class SparseMatrixStorage:
    """
    Cache is used to store factorizations and block decompositions
        So don't alter the members after they are initialized!
    """
    def __init__(self, a_i, a_j, a_x, dim: int, dim_j: int = None):
        self.a_i = a_i.detach()
        self.a_j = a_j.detach()
        self.a_x = a_x.detach()
        self.dim_i = dim
        self.dim_j = dim_j or self.dim_i
        self.device = self.a_i.device

        # Block decompositions
        self.n_blocks = 1
        self.sub_blocks = {}    # (bi, bj) -> SparseMatrixStorage
        self.block_inds = None

    def __matmul__(self, res):
        return torch_scatter.scatter_sum(res[self.a_j] * self.a_x, self.a_i, dim_size=self.dim_i)

    @property
    def shape(self):
        return self.dim_i, self.dim_j

    @classmethod
    def from_dict(cls, data: dict, dim: int, dim_j: int = None):
        return SparseMatrixStorage(data["i"], data["j"], data["x"], dim, dim_j)

    @ops.lru_cache_class(maxsize=None)
    def csr(self):      # --> compressed_row (long), col (long), value (float)
        # Compress row:
        a_p = torch_scatter.scatter_sum(
            torch.ones((self.a_i.size(0)), dtype=torch.long, device=self.device), self.a_i, dim_size=self.dim_i)
        a_p = torch.cat([torch.zeros((1,), dtype=torch.long, device=self.device), torch.cumsum(a_p, dim=0)])
        # Order columns and data:
        csr_seq = torch.argsort(self.a_i)
        return a_p, self.a_j[csr_seq], self.a_x[csr_seq]

    @ops.lru_cache_class(maxsize=None)
    def csc(self):      # --> compressed_col (long), row (long), value (float)
        # Compress col:
        a_p = torch_scatter.scatter(
            torch.ones((self.a_j.size(0)), dtype=torch.long, device=self.device), self.a_j, dim_size=self.dim_j)
        a_p = torch.cat([torch.zeros((1,), dtype=torch.long, device=self.device), torch.cumsum(a_p, dim=0)])
        # Order rows and data:
        csc_seq = torch.argsort(self.a_j)
        return a_p, self.a_i[csc_seq], self.a_x[csc_seq]

    @ops.lru_cache_class(maxsize=None)
    def diagonal(self):
        assert self.dim_i == self.dim_j
        diag_mask = self.a_i == self.a_j
        diag_values = self.a_x[diag_mask]
        diag_i = self.a_i[diag_mask]
        diag = torch.zeros((self.dim_i, ), device=self.device)
        diag[diag_i] = diag_values
        return diag

    @ops.lru_cache_class(maxsize=None)
    def dense(self):
        return torch.sparse_coo_tensor(
            torch.stack([self.a_i, self.a_j], dim=0), self.a_x, (self.dim_i, self.dim_j)).to_dense()


class Solver(torch.autograd.Function):
    """
    Differentiable linear solver class
    """
    @staticmethod
    def forward(ctx, a: SparseMatrixStorage, a_x: torch.Tensor, b: torch.Tensor,
                forward_solver, backward_solver):
        with torch.no_grad():
            x = forward_solver(a=a, b=b.detach())

        ctx.save_for_backward(x, b)
        ctx.a = a
        ctx.backward_solver = backward_solver
        return x

    @staticmethod
    def backward(ctx, grad_x):
        x, b = ctx.saved_tensors
        a = ctx.a
        with torch.no_grad():
            grad_b = ctx.backward_solver(a, grad_x)
            grad_a_x = -grad_b[a.a_i] * x[a.a_j]
        return None, grad_a_x, grad_b, None, None


def solve_sparse(a_i: torch.Tensor, a_j: torch.Tensor, a_x: torch.Tensor, b: torch.Tensor,
                 solver_type: str, verbose_str: str = None):
    """
    Solve a sparse linear system in a differentiable manner: Ax = b
    :param a_i: COO row indices of A
    :param a_j: COO col indices of A
    :param a_x: COO values of A
    :param b: right hand side.
    :param solver_type: name of the solver
    :param verbose_str:
    :return: x: Tensor
    """
    if verbose_str is not None:
        torch.cuda.synchronize()
        start_time = time.perf_counter()

    a_mat = SparseMatrixStorage(a_i, a_j, a_x, dim=b.size(0))
    solver_type, pcg_conf = _parse_solver_type(solver_type)

    if solver_type == "pcg":
        func = functools.partial(solve_pcg, pcg_conf=pcg_conf)
        res = Solver.apply(a_mat, a_x, b, func, func)

    elif solver_type == "mixed":
        solver_forward = functools.partial(
            solve_mixed,
            fast_solver=functools.partial(solve_pcg, pcg_conf={'maxiter': 1000, 'tol': 1.0e-4}),
            fallback_solver=solve_cholmod, tol=1.0e-2, name="forward")
        solver_backward = solve_cholmod
        res = Solver.apply(a_mat, a_x, b, solver_forward, solver_backward)

    elif solver_type == "cholmod":
        res = Solver.apply(a_mat, a_x, b, solve_cholmod, solve_cholmod)

    else:
        raise NotImplementedError

    if verbose_str is not None:
        torch.cuda.synchronize()
        elapsed_time = time.perf_counter() - start_time
        with torch.no_grad():
            residual = torch.linalg.norm(a_mat @ res - b).item()
        print(verbose_str.format(time=elapsed_time, residual=residual))

    return res


def solve_cholmod(a: SparseMatrixStorage, b: torch.Tensor):
    if b.size(0) == 0:
        return torch.zeros_like(b)

    a_p, a_j, a_x = a.csr()
    a_p = a_p.int().contiguous()
    a_j = a_j.int().contiguous()
    a_x = a_x.float().contiguous()
    b = b.float().contiguous()

    try:
        x = sparse_solve.solve_cusparse(a_p, a_j, a_x, b, 1.0e-6)
    except RuntimeError:
        torch.cuda.empty_cache()
        x = sparse_solve.solve_cusparse(a_p, a_j, a_x, b, 1.0e-6)

    return x


def solve_mixed(a: SparseMatrixStorage, b: torch.Tensor,
                fast_solver, fallback_solver, tol: float = 0.01, name: str = None):
    if b.size(0) == 0:
        return torch.zeros_like(b)

    solution = fast_solver(a, b)
    fast_residual = torch.linalg.norm(a @ solution - b).item()
    if fast_residual > tol:
        solution = fallback_solver(a, b)

    return solution


def _parse_solver_type(solver_type: str):
    solver_specs = solver_type.split("@")
    solver_configs = OmegaConf.create()
    for solver_spec in solver_specs[1:]:
        kw_name, kw_value = solver_spec.split("=")
        solver_configs[kw_name] = eval(kw_value)
    return solver_specs[0], solver_configs


def solve_pcg(a: SparseMatrixStorage, b: torch.Tensor, pcg_conf):
    inv_diag_a = 1.0 / a.diagonal()
    csr_p, csr_j, csr_x = a.csr()
    max_iter = pcg_conf.get("maxiter", -1)

    res, cg_iter = sparse_solve.solve_pcg_diag(
        csr_p.int(), csr_j.int(), csr_x, b, inv_diag_a,
        pcg_conf.get("tol", 1.0e-5), max_iter, pcg_conf.get("res_fix", False)
    )

    if pcg_conf.get("verbose", False):
        print("PCG Iteration ended in", cg_iter)

    return res
