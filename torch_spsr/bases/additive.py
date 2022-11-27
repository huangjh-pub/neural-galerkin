import os
from pathlib import Path

import tqdm
import numpy as np
import torch_scatter
from typing import List, Union, Tuple, Dict
from torch_spsr.bases.additive_components import *
from torch_spsr.bases.abc import BaseBasis
from torch_spsr.ext import integration

import itertools
import functools
from multiprocessing import Pool


def _integrate_ff(multiple_dx, func_i, func_j, x):
    """
    Compute theta_a and theta_a_dot as in Appendix B Eq.19.
    """
    multiple, dx = multiple_dx
    stride = 2 ** multiple
    func_i = func_i.subs(x, x + dx)
    func_j = func_j.subs(x, x / stride)
    d_func_i = sp.diff(func_i, x)
    d_func_j = sp.diff(func_j, x)
    return multiple, dx, sp.integrate(func_i * func_j, (x, -100., 100.)), \
           sp.integrate(d_func_i * d_func_j, (x, -100., 100.))


def _integrate_f(multiple_ra, func_i, x):
    """
    Compute rho_a and rho_a_dot as in Appendix B Eq.21.
    Here, the domain of Rgn is smaller than the domain of the function
    """
    multiple, ra = multiple_ra
    stride = 2 ** multiple
    func_i = func_i.subs(x, x / stride)
    d_func_i = sp.diff(func_i, x)
    return multiple, ra, sp.integrate(func_i, (x, ra, ra + 1)), sp.integrate(d_func_i, (x, ra, ra + 1))


def _integrate_f_inv(multiple_ra, func_i, x):
    """
    Compute rho_a and rho_a_dot as in Appendix B Eq.21.
    Here, the domain of Rgn is larger than the domain of the function
    """
    multiple, ra = multiple_ra
    stride = 2 ** multiple
    d_func_i = sp.diff(func_i, x)
    lb, ub = max(ra, -1.5), min(ra + stride, 1.5)
    return multiple, ra, sp.integrate(func_i, (x, lb, ub)), sp.integrate(d_func_i, (x, lb, ub))


class DDIFunc(torch.autograd.Function):
    """
    Differentiable function that computes nabla_B_k^T nabla_B_l
    """
    @staticmethod
    def forward(ctx, source_expanded, target_expanded, rel_pos, f_f_i, df_df_i):
        p_mats, q_mats = integration.ddi_forward(source_expanded, target_expanded, rel_pos, f_f_i, df_df_i)
        ctx.save_for_backward(rel_pos, source_expanded, target_expanded, f_f_i, df_df_i)
        return p_mats, q_mats

    @staticmethod
    def backward(ctx, grad_p, grad_q):
        rel_pos, source_expanded, target_expanded, f_f_i, df_df_i = ctx.saved_tensors
        grad_s, grad_t = integration.ddi_backward(source_expanded, target_expanded, rel_pos, f_f_i, df_df_i, grad_p, grad_q)
        return grad_s, grad_t, None, None, None


class EvalFunc(torch.autograd.Function):
    """
    Differentiable function that evaluate the additive basis function
    """
    @staticmethod
    def forward(ctx, xyz, coeffs, feat_ids, components):
        funcs = 0
        with torch.no_grad():
            for ci, comp in enumerate(components):
                funcs += comp.evaluate(xyz) * coeffs[feat_ids, ci * 3: ci * 3 + 3]

        ctx.save_for_backward(xyz, feat_ids)
        ctx.components = components
        ctx.coeff_size = coeffs.size(0)

        return funcs

    @staticmethod
    def backward(ctx, grad_funcs):
        xyz, feat_ids = ctx.saved_tensors
        components = ctx.components

        grad_coeffs = []
        with torch.no_grad():
            for ci, comp in enumerate(components):
                cur_func = comp.evaluate(xyz)
                grad_coeffs.append(torch_scatter.scatter_sum(grad_funcs * cur_func,
                                                             feat_ids, dim=0, dim_size=ctx.coeff_size))
        return None, torch.cat(grad_coeffs, dim=1), None, None


class EvalDerivFunc(torch.autograd.Function):
    """
    Differentiable function that evaluate the derivative of the additive basis function
    """
    @staticmethod
    def forward(ctx, xyz, coeffs, feat_ids, components, stride):
        funcs = 0
        dfuncs = 0
        with torch.no_grad():
            for ci, comp in enumerate(components):
                funcs += comp.evaluate(xyz) * coeffs[feat_ids, ci * 3: ci * 3 + 3]
                dfuncs += comp.evaluate_derivative(xyz) * coeffs[feat_ids, ci * 3: ci * 3 + 3] / stride

        ctx.save_for_backward(xyz, feat_ids)
        ctx.components = components
        ctx.coeff_size = coeffs.size(0)
        ctx.stride = stride
        return funcs, dfuncs

    @staticmethod
    def backward(ctx, grad_funcs, grad_dfuncs):
        xyz, feat_ids = ctx.saved_tensors
        components = ctx.components

        grad_coeffs = []
        with torch.no_grad():
            for ci, comp in enumerate(components):
                cur_func = comp.evaluate(xyz)
                cur_func_deriv = comp.evaluate_derivative(xyz) / ctx.stride
                grad_coeffs.append(torch_scatter.scatter_sum(grad_funcs * cur_func + grad_dfuncs * cur_func_deriv,
                                                             feat_ids, dim=0, dim_size=ctx.coeff_size))
        return None, torch.cat(grad_coeffs, dim=1), None, None, None


class AdditiveBasis(BaseBasis):
    """
    B(x) = sum m_u q_u
    """
    def __init__(self, components: List[Union[str, AdditiveComponent]], init_vals: List[float]):
        """
        :param components: list of AdditiveComponent
        :param init_vals: list of initial coefficients of these components
        """
        super().__init__()
        assert len(components) == len(init_vals)
        self.components = []
        self.init_vals = init_vals
        for c in components:
            if isinstance(c, str):
                c = eval(c)
            self.components.append(c)

        ff_integral_dict, f_integral_dict = self._get_integrals()
        n_mult = len(ff_integral_dict['0,0']['df_df'])
        n_components = len(self.components)

        # Re-arrange the arrays.
        ff_integral_dict = {
            mult: {
                'df_df': np.asarray([[ff_integral_dict[f'{s},{t}']['df_df'][mult]
                                      for t in range(n_components)] for s in range(n_components)]),
                'f_f': np.asarray([[ff_integral_dict[f'{s},{t}']['f_f'][mult]
                                    for t in range(n_components)] for s in range(n_components)]),
            } for mult in range(n_mult)
        }

        self.ff_integral = self._convert_parameter(ff_integral_dict, "ff_integral")
        self.f_integral = self._convert_parameter(f_integral_dict, "f_integral")

    def _get_integrals(self, n_multiples: int = 5) -> Tuple[Dict, Dict]:
        """
        Precompute the integrals mentioned in Appendix B.
        :param n_multiples, int is
        """
        ff_integral_dict = {}
        f_integral_dict = {}

        def get_cache_path(name):
            package_path = Path(__file__).parent.parent / "data" / "additive_integrals"
            cache_path = Path.home() / ".torch_spsr" / "additive_integrals"
            if (package_path / name).exists():
                return True, package_path / name
            cache_path.mkdir(parents=True, exist_ok=True)
            return (cache_path / name).exists(), cache_path / name

        for (ai, ca), (bi, cb) in itertools.product(enumerate(self.components), repeat=2):
            ni_exist, ni_path = get_cache_path(f"ff-{ca}-{cb}.npz")
            if ni_exist:
                data_dict = np.load(ni_path, allow_pickle=True)
                ff_integral_dict[f"{ai},{bi}"] = dict(data_dict)
                continue
            print("Computing integral for pair", ca, cb)

            x = sp.Symbol('x')
            func_i = ca.symbolic(x)
            func_j = cb.symbolic(x)
            f_f_values = [[] for _ in range(n_multiples)]
            df_df_values = [[] for _ in range(n_multiples)]
            with Pool(24) as p:
                m_dx = [(m, dx) for m in range(n_multiples)
                        for dx in np.arange(-1.5 * (2 ** m) - 0.5, 1.5 * (2 ** m) + 1.5)]
                for multiple, dx, val, d_val in tqdm.tqdm(p.imap(
                    functools.partial(_integrate_ff, func_i=func_i, func_j=func_j, x=x), m_dx
                ), total=len(m_dx)):
                    f_f_values[multiple].append(float(val))
                    df_df_values[multiple].append(float(d_val))
            np.savez_compressed(ni_path, f_f=f_f_values, df_df=df_df_values)

            return self._get_integrals()

        for (ai, ca) in enumerate(self.components):
            ni_exist, ni_path = get_cache_path(f"f-{ca}.npz")
            if ni_exist:
                data_dict = np.load(ni_path, allow_pickle=True)
                f_integral_dict[ai] = dict(data_dict)
                continue
            print("Computing integral", ca)

            x = sp.Symbol('x')
            func_i = ca.symbolic(x)

            f_values = [[] for _ in range(n_multiples)]
            df_values = [[] for _ in range(n_multiples)]
            with Pool(24) as p:
                m_ra = [(m, ra) for m in range(n_multiples)
                        for ra in np.arange(-1.5 * (2 ** m), 1.5 * (2 ** m))]
                for multiple, ra, val, d_val in tqdm.tqdm(p.imap(
                    functools.partial(_integrate_f, func_i=func_i, x=x), m_ra
                ), total=len(m_ra)):
                    f_values[multiple].insert(0, float(val))
                    df_values[multiple].insert(0, float(d_val))

            f_inv_values = [[] for _ in range(n_multiples - 1)]
            df_inv_values = [[] for _ in range(n_multiples - 1)]
            with Pool(24) as p:
                m_ra = [(m, ra) for m in range(1, n_multiples)
                        for ra in np.arange(-(2 ** m) - 0.5, 1.5)]
                for multiple, ra, val, d_val in tqdm.tqdm(p.imap(
                    functools.partial(_integrate_f_inv, func_i=func_i, x=x), m_ra
                ), total=len(m_ra)):
                    f_inv_values[multiple - 1].insert(0, float(val))
                    df_inv_values[multiple - 1].insert(0, float(d_val))

            np.savez_compressed(ni_path, f=f_values, df=df_values, f_inv=f_inv_values, df_inv=df_inv_values)

            return self._get_integrals()
        return ff_integral_dict, f_integral_dict

    def _convert_parameter(self, val, name):
        """
        Convert value into class torch parameters, so that device could be managed automatically.
        """
        if isinstance(val, list) or isinstance(val, np.ndarray):
            if isinstance(val[0], float) or isinstance(val[0], np.ndarray):
                new_param = torch.nn.Parameter(torch.tensor(val, dtype=torch.float), requires_grad=False)
                self.register_parameter(name, new_param)
                return new_param
            else:
                return [self._convert_parameter(t, name + f"-{i}") for i, t in enumerate(val)]
        elif isinstance(val, dict):
            return {k: self._convert_parameter(v, name + f"-{k}") for k, v in val.items()}
        else:
            raise NotImplementedError

    def feature_to_coefficient(self, feat: torch.Tensor) -> torch.Tensor:
        """
        Convert feature that describes the basis to real coefficients of the components.
        :param feat: torch.Tensor (N, x) feature
        :return: torch.Tensor (N, 3*len(self.components)) coefficients (for x,y,z axis)
        """
        return feat

    def initialize_feature_value(self, feat: torch.Tensor) -> None:
        """
        Initialize the input features, so that the basis behaves like Bezier.
            This would give a good start for basis optimization.
        :param feat: torch.Tensor (M, K), where K = self.get_feature_size()
        :return: None
        """
        assert feat.size(1) == self.get_feature_size()
        for ci in range(len(self.init_vals)):
            feat[:, ci * 3: ci * 3 + 3] = self.init_vals[ci]

    def get_feature_size(self) -> int:
        """
        Get the feature channels needed (i.e. the dimension of $m_k^s$)
        :return: int. feature_size
        """
        return len(self.components) * 3

    def evaluate(self, feat: torch.Tensor, xyz: torch.Tensor, feat_ids: torch.Tensor) -> torch.Tensor:
        """
        :param feat: torch.Tensor (M, K),     the basis feature, i.e. $m_k^s$.
        :param xyz:  torch.Tensor (N, 3),     local coordinates w.r.t. the voxel's center.
        :param feat_ids: torch.Tensor (N, ),  the index (0~M-1) into feat.
        :return: evaluated basis values: torch.Tensor (N, )
        """
        assert feat_ids.size(0) == xyz.size(0), "Input sizes not equal!"
        coeffs = self.feature_to_coefficient(feat)
        funcs = EvalFunc().apply(xyz, coeffs, feat_ids, self.components)
        return funcs[:, 0] * funcs[:, 1] * funcs[:, 2]

    def evaluate_derivative(self, feat: torch.Tensor, xyz: torch.Tensor,
                            feat_ids: torch.Tensor, stride: int) -> torch.Tensor:
        """
        :param feat: torch.Tensor (M, K),     the basis feature, i.e. $m_k^s$.
        :param xyz:  torch.Tensor (N, 3),     local coordinates w.r.t. the voxel's center.
        :param feat_ids: torch.Tensor (N, ),  the index (0~M-1) into feat.
        :param stride: int, denominator of xyz, this is needed for derivative computation.
        :return: evaluated derivative: torch.Tensor (N, 3)
        """
        coeffs = self.feature_to_coefficient(feat)
        funcs, dfuncs = EvalDerivFunc().apply(xyz, coeffs, feat_ids, self.components, stride)
        return torch.stack([dfuncs[:, 0] * funcs[:, 1] * funcs[:, 2],
                            funcs[:, 0] * dfuncs[:, 1] * funcs[:, 2],
                            funcs[:, 0] * funcs[:, 1] * dfuncs[:, 2]], dim=1)

    def integrate_deriv_deriv_product(self, source_feat: torch.Tensor, target_feat: torch.Tensor,
                                      rel_pos: torch.Tensor, source_stride: int, target_stride: int,
                                      source_ids: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
        """
        Compute $integrate_Omega nabla B_source^T nabla B_target$, as appeared in LHS.
        :param source_feat: torch.Tensor (M_source, K), the source basis feature
        :param target_feat: torch.Tensor (M_target, K), the target basis feature
        :param rel_pos:     torch.Tensor (N, 3)
        :param source_stride: int, stride of source voxel
        :param target_stride: int, stride of target voxel
        :param source_ids: torch.Tensor (N, ), the index into source_feat
        :param target_ids: torch.Tensor (N, ), the index into target_feat
        :return: evaluated integral: torch.Tensor (N, )
        """
        assert source_stride <= target_stride, "Query must be smaller than reference."
        assert source_ids.size(0) == target_ids.size(0) == rel_pos.size(0), "Input sizes not equal!"

        mult = target_stride // source_stride
        rel_pos = (rel_pos + mult * 3 / 2 + 0.5).round().long()
        mult = [1, 2, 4, 8, 16].index(mult)

        max_rel_pos = self.ff_integral[mult]['f_f'].size(-1)
        invalid_range_val = ~torch.logical_and(rel_pos >= 0, rel_pos < max_rel_pos)
        rel_pos.clamp_(min=0, max=max_rel_pos - 1)

        source_coeffs = self.feature_to_coefficient(source_feat)
        target_coeffs = self.feature_to_coefficient(target_feat)

        source_expanded = source_coeffs[source_ids]
        target_expanded = target_coeffs[target_ids]

        p_mats, q_mats = DDIFunc().apply(source_expanded, target_expanded, rel_pos,
                                         self.ff_integral[mult]['f_f'].data * source_stride,
                                         self.ff_integral[mult]['df_df'].data / source_stride)

        p_mats[invalid_range_val] = 0.0
        q_mats[invalid_range_val] = 0.0
        res = q_mats[:, 0] * p_mats[:, 1] * p_mats[:, 2] + \
              p_mats[:, 0] * q_mats[:, 1] * p_mats[:, 2] + \
              p_mats[:, 0] * p_mats[:, 1] * q_mats[:, 2]

        return res

    def integrate_const_deriv_product(self, data: torch.Tensor, target_feat: torch.Tensor,
                                      rel_pos: torch.Tensor, data_stride: int, target_stride: int,
                                      target_ids: torch.Tensor) -> torch.Tensor:
        """
        Compute $integrate_Omega nabla B_target^T data$, as appeared in RHS.
        :param data:        torch.Tensor (N, K), the data (N) to be integrated
        :param target_feat: torch.Tensor (M_target, K), the target basis feature
        :param rel_pos:     torch.Tensor (N, 3)
        :param data_stride: int, stride of the data voxel
        :param target_stride: int, stride of target voxel
        :param target_ids: torch.Tensor (N, ), the index into target_feat
        :return: evaluated integral: torch.Tensor (N, )
        """
        assert data.size(0) == target_ids.size(0) == rel_pos.size(0), "Input sizes not equal!"
        if target_stride >= data_stride:
            mult = target_stride // data_stride
            abs_val = (rel_pos + (3 * mult - 1) / 2.).round().long()
            mult = [1, 2, 4, 8, 16].index(mult)
            p_suffix = ""
        else:
            mult = data_stride // target_stride
            abs_val = (rel_pos * mult + mult / 2 + 0.5).round().long()
            mult = [2, 4, 8, 16].index(mult)
            p_suffix = "_inv"
            data_stride, target_stride = target_stride, data_stride

        target_coeffs = self.feature_to_coefficient(target_feat)
        target_expanded = target_coeffs[target_ids]

        p_mats = 0
        q_mats = 0
        for si, ca in enumerate(self.components):
            si3 = si * 3
            p_mats += target_expanded[:, si3: si3 + 3] * \
                      (self.f_integral[si]['f' + p_suffix][mult][abs_val] * data_stride)
            q_mats += target_expanded[:, si3: si3 + 3] * \
                      (self.f_integral[si]['df' + p_suffix][mult][abs_val])

        res = data[:, 0] * q_mats[:, 0] * p_mats[:, 1] * p_mats[:, 2] + \
              data[:, 1] * p_mats[:, 0] * q_mats[:, 1] * p_mats[:, 2] + \
              data[:, 2] * p_mats[:, 0] * p_mats[:, 1] * q_mats[:, 2]
        return res

    @staticmethod
    def get_bezier():
        # Note: this would not contain any parameters
        return AdditiveBasis([Bezier()], [1.0])

    @staticmethod
    def get_poly_6():
        return AdditiveBasis(["Power(0)", "Power(1)", "Power(2)", "Power(3)", "Power(4)", "Power(5)", "Power(6)"],
                             [0.7394, 0.0, -0.6572, 0.0, 0.1461, 0.0, 0.0])
