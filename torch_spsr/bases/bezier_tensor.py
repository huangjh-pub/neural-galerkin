import torch
import os
import math
import numpy as np
import sympy as sp
import pickle

from torch_spsr.bases.abc import BaseBasis


class BezierTensorBasis(BaseBasis):
    """
        Centered at the voxel center (0), this bezier tensor-product is a dimension-separable basis,
    where each dim takes the form:
            B^k = (B^0)^(*k), where ^(*k) is convolution with itself k times.
            B^0 is a box function, being 1 when |x|<0.5, and 0 otherwise.
        We implement the one used in [Kazhdan et al., 06], that chooses k=2, expressed as:
            B^2(x) = 0,                           x < -1.5
                     (x + 1.5)^2,          -1.5 <= x < -0.5
                     -2x^2 + 1.5,          -0.5 <= x < 0.5
                     (x - 1.5)^2,           0.5 <= x < 1.5
                     0.                     1.5 <= x
        * We scale that up by 2 but this should not matter a lot.
    """
    INTEGRAL_CACHE_PATH = "../dp-data/bases/bezier.pkl"

    @staticmethod
    def _initialize_constants(max_mult):
        ni_path = BezierTensorBasis.INTEGRAL_CACHE_PATH
        if os.path.exists(ni_path):
            with open(ni_path, 'rb') as f:
                BezierTensorBasis.PARTIAL_INTEGRAL, BezierTensorBasis.DERIVATIVE_PARTIAL_INTEGRAL, \
                BezierTensorBasis.INV_PARTIAL_INTEGRAL, BezierTensorBasis.INV_DERIVATIVE_PARTIAL_INTEGRAL, \
                BezierTensorBasis.SHIFTED_SELF_INTEGRAL, BezierTensorBasis.SHIFTED_DERIVATIVE_INTEGRAL, \
                BezierTensorBasis.SHIFTED_SELF_DERIV_INTEGRAL, BezierTensorBasis.INV_SHIFTED_SELF_DERIV_INTEGRAL \
                    = pickle.load(f)
            return

        x = sp.Symbol('x')
        bfunc_x = sp.Piecewise(
            (0, x < -1.5), ((x + 1.5) ** 2, x < -0.5), (-2 * x ** 2 + 1.5, x < 0.5),
            ((x - 1.5) ** 2, x < 1.5), (0, True)
        )
        dbfunc_x = sp.diff(bfunc_x, x)

        partial_integrals, derivative_partial_integrals = [], []
        for multiple in range(max_mult + 1):
            print(f"Computing [1/4] of {multiple}...")
            stride = 2 ** multiple
            wb_func = bfunc_x.subs(x, x / stride)
            wdb_func = dbfunc_x.subs(x, x / stride)
            pi, dpi = [], []
            for t in np.linspace(-1.5 * stride, 1.5 * stride - 1, 3 * stride):
                pi.insert(0, float(sp.integrate(wdb_func, (x, t, t + 1.0))))
                dpi.insert(0, float(sp.integrate(wb_func, (x, t, t + 1.0))))
            partial_integrals.append(pi)
            derivative_partial_integrals.append(dpi)

        inv_partial_integrals, inv_derivative_partial_integrals = [], []
        for multiple in range(1, max_mult + 1):
            print(f"Computing [2/4] of {multiple}...")
            data_length = 2 ** multiple
            ipi, idpi = [], []
            for ep in np.arange(-0.5, data_length + 1.5):
                stp = ep - data_length
                ipi.insert(0, float(sp.integrate(dbfunc_x, (x, stp, ep))))
                idpi.insert(0, float(sp.integrate(bfunc_x, (x, stp, ep))))
            inv_partial_integrals.append(ipi)
            inv_derivative_partial_integrals.append(idpi)

        shifted_self_integral, shifted_derivative_integral, shifted_self_deriv_integral = [], [], []
        for multiple in range(max_mult + 1):
            print(f"Computing [3/4] of {multiple}...")
            stride = 2 ** multiple
            bfunc_wx = bfunc_x.subs(x, x / stride)
            dbfunc_wx = dbfunc_x.subs(x, x / stride)
            ssi, sdi, ssdi = [], [], []
            for dx in np.arange(-1.5 * stride - 0.5, 1.5 * stride + 1.5):
                integrand_x = float(sp.integrate(bfunc_wx * bfunc_x.subs(x, x + dx), (x, -1000., 1000.)))
                integrand_dx = float(sp.integrate(dbfunc_wx * dbfunc_x.subs(x, x + dx), (x, -1000., 1000.)))
                integrand_xdx = float(sp.integrate(dbfunc_wx * bfunc_x.subs(x, x + dx), (x, -1000., 1000.)))
                ssi.append(integrand_x)
                sdi.append(integrand_dx)
                ssdi.append(integrand_xdx)
            shifted_self_integral.append(ssi)
            shifted_derivative_integral.append(sdi)
            shifted_self_deriv_integral.append(ssdi)

        inv_shifted_self_deriv_integral = []
        for multiple in range(1, max_mult + 1):
            print(f"Computing [4/4] of {multiple}...")
            stride = 2 ** multiple
            bfunc_wx = bfunc_x.subs(x, x / stride)
            issdi = []
            for dx in np.arange(-1.5 * stride - 0.5, 1.5 * stride + 1.5):
                integrand_xdx = float(sp.integrate(dbfunc_x * bfunc_wx.subs(x, x + dx), (x, -1000., 1000.)))
                issdi.append(integrand_xdx)
            inv_shifted_self_deriv_integral.append(issdi)

        with open(ni_path, 'wb') as f:
            pickle.dump([partial_integrals, derivative_partial_integrals,
                         inv_partial_integrals, inv_derivative_partial_integrals,
                         shifted_self_integral, shifted_derivative_integral, shifted_self_deriv_integral,
                         inv_shifted_self_deriv_integral], f)
        BezierTensorBasis._initialize_constants(max_mult)

    @staticmethod
    def evaluate_single(x: torch.Tensor):
        b1 = (x + 1.5) ** 2
        b2 = -2 * (x ** 2) + 1.5
        b3 = (x - 1.5) ** 2
        m1 = (x >= -1.5) & (x < -0.5)
        m2 = (x >= -0.5) & (x < 0.5)
        m3 = (x >= 0.5) & (x < 1.5)
        return m1 * b1 + m2 * b2 + m3 * b3

    @staticmethod
    def evaluate_derivative_single(x: torch.Tensor):
        b1 = 2 * x + 3
        b2 = -4 * x
        b3 = 2 * x - 3
        m1 = (x >= -1.5) & (x < -0.5)
        m2 = (x >= -0.5) & (x < 0.5)
        m3 = (x >= 0.5) & (x < 1.5)
        return m1 * b1 + m2 * b2 + m3 * b3

    def evaluate(self, feat: torch.Tensor, xyz: torch.Tensor, feat_ids: torch.Tensor):
        # (..., 3) in normalized coordinates -> (..., )
        x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
        bx = self.evaluate_single(x)
        by = self.evaluate_single(y)
        bz = self.evaluate_single(z)
        return bx * by * bz

    def evaluate_derivative(self, feat: torch.Tensor, xyz: torch.Tensor, feat_ids: torch.Tensor, stride: int):
        x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
        bx = self.evaluate_single(x)
        by = self.evaluate_single(y)
        bz = self.evaluate_single(z)
        dx = self.evaluate_derivative_single(x) / stride
        dy = self.evaluate_derivative_single(y) / stride
        dz = self.evaluate_derivative_single(z) / stride
        return torch.stack([dx * by * bz, bx * dy * bz, bx * by * dz], dim=1)

    def integrate_deriv_deriv_product(self, rel_pos: torch.Tensor, source_stride: int, target_stride: int, **kwargs):
        # For now we assume rel_pos are unit interval relative to source (Nx3)
        #   i.e. source-basis + rel_pos = target-basis
        assert source_stride <= target_stride, "Query must be smaller than reference."
        mult = target_stride // source_stride
        abs_val = (rel_pos + mult * 3 / 2 + 0.5).round().long()
        mult = int(math.log2(mult))

        invalid_range_val = ~torch.logical_and(abs_val >= 0, abs_val < len(self.SHIFTED_SELF_INTEGRAL[mult]))
        abs_val.clamp_(min=0, max=len(self.SHIFTED_SELF_INTEGRAL[mult]) - 1)

        # print(np.array(self.SHIFTED_SELF_INTEGRAL[mult]) * source_stride)
        # print(np.array(self.SHIFTED_DERIVATIVE_INTEGRAL[mult]) / target_stride)

        const_val = torch.tensor(self.SHIFTED_SELF_INTEGRAL[mult],
                                 dtype=torch.float, device=rel_pos.device)[abs_val] * source_stride
        const_val[invalid_range_val] = 0.0
        integral_val = torch.tensor(self.SHIFTED_DERIVATIVE_INTEGRAL[mult],
                                    dtype=torch.float, device=rel_pos.device)[abs_val] / target_stride
        integral_val[invalid_range_val] = 0.0
        res = integral_val[:, 0] * const_val[:, 1] * const_val[:, 2] + \
              const_val[:, 0] * integral_val[:, 1] * const_val[:, 2] + \
              const_val[:, 0] * const_val[:, 1] * integral_val[:, 2]

        return res

    def integrate_basis_deriv_product(self, data: torch.Tensor, rel_pos: torch.Tensor, data_stride: int,
                                      target_stride: int, **kwargs):
        # Signature follows 'integrate_const_deriv_product'
        if target_stride >= data_stride:
            mult = target_stride // data_stride
            abs_val = (rel_pos + (3 * mult + 1) / 2.).round().long()
            mult = int(math.log2(mult))
            dpi, pi = self.SHIFTED_SELF_INTEGRAL[mult], self.SHIFTED_SELF_DERIV_INTEGRAL[mult]
        else:
            mult = data_stride // target_stride
            abs_val = (rel_pos * mult + (3 * mult + 1) / 2.).round().long()  # Same
            mult = int(math.log2(mult)) - 1

            # Turns out that self-integral is symmetric.
            dpi, pi = self.SHIFTED_SELF_INTEGRAL[1:][mult], self.INV_SHIFTED_SELF_DERIV_INTEGRAL[mult]
            data_stride, target_stride = target_stride, data_stride

        invalid_range_val = ~torch.logical_and(abs_val >= 0, abs_val < len(dpi))
        abs_val.clamp_(min=0, max=len(dpi) - 1)

        const_val = torch.tensor(dpi, dtype=torch.float, device=rel_pos.device)[abs_val] * data_stride
        const_val[invalid_range_val] = 0.0

        integral_val = torch.tensor(pi, dtype=torch.float, device=rel_pos.device)[abs_val] * data_stride / target_stride
        integral_val[invalid_range_val] = 0.0

        res = data[:, 0] * integral_val[:, 0] * const_val[:, 1] * const_val[:, 2] + \
              data[:, 1] * const_val[:, 0] * integral_val[:, 1] * const_val[:, 2] + \
              data[:, 2] * const_val[:, 0] * const_val[:, 1] * integral_val[:, 2]

        return res

    def integrate_const_deriv_product(self, data: torch.Tensor, rel_pos: torch.Tensor,
                                      data_stride: int, target_stride: int, **kwargs):
        # rel_pos: (Nx3), data: (Nx3)
        # data takes the region of data_stride (1x1x1), and is rel_pos to this basis.
        #   i.e. data + rel_pos = this-basis
        if target_stride >= data_stride:
            mult = target_stride // data_stride
            abs_val = (rel_pos + (3 * mult - 1) / 2.).round().long()
            mult = int(math.log2(mult))
            dpi, pi = self.DERIVATIVE_PARTIAL_INTEGRAL[mult], self.PARTIAL_INTEGRAL[mult]
            i_mult = data_stride / target_stride

            # print("pos", data_stride, target_stride)
            # print(np.array(self.DERIVATIVE_PARTIAL_INTEGRAL[mult]) * data_stride)
            # print(np.array(self.PARTIAL_INTEGRAL[mult]) * data_stride / target_stride)
        else:
            mult = data_stride // target_stride
            abs_val = (rel_pos * mult + mult / 2 + 0.5).round().long()
            mult = int(math.log2(mult)) - 1
            dpi, pi = self.INV_DERIVATIVE_PARTIAL_INTEGRAL[mult], self.INV_PARTIAL_INTEGRAL[mult]
            data_stride, target_stride = target_stride, data_stride
            i_mult = 1.0

            # print("inv", data_stride, target_stride)
            # print(np.array(self.INV_DERIVATIVE_PARTIAL_INTEGRAL[mult]) * data_stride)
            # print(np.array(self.INV_PARTIAL_INTEGRAL[mult]))

        const_val = torch.tensor(dpi, dtype=torch.float, device=rel_pos.device)[abs_val] * data_stride
        integral_val = torch.tensor(pi, dtype=torch.float, device=rel_pos.device)[abs_val] * i_mult
        res = data[:, 0] * integral_val[:, 0] * const_val[:, 1] * const_val[:, 2] + \
              data[:, 1] * const_val[:, 0] * integral_val[:, 1] * const_val[:, 2] + \
              data[:, 2] * const_val[:, 0] * const_val[:, 1] * integral_val[:, 2]

        return res


BezierTensorBasis._initialize_constants(7)
