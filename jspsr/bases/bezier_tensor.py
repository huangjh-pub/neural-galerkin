import math
from pathlib import Path
import pickle

import torch

from jspsr.bases.abc import BaseBasis
import jittor as jt


class BezierTensorBasis(BaseBasis):
    """
        Centered at the voxel center (0), this bezier tensor-product is an axis-separable basis,
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

    @classmethod
    def _initialize_constants(cls):
        ni_path = Path(__file__).parent.parent / "data" / "bezier_integrals.pkl"
        assert ni_path.exists(), f"Missing {ni_path.name}, Please check whether installation is complete."
        with ni_path.open("rb") as f:
            cls.PARTIAL_INTEGRAL, cls.DERIVATIVE_PARTIAL_INTEGRAL, \
                cls.INV_PARTIAL_INTEGRAL, cls.INV_DERIVATIVE_PARTIAL_INTEGRAL, \
                cls.SHIFTED_SELF_INTEGRAL, cls.SHIFTED_DERIVATIVE_INTEGRAL, \
                cls.SHIFTED_SELF_DERIV_INTEGRAL, cls.INV_SHIFTED_SELF_DERIV_INTEGRAL = pickle.load(f)

    @classmethod
    def evaluate_single(cls, x: jt.Var):
        b1 = (x + 1.5) ** 2
        b2 = -2 * (x ** 2) + 1.5
        b3 = (x - 1.5) ** 2
        m1 = (x >= -1.5) & (x < -0.5)
        m2 = (x >= -0.5) & (x < 0.5)
        m3 = (x >= 0.5) & (x < 1.5)
        return m1 * b1 + m2 * b2 + m3 * b3

    @classmethod
    def evaluate_derivative_single(cls, x: jt.Var):
        b1 = 2 * x + 3
        b2 = -4 * x
        b3 = 2 * x - 3
        m1 = (x >= -1.5) & (x < -0.5)
        m2 = (x >= -0.5) & (x < 0.5)
        m3 = (x >= 0.5) & (x < 1.5)
        return m1 * b1 + m2 * b2 + m3 * b3

    def evaluate(self, feat: jt.Var, xyz: jt.Var, feat_ids: jt.Var):
        """jt.Var"""
        x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
        bx = self.evaluate_single(x)
        by = self.evaluate_single(y)
        bz = self.evaluate_single(z)
        return bx * by * bz

    def evaluate_derivative(self, feat: jt.Var, xyz: jt.Var, feat_ids: jt.Var, stride: int):
        """jt.Var"""
        x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
        bx = self.evaluate_single(x)
        by = self.evaluate_single(y)
        bz = self.evaluate_single(z)
        dx = self.evaluate_derivative_single(x) / stride
        dy = self.evaluate_derivative_single(y) / stride
        dz = self.evaluate_derivative_single(z) / stride
        return torch.stack([dx * by * bz, bx * dy * bz, bx * by * dz], dim=1)

    def integrate_deriv_deriv_product(self, rel_pos: jt.Var, source_stride: int, target_stride: int, **kwargs):
        """jt.Var"""
        assert source_stride <= target_stride, "Source stride cannot be larger than target stride."

        mult = target_stride // source_stride
        abs_val = (rel_pos + mult * 3 / 2 + 0.5).round().long()
        mult = int(math.log2(mult))

        invalid_range_val = ~torch.logical_and(abs_val >= 0, abs_val < len(self.SHIFTED_SELF_INTEGRAL[mult]))
        abs_val.clamp_(min=0, max=len(self.SHIFTED_SELF_INTEGRAL[mult]) - 1)

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

    def integrate_const_deriv_product(self, data: jt.Var, rel_pos: jt.Var,
                                      data_stride: int, target_stride: int, **kwargs):
        """jt.Var"""
        if target_stride >= data_stride:
            mult = target_stride // data_stride
            abs_val = (rel_pos + (3 * mult - 1) / 2.).round().long()
            mult = int(math.log2(mult))
            dpi, pi = self.DERIVATIVE_PARTIAL_INTEGRAL[mult], self.PARTIAL_INTEGRAL[mult]
            i_mult = data_stride / target_stride
        else:
            mult = data_stride // target_stride
            abs_val = (rel_pos * mult + mult / 2 + 0.5).round().long()
            mult = int(math.log2(mult)) - 1
            dpi, pi = self.INV_DERIVATIVE_PARTIAL_INTEGRAL[mult], self.INV_PARTIAL_INTEGRAL[mult]
            data_stride, target_stride = target_stride, data_stride
            i_mult = 1.0

        const_val = torch.tensor(dpi, dtype=torch.float, device=rel_pos.device)[abs_val] * data_stride
        integral_val = torch.tensor(pi, dtype=torch.float, device=rel_pos.device)[abs_val] * i_mult
        res = data[:, 0] * integral_val[:, 0] * const_val[:, 1] * const_val[:, 2] + \
            data[:, 1] * const_val[:, 0] * integral_val[:, 1] * const_val[:, 2] + \
            data[:, 2] * const_val[:, 0] * const_val[:, 1] * integral_val[:, 2]

        return res

    def initialize_feature_value(self, feat: jt.Var) -> None:
        pass


# Load precomputed integral values
BezierTensorBasis._initialize_constants()
