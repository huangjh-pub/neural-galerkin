import torch
import numpy as np
from scipy.linalg import null_space
from jspsr.bases.additive import AdditiveBasis, Power


def _get_ns(n_degrees):
    """
    Null space satisfying: b(-1.5) = grad b (-1.5) = b(1.5) = grad b (1.5) = 0.
    """
    coef_mat = np.zeros((4, n_degrees + 1))
    for i in range(n_degrees + 1):
        deg = n_degrees - i
        coef_mat[0, i] = (3 / 2) ** deg
        coef_mat[1, i] = (-3 / 2) ** deg
        if i <= n_degrees:
            coef_mat[2, i] = deg * (3 / 2) ** (deg - 1)
            coef_mat[3, i] = deg * (-3 / 2) ** (deg - 1)
    ns = null_space(coef_mat)[::-1].T
    return ns


class AdditiveHermite(AdditiveBasis):
    """
        Addition of polynomial and other elementary functions, where the coeffs of the polynomials
    lie within the Null space of boundary + hermite constraint:
        b(-1.5) = grad b (-1.5) = b(1.5) = grad b (1.5) = 0.
    """
    def __init__(self, n_degrees, components, init_vals):
        assert n_degrees > 3, "Degrees <= cubic does not admit two hermite constraints!"

        super().__init__(
            list(components) + [Power(p) for p in range(n_degrees + 1)],
            list(init_vals) + [0.0 for _ in range(n_degrees + 1)]
        )
        self.n_degrees = n_degrees
        ns = _get_ns(self.n_degrees).copy()
        ns = torch.from_numpy(ns).float()
        ns = torch.kron(ns, torch.eye(3))
        self.ns = torch.nn.Parameter(ns, requires_grad=False)
        self.n_acomp = len(components)

    def feature_to_coefficient(self, feat: torch.Tensor) -> torch.Tensor:
        """
        Convert feature that describes the basis to real coefficients of the components.
        :param feat: torch.Tensor (N, x) feature
        :return: torch.Tensor (N, 3*len(self.components)) coefficients (for x,y,z axis)
        """
        return torch.cat([
            feat[:, :self.n_acomp * 3], feat[:, self.n_acomp * 3:] @ self.ns
        ], dim=1)

    def initialize_feature_value(self, feat: torch.Tensor) -> None:
        """
        Initialize the input features, so that the basis behaves like Bezier.
            This would give a good start for basis optimization.
        :param feat: torch.Tensor (M, K), where K = self.get_feature_size()
        :return: None
        """
        assert feat.size(1) == self.get_feature_size()
        for ci in range(self.n_acomp):
            feat[:, ci * 3: ci * 3 + 3] = self.init_vals[ci]

        ns4 = _get_ns(4)
        ns_my = _get_ns(self.n_degrees)
        initial_coeffs = np.zeros((self.n_degrees + 1, ))
        initial_coeffs[0] = ns4[0, 0]
        initial_coeffs[2] = ns4[0, 2]
        initial_coeffs[4] = ns4[0, 4]
        initial_feat = np.linalg.lstsq(ns_my.T, initial_coeffs, rcond=None)[0]
        base = self.n_acomp * 3
        for deg in range(initial_feat.shape[0]):
            feat[:, base + deg * 3: base + deg * 3 + 3] = initial_feat[deg]

    def get_feature_size(self) -> int:
        """
        Get the feature channels needed (i.e. the dimension of $m_k^s$)
        :return: int. feature_size
        """
        return self.n_acomp * 3 + self.ns.size(0)
