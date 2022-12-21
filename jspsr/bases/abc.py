import jittor as jt
import torch
from abc import ABC, abstractmethod


class BaseBasis(torch.nn.Module, ABC):
    def __init__(self):
        super().__init__()

    @classmethod
    def get_domain_range(cls) -> int:
        """
        Get the support of the current basis.
        :return: support size in the unit of voxel sizes.
        """
        return 3

    def get_feature_size(self) -> int:
        """
        Get the feature channels needed (i.e. the dimension of $m_k^s$)
        :return: int. feature_size
        """
        return 0

    @abstractmethod
    def initialize_feature_value(self, feat: jt.Var) -> None:
        """
        Initialize the input features, so that the basis behaves like Bezier.
            This would give a good start for basis optimization.
        :param feat: jt.Var (M, K), where K = self.get_feature_size()
        :return: None
        """
        pass

    @abstractmethod
    def evaluate(self, feat: jt.Var, xyz: jt.Var, feat_ids: jt.Var) -> jt.Var:
        """
        :param feat: jt.Var (M, K),     the basis feature, i.e. $m_k^s$.
        :param xyz:  torch.Tensor (N, 3),     local coordinates w.r.t. the voxel's center.
        :param feat_ids: jt.Var (N, ),  the index (0~M-1) into feat.
        :return: evaluated basis values: jt.Var (N, )
        """
        pass

    @abstractmethod
    def evaluate_derivative(self, feat: jt.Var, xyz: jt.Var,
                            feat_ids: jt.Var, stride: int) -> jt.Var:
        """
        :param feat: jt.Var (M, K),     the basis feature, i.e. $m_k^s$.
        :param xyz:  torch.Tensor (N, 3),     local coordinates w.r.t. the voxel's center.
        :param feat_ids: jt.Var (N, ),  the index (0~M-1) into feat.
        :param stride: int, denominator of xyz, this is needed for derivative computation.
        :return: evaluated derivative: jt.Var (N, 3)
        """
        pass

    @abstractmethod
    def integrate_deriv_deriv_product(self, source_feat: jt.Var, target_feat: jt.Var,
                                      rel_pos: jt.Var, source_stride: int, target_stride: int,
                                      source_ids: jt.Var, target_ids: jt.Var) -> jt.Var:
        """
        Compute $integrate_Omega nabla B_source^T nabla B_target$, as appeared in LHS.
        :param source_feat: jt.Var (M_source, K), the source basis feature
        :param target_feat: jt.Var (M_target, K), the target basis feature
        :param rel_pos:     torch.Tensor (N, 3)
        :param source_stride: int, stride of source voxel
        :param target_stride: int, stride of target voxel
        :param source_ids: jt.Var (N, ), the index into source_feat
        :param target_ids: jt.Var (N, ), the index into target_feat
        :return: evaluated integral: jt.Var (N, )
        """
        pass

    @abstractmethod
    def integrate_const_deriv_product(self, data: jt.Var, target_feat: jt.Var,
                                      rel_pos: jt.Var, data_stride: int, target_stride: int,
                                      target_ids: jt.Var) -> jt.Var:
        """
        Compute $integrate_Omega nabla B_target^T data$, as appeared in RHS.
        :param data:        torch.Tensor (N, K), the data (N) to be integrated
        :param target_feat: jt.Var (M_target, K), the target basis feature
        :param rel_pos:     torch.Tensor (N, 3)
        :param data_stride: int, stride of the data voxel
        :param target_stride: int, stride of target voxel
        :param target_ids: jt.Var (N, ), the index into target_feat
        :return: evaluated integral: jt.Var (N, )
        """
        pass
