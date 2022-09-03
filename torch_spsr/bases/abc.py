import torch
from abc import ABC, abstractmethod


class BaseBasis(torch.nn.Module, ABC):
    def __init__(self):
        super().__init__()

    @classmethod
    def get_domain_range(cls):
        """
        Get the support of the current basis.
        :return: support size in the unit of voxel sizes.
        """
        return 3

    @classmethod
    @abstractmethod
    def get_feature_size(cls):
        """
        Get the feature channels needed (i.e. the dimension of $m_k^s$)
        :return: int. feature_size
        """
        pass

    @abstractmethod
    def initialize_feature_value(self, feat: torch.Tensor):
        """
        Initialize the input features, so that the basis behaves like Bezier.
            This would give a good start for basis optimization.
        :param feat: torch.Tensor (M, K), where K = self.get_feature_size()
        :return: None
        """
        pass

    @abstractmethod
    def evaluate(self, feat: torch.Tensor, xyz: torch.Tensor, feat_ids: torch.Tensor):
        """
        :param feat: torch.Tensor (M, K),     the basis feature, i.e. $m_k^s$.
        :param xyz:  torch.Tensor (N, 3),     local coordinates w.r.t. the voxel's center.
        :param feat_ids: torch.Tensor (N, ),  the index (0~M-1) into feat.
        :return: evaluated basis values: torch.Tensor (N, )
        """
        pass

    @abstractmethod
    def evaluate_derivative(self, feat: torch.Tensor, xyz: torch.Tensor, feat_ids: torch.Tensor, stride: int):
        """
        :param feat: torch.Tensor (M, K),     the basis feature, i.e. $m_k^s$.
        :param xyz:  torch.Tensor (N, 3),     local coordinates w.r.t. the voxel's center.
        :param feat_ids: torch.Tensor (N, ),  the index (0~M-1) into feat.
        :param stride: int, denominator of xyz, this is needed for derivative computation.
        :return: evaluated derivative: torch.Tensor (N, 3)
        """
        pass

    @abstractmethod
    def integrate_deriv_deriv_product(self, source_feat: torch.Tensor, target_feat: torch.Tensor,
                                      rel_pos: torch.Tensor, source_stride: int, target_stride: int,
                                      source_ids: torch.Tensor, target_ids: torch.Tensor):
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
        pass

    @abstractmethod
    def integrate_const_deriv_product(self, data: torch.Tensor, target_feat: torch.Tensor,
                                      rel_pos: torch.Tensor, data_stride: int, target_stride: int,
                                      target_ids: torch.Tensor):
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
        pass
