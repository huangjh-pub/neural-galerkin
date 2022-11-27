import torch
import torch.nn as nn
from torch_scatter import scatter_mean, scatter_max
from models.ascn.utils import get_embedder, ResnetBlockFC
from torch_spsr.core.hashtree import HashTree


class LocalPoolPointnet(nn.Module):
    """ PointNet-based encoder network with ResNet blocks for each point.
        Number of input points are fixed.

    Args:
        c_dim (int): dimension of latent code c
        dim (int): input points dimension
        hidden_dim (int): hidden dimension of the network
        scatter_type (str): feature aggregation when doing local pooling
        n_blocks (int): number of blocks ResNetBlockFC layers
        map2local (function): map global coordintes to local ones
        pos_encoding (int): frequency for the positional encoding
    """

    def __init__(self, c_dim=32, dim=3, hidden_dim=32, scatter_type='max', n_blocks=5,
                 map2local=False, pos_encoding=0):
        super().__init__()

        self.c_dim = c_dim
        self.map2local = map2local

        if pos_encoding > 0:
            embed_fn, input_ch = get_embedder(pos_encoding, d_in=dim)
            self.pe = embed_fn
            self.fc_pos = nn.Linear(input_ch, 2 * hidden_dim)
        else:
            self.pe = None
            self.fc_pos = nn.Linear(dim, 2 * hidden_dim)

        self.blocks = nn.ModuleList([
            ResnetBlockFC(2 * hidden_dim, hidden_dim) for _ in range(n_blocks)
        ])
        self.fc_c = nn.Linear(hidden_dim, c_dim)

        self.hidden_dim = hidden_dim

        if scatter_type == 'max':
            self.scatter = scatter_max
        elif scatter_type == 'mean':
            self.scatter = scatter_mean
        else:
            raise ValueError('incorrect scatter type')

    def forward(self, feat, hash_tree: HashTree):
        xyz = hash_tree.xyz

        # Compute pts-to-voxel mapping.
        pids, vids, _, _ = hash_tree.get_neighbours_data(
            torch.div(xyz, hash_tree.voxel_size, rounding_mode='floor').int(), source_stride=1, target_depth=0,
            nn_kernel=hash_tree.get_range_kernel(1), branch=hash_tree.ENCODER, conv_based=True)
        xyz = xyz[pids]

        if self.map2local:
            xyz = (xyz % hash_tree.voxel_size) / hash_tree.voxel_size

        # Currently, not used.
        if self.pe is not None:
            xyz = self.pe(xyz)

        if feat is None:
            feat = self.fc_pos(xyz)
        else:
            feat = feat[pids]
            feat = self.fc_pos(torch.cat([xyz, feat], dim=1))

        feat = self.blocks[0](feat)
        for block in self.blocks[1:]:
            pooled = self.scatter(feat, vids, dim=0, dim_size=hash_tree.get_coords_size(hash_tree.ENCODER, 0))
            if self.scatter == scatter_max:
                pooled = pooled[0]
            pooled = pooled[vids]
            feat = torch.cat([feat, pooled], dim=1)
            feat = block(feat)

        c = self.fc_c(feat)
        c = scatter_mean(c, vids, dim=0, dim_size=hash_tree.get_coords_size(hash_tree.ENCODER, 0))

        return c
