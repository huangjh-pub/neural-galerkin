import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_spsr.core.hashtree import HashTree
from models.ascn.utils import ResnetBlockFC


class MultiscaleDecoder(nn.Module):
    def __init__(self, p_dim=3, c_dim=128, out_dim=3, hidden_size=256, n_blocks=5, leaky=False,
                 multiscale_depths=3, output_normalize=False, out_init=None):
        super().__init__()
        self.c_dim = c_dim
        self.n_blocks = n_blocks
        self.multiscale_depths = multiscale_depths

        self.fc_c = nn.ModuleList([nn.Linear(c_dim, hidden_size) for _ in range(n_blocks)])
        self.fc_p = nn.Linear(p_dim, hidden_size)
        self.blocks = nn.ModuleList([
            ResnetBlockFC(hidden_size) for _ in range(n_blocks)
        ])
        self.fc_out = nn.Linear(hidden_size, out_dim)

        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

        self.out_dim = out_dim
        self.output_normalize = output_normalize

        # Init parameters
        if out_init is not None:
            assert 0.0 <= out_init <= 1.0, "out_init is after sigmoid!"
            assert self.output_normalize
            nn.init.zeros_(self.fc_out.weight)
            b = np.log(out_init / (1 - out_init))
            nn.init.constant_(self.fc_out.bias, b)

    def forward(self, hash_tree: HashTree, multiscale_feat: dict, branch: int):
        p_feats = []
        for did in range(self.multiscale_depths):
            vs = hash_tree.get_stride(branch, did) * hash_tree.voxel_size
            p = (hash_tree.xyz % vs) / vs
            p_feats.append(p)
        p = torch.cat(p_feats, dim=1)

        c_feats = []
        for did in range(self.multiscale_depths):
            c = hash_tree.split_data(hash_tree.xyz, branch, did, multiscale_feat[did])  # (N, C)
            c_feats.append(c)
        c = torch.cat(c_feats, dim=1)

        net = self.fc_p(p)
        for i in range(self.n_blocks):
            net = net + self.fc_c[i](c)
            net = self.blocks[i](net)
        out = self.fc_out(self.actvn(net))

        if self.output_normalize:
            out = torch.sigmoid(out)

        return out


class LocalDecoder(nn.Module):
    ''' Decoder.
        Instead of conditioning on global features, on plane/volume local features.
    Args:
        dim (int): input dimension
        c_dim (int): dimension of latent conditioned code c
        hidden_size (int): hidden size of Decoder network
        n_blocks (int): number of blocks ResNetBlockFC layers
        leaky (bool): whether to use leaky ReLUs
    '''

    def __init__(self, dim=3, c_dim=128, out_dim=3, hidden_size=256, n_blocks=5, leaky=False, map2local=False):
        super().__init__()
        self.c_dim = c_dim
        self.n_blocks = n_blocks

        self.fc_c = nn.ModuleList([
            nn.Linear(c_dim, hidden_size) for _ in range(n_blocks)
        ])

        self.fc_p = nn.Linear(dim, hidden_size)
        self.blocks = nn.ModuleList([
            ResnetBlockFC(hidden_size) for _ in range(n_blocks)
        ])
        self.fc_out = nn.Linear(hidden_size, out_dim)

        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

        self.map2local = map2local
        self.out_dim = out_dim

    def forward(self, hash_tree: HashTree, feat: torch.Tensor, feat_depth: int):
        p = hash_tree.xyz
        c = hash_tree.split_data(p, hash_tree.DECODER, feat_depth, feat)  # (N, C)

        if self.map2local:
            p = (p % hash_tree.voxel_size) / hash_tree.voxel_size

        net = self.fc_p(p)
        for i in range(self.n_blocks):
            net = net + self.fc_c[i](c)
            net = self.blocks[i](net)
        out = self.fc_out(self.actvn(net))

        return out
