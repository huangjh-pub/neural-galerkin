import math
from enum import Enum

import numpy as np
import torch
import torch.nn as nn
import torch_scatter

from models.ascn.utils import ConvolutionFunction
from torch_spsr.core.hashtree import HashTree
from torch_spsr.core.ops import torch_unique


class Activation(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, hash_tree: HashTree, feat: torch.Tensor, feat_depth: int):
        feat = self.module(feat)
        return feat, feat_depth


class GroupNorm(nn.GroupNorm):
    def forward(self, hash_tree: HashTree, feat: torch.Tensor, feat_depth: int):
        num_channels = feat.size(1)
        feat = feat.transpose(0, 1).reshape(1, num_channels, -1)
        feat = super().forward(feat)
        feat = feat.reshape(num_channels, -1).transpose(0, 1)
        return feat, feat_depth


class NearestUpsampling(nn.Module):
    def __init__(self, target_branch: int = HashTree.DECODER):
        super().__init__()
        self.target_branch = target_branch
        assert self.target_branch in [HashTree.DECODER, HashTree.DECODER_TMP]

    def forward(self, hash_tree: HashTree, feat: torch.Tensor, feat_depth: int, mask: torch.Tensor = None):
        assert feat.size(0) == hash_tree.get_coords_size(hash_tree.DECODER, feat_depth)

        if self.target_branch == hash_tree.DECODER_TMP:
            # Create conforming structure...
            base_coords = hash_tree.get_coords(hash_tree.DECODER, feat_depth)[mask]
            conform_offsets = torch.tensor(
                [(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1),
                 (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)],
                dtype=torch.int32, device=feat.device) * hash_tree.get_stride(hash_tree.DECODER, feat_depth - 1)
            hash_tree.update_coords(
                hash_tree.DECODER_TMP, feat_depth - 1,
                (base_coords.unsqueeze(dim=1).repeat(1, 8, 1) + conform_offsets.unsqueeze(0)).view(-1, 3))
            output_feats = feat[mask].unsqueeze(dim=1).repeat(1, 8, 1).view(-1, feat.size(1))
        else:
            # We already have that coordinates, use that.
            src_ids, tgt_ids, ntypes, _ = hash_tree.get_neighbours(
                feat_depth, feat_depth - 1, target_range=3, branch=hash_tree.DECODER, conv_based=True)
            p_masks = torch.all(ntypes >= 0, dim=1)
            # Should be only one value, so pure scatter also works
            output_feats = torch_scatter.scatter_mean(
                feat[src_ids[p_masks]], tgt_ids[p_masks], dim=0,
                dim_size=hash_tree.get_coords_size(hash_tree.DECODER, feat_depth - 1))

        return output_feats, feat_depth - 1


class MaxPooling(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hash_tree: HashTree, feat: torch.Tensor, feat_depth: int):
        assert feat.size(0) == hash_tree.get_coords_size(hash_tree.ENCODER, feat_depth), \
            "This layer can only be used in encoder coordinates!"
        src_ids, tgt_ids, ntypes, _ = hash_tree.get_neighbours(
            feat_depth + 1, feat_depth, target_range=3, branch=hash_tree.ENCODER, conv_based=True)
        p_masks = torch.all(ntypes >= 0, dim=1)
        output_feats = torch_scatter.scatter_max(
            feat[tgt_ids[p_masks]], src_ids[p_masks], dim=0,
            dim_size=hash_tree.get_coords_size(hash_tree.ENCODER, feat_depth + 1))[0]
        return output_feats, feat_depth + 1


class CoordinateTransform(nn.Module):
    def __init__(self, src_branch: int, tgt_branch: int):
        super().__init__()
        self.src_branch = src_branch
        self.tgt_branch = tgt_branch

    def forward(self, hash_tree: HashTree, src_feat: torch.Tensor, feat_depth: int):
        assert src_feat.size(0) == hash_tree.get_coords_size(self.src_branch, feat_depth)

        if hash_tree.is_reflected and self.src_branch == HashTree.ENCODER and self.tgt_branch == HashTree.DECODER:
            return src_feat, feat_depth

        enc_aligned = torch.zeros(
            (hash_tree.get_coords_size(self.tgt_branch, feat_depth), src_feat.size(1)),
            device=src_feat.device, dtype=src_feat.dtype)
        # No cache as this happens very seldom.
        e_idx, d_idx, _, _ = hash_tree.get_neighbours_data(
            hash_tree.get_coords(self.src_branch, feat_depth),
            hash_tree.get_stride(self.src_branch, feat_depth),
            feat_depth, hash_tree.get_range_kernel(1), self.tgt_branch, conv_based=True)
        enc_aligned[d_idx] = src_feat[e_idx]
        return enc_aligned, feat_depth


class CoordinateDensification(nn.Module):

    class DenseType(Enum):
        UNCHANGED = 0       # i.e. SPARSE_EXPAND_1
        SPARSE_EXPAND_3 = 1
        SPARSE_EXPAND_5 = 2
        FULL_EXPAND_1 = 3
        FULL_EXPAND_3 = 4

    def __init__(self, strategy: DenseType):
        super().__init__()
        self.strategy = strategy

    def forward(self, coords: torch.Tensor, stride: int):
        if self.strategy == self.DenseType.UNCHANGED:
            return torch.clone(coords)
        elif self.strategy in [self.DenseType.FULL_EXPAND_1, self.DenseType.FULL_EXPAND_3]:
            min_bound = torch.min(coords, dim=0).values.cpu().numpy()
            max_bound = torch.max(coords, dim=0).values.cpu().numpy() + stride
            if self.strategy == self.DenseType.FULL_EXPAND_3:
                min_bound -= stride
                max_bound += stride
            cx = torch.arange(min_bound[0], max_bound[0], stride, dtype=torch.int32, device=coords.device)
            cy = torch.arange(min_bound[1], max_bound[1], stride, dtype=torch.int32, device=coords.device)
            cz = torch.arange(min_bound[2], max_bound[2], stride, dtype=torch.int32, device=coords.device)
            coords = torch.stack(torch.meshgrid(cx, cy, cz, indexing='ij'), dim=3).view(-1, 3)
            return coords
        elif self.strategy in [self.DenseType.SPARSE_EXPAND_3, self.DenseType.SPARSE_EXPAND_5]:
            if self.strategy == self.DenseType.SPARSE_EXPAND_3:
                offsets = np.arange(-1, 2)
            else:
                offsets = np.arange(-2, 3)
            kernel = np.array([[z, y, x, 0] for z in offsets for y in offsets for x in offsets])
            mc_offsets = torch.tensor(kernel * stride, dtype=torch.int32, device=coords.device)
            coords = (coords.unsqueeze(dim=1).repeat(1, mc_offsets.size(0), 1) +
                      mc_offsets.unsqueeze(0)).view(-1, 3)
            coords = torch_unique(coords, dim=0)
            return coords
        else:
            raise NotImplementedError


# noinspection PyTypeChecker
class Conv3d(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 bias: bool = False,
                 transposed: bool = False,
                 branch: int = HashTree.ENCODER) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.transposed = transposed
        self.branch = branch

        assert self.kernel_size in [1, 2, 3]
        assert self.stride in [1, 2]
        assert not (transposed and branch == HashTree.ENCODER), "Why use transpose for encoder?"

        self.kernel_volume = self.kernel_size ** 3
        if self.kernel_volume > 1:
            self.kernel = nn.Parameter(
                torch.zeros(self.kernel_volume, in_channels, out_channels))
        else:
            self.kernel = nn.Parameter(torch.zeros(in_channels, out_channels))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def extra_repr(self) -> str:
        branch_text = ['ENC', 'DEC', 'DEC_TMP'][self.branch]
        s = '[' + branch_text + '] {in_channels}, {out_channels}, kernel_size={kernel_size}'
        if self.stride != 1:
            s += ', stride={stride}'
        if self.bias is None:
            s += ', bias=False'
        if self.transposed:
            s += ', transposed=True'
        return s.format(**self.__dict__)

    def reset_parameters(self) -> None:
        std = 1 / math.sqrt(
            (self.out_channels if self.transposed else self.in_channels)
            * self.kernel_volume)
        self.kernel.data.uniform_(-std, std)
        if self.bias is not None:
            self.bias.data.uniform_(-std, std)

    def forward(self, hash_tree: HashTree, feat: torch.Tensor, feat_depth: int):
        assert self.stride in [1, 2]

        if self.kernel_size == 1 and self.stride == 1:
            # 1x1 convolution
            output_feats = feat.matmul(self.kernel)
            output_depth = feat_depth
        else:
            if self.stride == 1:
                output_depth = feat_depth
            else:
                output_depth = feat_depth - 1 if self.transposed else feat_depth + 1

            # Obtain and convert nmap to kmap.
            if self.transposed:
                src_ids, tgt_ids, _, nbsizes = hash_tree.get_neighbours(
                    feat_depth, output_depth, self.kernel_size, self.branch, conv_based=True)
                sizes = (hash_tree.get_coords_size(self.branch, output_depth),
                         hash_tree.get_coords_size(self.branch, feat_depth))
            else:
                src_ids, tgt_ids, _, nbsizes = hash_tree.get_neighbours(
                    output_depth, feat_depth, self.kernel_size, self.branch, conv_based=True)
                sizes = (hash_tree.get_coords_size(self.branch, feat_depth),
                         hash_tree.get_coords_size(self.branch, output_depth))

            nbmaps = torch.stack([tgt_ids, src_ids], dim=1)
            output_feats = ConvolutionFunction.apply(
                feat, self.kernel, nbmaps, nbsizes, sizes, self.transposed, True
            )

        if self.bias is not None:
            output_feats += self.bias

        return output_feats, output_depth


class SparseSequential(nn.Module):
    def forward(self, hash_tree: HashTree, feat: torch.Tensor, feat_depth: int):
        for module in self._modules.values():
            feat, feat_depth = module(hash_tree, feat, feat_depth)
        return feat, feat_depth


class SparseConvBlock(SparseSequential):
    def __init__(self, in_channels, out_channels, order, num_groups, branch, kernel_size=3):
        super().__init__()
        for i, char in enumerate(order):
            if char == 'r':
                self.add_module('ReLU', Activation(nn.ReLU(inplace=True)))
            elif char == 'l':
                self.add_module('LeakyReLU', Activation(nn.LeakyReLU(negative_slope=0.1, inplace=True)))
            elif char == 'c':
                # add learnable bias only in the absence of batchnorm/groupnorm
                self.add_module('Conv', Conv3d(in_channels, out_channels, kernel_size, 1, bias='g' not in order,
                                               transposed=False, branch=branch))
            elif char == 'g':
                if i < order.index('c'):
                    num_channels = in_channels
                else:
                    num_channels = out_channels

                # use only one group if the given number of groups is greater than the number of channels
                if num_channels < num_groups:
                    num_groups = 1

                assert num_channels % num_groups == 0, \
                    f'Expected number of channels in input to be divisible by num_groups. ' \
                    f'num_channels={num_channels}, num_groups={num_groups}'
                self.add_module('GroupNorm', GroupNorm(num_groups=num_groups, num_channels=num_channels))
            else:
                raise NotImplementedError


class SparseDoubleConv(SparseSequential):
    def __init__(self, in_channels, out_channels, branch, order, num_groups, pooling=None):
        super().__init__()
        if branch == HashTree.ENCODER:
            # we're in the encoder path
            conv1_in_channels = in_channels
            conv1_out_channels = out_channels // 2
            if conv1_out_channels < in_channels:
                conv1_out_channels = in_channels
            conv2_in_channels, conv2_out_channels = conv1_out_channels, out_channels
            assert pooling in [None, 'max', 'conv']
            if pooling == 'max':
                self.add_module('MaxPool', MaxPooling())
            elif pooling == 'conv':
                self.add_module('DownConv',
                                Conv3d(conv1_in_channels, conv1_in_channels, 3, 2, bias=order[0] != 'g'))
        else:
            # we're in the decoder path, decrease the number of channels in the 1st convolution
            conv1_in_channels, conv1_out_channels = in_channels, out_channels
            conv2_in_channels, conv2_out_channels = out_channels, out_channels

        self.add_module('SingleConv1',
                        SparseConvBlock(conv1_in_channels, conv1_out_channels, order, num_groups, branch))
        self.add_module('SingleConv2',
                        SparseConvBlock(conv2_in_channels, conv2_out_channels, order, num_groups, branch))


class SparseUNet(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks, f_maps=64, order='gcr', num_groups=8,
                 pooling='max', upsample='nearest', skip_connection=True):
        super().__init__()
        n_features = [in_channels] + [f_maps * 2 ** k for k in range(num_blocks)]
        self.encoders = nn.ModuleList()
        self.upsamplers = nn.ModuleList()
        self.decoders = nn.ModuleList()

        self.skip_connection = skip_connection

        for layer_idx in range(num_blocks):
            self.encoders.add_module(f'Enc{layer_idx}', SparseDoubleConv(
                n_features[layer_idx],
                n_features[layer_idx + 1], True, order, num_groups,
                pooling if layer_idx > 0 else None
            ))

        for layer_idx in range(-2, -num_blocks - 1, -1):
            self.decoders.add_module(f'Dec{layer_idx}', SparseDoubleConv(
                n_features[layer_idx + 1] + (n_features[layer_idx] if skip_connection else 0),
                n_features[layer_idx], False, order, num_groups, None
            ))
            if upsample == 'nearest':
                up_module = NearestUpsampling()
            elif upsample == 'deconv':
                up_module = Conv3d(
                    n_features[layer_idx + 1], n_features[layer_idx + 1],
                    3, 2, bias=True, transposed=True)
            else:
                raise NotImplementedError
            self.upsamplers.add_module(f'Up{layer_idx}', up_module)

        self.final_conv = Conv3d(n_features[1], out_channels, 1, 1, bias=True)

    def forward(self, hash_tree: HashTree, featx: torch.Tensor, feat_depth: int):
        assert hash_tree.is_reflected, "During the middle transfer, decoder branch and encoder should align"

        encoder_features = {}
        feat = featx
        for module in self.encoders:
            feat, feat_depth = module(hash_tree, feat, feat_depth)
            encoder_features[feat_depth] = feat

        for module, upsampler in zip(self.decoders, self.upsamplers):
            feat, feat_depth = upsampler(hash_tree, feat, feat_depth)
            feat = torch.cat([encoder_features[feat_depth], feat], dim=1)
            feat, feat_depth = module(hash_tree, feat, feat_depth)

        feat, feat_depth = self.final_conv(hash_tree, feat, feat_depth)

        return feat, feat_depth


class SparsePVNet(nn.Module):
    def __init__(self, in_channels, out_channels, grid_channels, num_blocks,
                 f_maps=64, order='gcr', num_groups=8,
                 pooling='max', upsample='nearest', skip_connection=True):
        super().__init__()
        n_features = [in_channels] + [f_maps * 2 ** k for k in range(num_blocks)]
        self.encoders = nn.ModuleList()
        self.upsamplers = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.grid_convs = nn.ModuleList()

        self.skip_connection = skip_connection
        self.num_blocks = num_blocks

        for layer_idx in range(num_blocks):
            self.encoders.add_module(f'Enc{layer_idx}', SparseDoubleConv(
                n_features[layer_idx],
                n_features[layer_idx + 1], True, order, num_groups,
                pooling if layer_idx > 0 else None
            ))

        for layer_idx in range(-2, -num_blocks - 1, -1):
            self.decoders.add_module(f'Dec{layer_idx}', SparseDoubleConv(
                n_features[layer_idx + 1] + (n_features[layer_idx] if skip_connection else 0),
                n_features[layer_idx], False, order, num_groups, None
            ))
            if upsample == 'nearest':
                up_module = NearestUpsampling()
            elif upsample == 'deconv':
                up_module = Conv3d(
                    n_features[layer_idx + 1], n_features[layer_idx + 1],
                    3, 2, bias=True, transposed=True)
            else:
                raise NotImplementedError
            self.upsamplers.add_module(f'Up{layer_idx}', up_module)

        for layer_idx in range(num_blocks):
            # Try simple conv block
            self.grid_convs.add_module(f'Grid{layer_idx}', Conv3d(
                n_features[layer_idx + 1], grid_channels, 3, 1, bias=True, transposed=False))

        self.final_conv = Conv3d(n_features[1], out_channels, 1, 1, bias=True)

    def forward(self, hash_tree: HashTree, feat: torch.Tensor, feat_depth: int):
        assert hash_tree.is_reflected, "During the middle transfer, decoder branch and encoder should align"

        encoder_features = {}
        grid_features = {}

        # Down-sample
        for module in self.encoders:
            feat, feat_depth = module(hash_tree, feat, feat_depth)
            encoder_features[feat_depth] = feat

        grid_features[feat_depth] = encoder_features[feat_depth]

        # Up-sample
        for module, upsampler in zip(self.decoders, self.upsamplers):
            feat, feat_depth = upsampler(hash_tree, feat, feat_depth)
            feat = torch.cat([encoder_features[feat_depth], feat], dim=1)
            feat, feat_depth = module(hash_tree, feat, feat_depth)
            grid_features[feat_depth] = feat

        feat, feat_depth = self.final_conv(hash_tree, feat, feat_depth)

        # Output side-way grid features.
        for depth, grid_conv in enumerate(self.grid_convs):
            grid_features[depth], _ = grid_conv(hash_tree, grid_features[depth], depth)

        return feat, feat_depth, grid_features
