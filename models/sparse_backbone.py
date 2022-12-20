import torch
import torch.nn as nn
from jspsr.core.hashtree import HashTree, VoxelStatus

from models.ascn.unet import SparseDoubleConv, NearestUpsampling, Conv3d, \
    CoordinateTransform, SparseSequential, SparseConvBlock, CoordinateDensification


class SparseHead(SparseSequential):
    def __init__(self, in_channels, out_channels, branch, order, num_groups, enhanced=False):
        super().__init__()
        self.add_module('SingleConv', SparseConvBlock(in_channels, in_channels, order, num_groups, branch))
        if enhanced:
            mid_channels = min(64, in_channels)
            self.add_module('SingleConv2', SparseConvBlock(in_channels, mid_channels, order, num_groups, branch))
            self.add_module('OneConv0', SparseConvBlock(mid_channels, mid_channels, order, num_groups, branch,
                                                        kernel_size=1))
            self.add_module('OutConv', Conv3d(mid_channels, out_channels, 1, bias=True, branch=branch))
        else:
            self.add_module('OutConv', Conv3d(in_channels, out_channels, 1, bias=True, branch=branch))


class SelfAttentionBlock(nn.Module):
    def __init__(self, dim, mlp_dim, n_heads):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, dim),
        )

    def forward(self, x):
        n_x = self.norm(x).unsqueeze(0)
        n_x = self.attn(n_x, n_x, n_x, need_weights=False)[0][0]
        x = x + n_x

        n_x = self.norm(x)
        n_x = self.ff(n_x)

        return x + n_x


class FeatureEnhancement(nn.Module):
    def __init__(self, in_channels, emb_channels, mlp_channels, n_blocks, n_heads):
        super().__init__()
        self.to_emb = nn.Linear(in_channels, emb_channels)
        self.blocks = nn.ModuleList([
            SelfAttentionBlock(emb_channels, mlp_channels, n_heads) for _ in range(n_blocks)
        ])
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(emb_channels),
            nn.Linear(emb_channels, in_channels)
        )

    def forward(self, feat: torch.Tensor):
        x = self.to_emb(feat)
        # We omit cls_tokens and pos_embedding
        for block in self.blocks:
            x = block(x)
        return self.mlp_head(x)


class SparseStructureNet(nn.Module):
    def __init__(self, in_channels, num_blocks, basis_channels, normal_channels=3,
                 f_maps=64, order='gcr', num_groups=8,
                 pooling='max', upsample='nearest', skip_connection=True,
                 neck_dense_type="UNCHANGED", enhance_level: int = 0,
                 w_branch_cfg=None):
        super().__init__()
        n_features = [in_channels] + [f_maps * 2 ** k for k in range(num_blocks)]
        self.encoders = nn.ModuleList()
        self.upsamplers = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.struct_convs = nn.ModuleList()
        self.normal_convs = nn.ModuleList()
        self.basis_convs = nn.ModuleList()
        self.enhance_level = enhance_level
        self.w_branch_cfg = w_branch_cfg
        self.w_convs = nn.ModuleList() if self.w_branch_cfg is not None else [None for _ in range(num_blocks)]

        self.skip_connection = skip_connection
        self.num_blocks = num_blocks

        for layer_idx in range(num_blocks):
            self.encoders.add_module(f'Enc{layer_idx}', SparseDoubleConv(
                n_features[layer_idx],
                n_features[layer_idx + 1], HashTree.ENCODER, order, num_groups,
                pooling if layer_idx > 0 else None
            ))

        for layer_idx in range(-1, -num_blocks - 1, -1):
            self.struct_convs.add_module(f'Struct{layer_idx}', SparseHead(
                n_features[layer_idx], 3,
                HashTree.DECODER_TMP, order, num_groups))
            if self.w_branch_cfg is not None:
                self.w_convs.add_module(f'W{layer_idx}', SparseHead(
                    n_features[layer_idx], self.w_branch_cfg.dim,
                    HashTree.DECODER_TMP if self.w_branch_cfg.is_tmp_branch else HashTree.DECODER,
                    order, num_groups))
            self.normal_convs.add_module(f'Normal{layer_idx}', SparseHead(
                n_features[layer_idx], normal_channels,
                HashTree.DECODER, order, num_groups))
            self.basis_convs.add_module(f'Basis{layer_idx}', SparseHead(
                n_features[layer_idx], basis_channels,
                HashTree.DECODER, order, num_groups, enhanced=self.enhance_level > 0))
            if layer_idx < -1:
                self.decoders.add_module(f'Dec{layer_idx}', SparseDoubleConv(
                    n_features[layer_idx + 1] + (n_features[layer_idx] if skip_connection else 0),
                    n_features[layer_idx], HashTree.DECODER_TMP, order, num_groups, None
                ))
                if upsample == 'nearest':
                    up_module = NearestUpsampling(target_branch=HashTree.DECODER_TMP)
                elif upsample == 'deconv':
                    raise NotImplementedError
                    # up_module = Conv3d(
                    #     n_features[layer_idx + 1], n_features[layer_idx + 1],
                    #     3, 2, bias=True, transposed=True)
                else:
                    raise NotImplementedError
                self.upsamplers.add_module(f'Up{layer_idx}', up_module)

        self.trans = CoordinateTransform(HashTree.ENCODER, HashTree.DECODER_TMP)
        self.neck = CoordinateDensification(CoordinateDensification.DenseType[neck_dense_type])

        if self.enhance_level > 1:
            self.enhance = FeatureEnhancement(n_features[-1], n_features[-1], n_features[-1] * 2,
                                              n_blocks=4, n_heads=4)

    def initialize_basis_branch(self, basis):
        for bconv in self.basis_convs:
            last_module = bconv._modules['OutConv']
            last_module.kernel.data.zero_()
            basis.initialize_feature_value(last_module.bias.data[None, :])

    def forward(self, hash_tree: HashTree, feat: torch.Tensor, feat_depth: int, gt_hash_tree: HashTree = None):
        # Sanity Check:
        dec_voxel_counts = [hash_tree.get_coords_size(hash_tree.DECODER, d) > 0 for d in range(hash_tree.depth)]
        if any(dec_voxel_counts):
            print("Warning: Decoder structure seems already built.")

        encoder_features = {}
        struct_features = {}
        normal_features = {}
        basis_features = {}
        w_features = {}

        # Down-sample
        for module in self.encoders:
            feat, feat_depth = module(hash_tree, feat, feat_depth)
            encoder_features[feat_depth] = feat

        # Bottleneck processing
        dec_main_feature = encoder_features[feat_depth]
        hash_tree.update_coords(hash_tree.DECODER_TMP, feat_depth, self.neck(
            hash_tree.get_coords(hash_tree.ENCODER, feat_depth),
            hash_tree.get_stride(hash_tree.ENCODER, feat_depth)))
        dec_main_feature, _ = self.trans(hash_tree, dec_main_feature, feat_depth)

        # Way 1 to do it here.
        if self.enhance_level > 1:
            dec_main_feature = self.enhance(dec_main_feature)

        # Up-sample
        upsample_mask = None
        for module, upsampler, struct_conv, normal_conv, basis_conv, w_conv in zip(
                [None] + list(self.decoders), [None] + list(self.upsamplers),
                self.struct_convs, self.normal_convs, self.basis_convs, self.w_convs):
            if module is not None:
                dec_main_feature, feat_depth = upsampler(hash_tree, dec_main_feature, feat_depth, upsample_mask)
                enc_feat, _ = self.trans(hash_tree, encoder_features[feat_depth], feat_depth)
                dec_main_feature = torch.cat([enc_feat, dec_main_feature], dim=1)
                dec_main_feature, feat_depth = module(hash_tree, dec_main_feature, feat_depth)

            # Do structure inference.
            struct_features[feat_depth], _ = struct_conv(hash_tree, dec_main_feature, feat_depth)

            if w_conv is not None and self.w_branch_cfg.is_tmp_branch:
                w_features[feat_depth], _ = w_conv(hash_tree, dec_main_feature, feat_depth)

            if gt_hash_tree is None:
                struct_decision = torch.argmax(struct_features[feat_depth], dim=1)
            else:
                struct_decision = gt_hash_tree.evaluate_voxel_status(gt_hash_tree.ENCODER,
                    hash_tree.get_coords(hash_tree.DECODER_TMP, feat_depth), feat_depth)
            exist_mask = struct_decision != VoxelStatus.VS_NON_EXIST.value

            # If the predicted structure is empty, then stop early.
            #   (Related branch will not have gradient)
            if not torch.any(exist_mask):
                break

            hash_tree.update_coords(hash_tree.DECODER, feat_depth,
                                    hash_tree.get_coords(hash_tree.DECODER_TMP, feat_depth)[exist_mask])
            dec_main_feature = dec_main_feature[exist_mask]
            upsample_mask = (struct_decision == VoxelStatus.VS_EXIST_CONTINUE.value)[exist_mask]

            # Do normal&basis prediction.
            normal_features[feat_depth], _ = normal_conv(hash_tree, dec_main_feature, feat_depth)
            basis_features[feat_depth], _ = basis_conv(hash_tree, dec_main_feature, feat_depth)

            if w_conv is not None and not self.w_branch_cfg.is_tmp_branch:
                w_features[feat_depth], _ = w_conv(hash_tree, dec_main_feature, feat_depth)

        # For missing tree features, fill them in.
        def get_empty(ref_dict):
            ref_tensor = ref_dict.values().__iter__().__next__()
            return torch.empty((0, ref_tensor.size(1)), device=ref_tensor.device, dtype=ref_tensor.dtype)

        missing_depths = set(encoder_features.keys()).difference(set(normal_features.keys()))
        for d in missing_depths:
            normal_features[d] = get_empty(normal_features)
            basis_features[d] = get_empty(basis_features)
            hash_tree.update_coords(hash_tree.DECODER, d, None)
            if d not in struct_features:
                struct_features[d] = get_empty(struct_features)
            if self.w_branch_cfg is not None and d not in w_features:
                w_features[d] = get_empty(w_features)

        return struct_features, normal_features, basis_features, w_features, feat_depth
