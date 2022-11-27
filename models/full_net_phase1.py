import torch
import torch.nn.functional as F
from pycg import exp
from dataset.base import DatasetSpec as DS
from models.full_net import Model as BaseModel


class Model(BaseModel):
    def __init__(self, hparams):
        super().__init__(hparams)

    def forward(self, batch, out: dict):
        input_xyz = batch[DS.INPUT_PC][0]
        assert input_xyz.ndim == 2, "Can only forward single batch."

        hash_tree = self.build_hashtree(input_xyz)
        unet_feat = self.encoder(
            None if not self.hparams.use_input_normal else batch[DS.TARGET_NORMAL][0],
            hash_tree
        )
        structure_features, normal_features, basis_features, w_features, _ = self.unet(
            hash_tree, unet_feat, 0, out.get('gt_tree', None))
        out.update({'tree': hash_tree})

        normal_data = {
            feat_depth: normal_data / hash_tree.get_stride(hash_tree.DECODER, feat_depth)
            for feat_depth, normal_data in normal_features.items() if feat_depth < self.hparams.adaptive_depth}

        out.update({'structure_features': structure_features, 'basis_features': basis_features,
                    'reconstructor': None, 'normal_data': normal_data})
        return out

    def compute_loss(self, batch, out, compute_metric: bool):
        loss_dict = exp.TorchLossMeter()
        metric_dict = exp.TorchLossMeter()

        # Learn structure
        hash_tree_gt = self.compute_gt_hashtree(batch, out)
        for feat_depth, struct_feat in out['structure_features'].items():
            base_coords = out['tree'].get_coords(hash_tree_gt.DECODER_TMP, feat_depth)
            gt_status = hash_tree_gt.evaluate_voxel_status(hash_tree_gt.ENCODER, base_coords, feat_depth)
            loss_dict.add_loss(f"struct-{feat_depth}", F.cross_entropy(struct_feat, gt_status),
                               self.hparams.supervision.structure_weight)
            if compute_metric:
                metric_dict.add_loss(f"struct-acc-{feat_depth}",
                                     torch.mean((struct_feat.argmax(dim=1) == gt_status).float()))

        # Supervise normal on grid level.
        if self.hparams.supervision.vox_normal.weight > 0.0:
            gt_vox_normal = self.compute_gt_normal_data(batch, out, 1.0)
            pd_vox_normal = out['normal_data']
            depth_levels = set(gt_vox_normal.keys()).intersection(set(pd_vox_normal.keys()))
            for d in depth_levels:
                gt_normal, pd_normal = gt_vox_normal[d], pd_vox_normal[d]
                loss_dict.add_loss(f"vn-{d}", torch.abs(gt_normal - pd_normal).mean(),
                                   self.hparams.supervision.vox_normal.weight)

        return loss_dict, metric_dict
