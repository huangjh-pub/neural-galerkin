import random

import pandas as pd
import torch
import torch.nn.functional as F
import torch_scatter
from pycg import exp, vis

from mesh_evaluator import MeshEvaluator
from models.ascn.decoder import MultiscaleDecoder
from models.ascn.encoder import LocalPoolPointnet
from torch_spsr.bases import get_basis
from torch_spsr.core.hashtree import HashTree
from torch_spsr.core.reconstructor import Reconstructor
from dataset.base import DatasetSpec as DS, list_collate
from models.sparse_backbone import SparseStructureNet

from models.base_model import BaseModel
from ext import sdf_from_points


def get_normal_variation_policy(tau: float = 0.1, take_abs: bool = False):
    """
    Compute Var(nx) + Var(ny) + Var(nz).
    """
    def sp_impl(xyz: torch.Tensor, inds: torch.Tensor, normal: torch.Tensor):
        nx, ny, nz = normal[:, 0], normal[:, 1], normal[:, 2]
        if take_abs:
            nx, ny, nz = torch.abs(nx), torch.abs(ny), torch.abs(nz)
        vnx = torch_scatter.scatter_std(nx, inds)
        vny = torch_scatter.scatter_std(ny, inds)
        vnz = torch_scatter.scatter_std(nz, inds)
        return ((vnx + vny + vnz) > tau)[inds]

    return sp_impl


class Model(BaseModel):
    def __init__(self, hparams):
        super().__init__(hparams)
        self.basis = get_basis(self.hparams.basis.name, **self.hparams.basis.kwargs)
        if self.hparams.use_input_normal:
            self.hparams.network.encoder.dim = 6
        self.encoder = LocalPoolPointnet(**self.hparams.network.encoder)
        self.unet = SparseStructureNet(**self.hparams.network.unet,
                                       basis_channels=self.basis.get_feature_size(),
                                       w_branch_cfg=None if not self.hparams.weighted.enabled else
                                       self.hparams.weighted.w_branch)
        if self.hparams.init_basis_branch:
            self.unet.initialize_basis_branch(self.basis)

        if self.hparams.weighted.enabled:
            self.decoder = MultiscaleDecoder(
                p_dim=3 * self.hparams.weighted.decoder.multiscale_depths,
                c_dim=self.hparams.weighted.w_branch.dim * self.hparams.weighted.decoder.multiscale_depths,
                out_dim=1, **self.hparams.weighted.decoder
            )

        # We might want to use a pre-trained network.
        if self.hparams.load_pretrained is not None:
            self.load_state_dict(torch.load(self.hparams.load_pretrained)['state_dict'],
                                 strict=self.hparams.load_pretrained_strict)
            exp.logger.info("Loaded check point state from", self.hparams.load_pretrained)

    def build_hashtree(self, input_xyz, is_gt=False, input_normal=None):
        hash_tree = HashTree(input_xyz, voxel_size=self.hparams.voxel_size, depth=self.hparams.tree_depth)
        if self.hparams.tree_expand > 0:
            hash_tree.build_encoder_hierarchy_dense(expand_range=self.hparams.tree_expand, uniform_density=True)
        else:
            if is_gt:
                if self.hparams.adaptive_policy.method == "density":
                    hash_tree.build_encoder_hierarchy_adaptive(
                        density_depth=1, min_density=self.hparams.adaptive_policy.min_density,
                        limit_adaptive_depth=self.hparams.adaptive_depth)
                elif self.hparams.adaptive_policy.method == "normal":
                    hash_tree.build_hierarchy_subdivide(
                        subdivide_policy=get_normal_variation_policy(self.hparams.adaptive_policy.tau, take_abs=True),
                        density_depth=1, expand=True,
                        limit_adaptive_depth=self.hparams.adaptive_depth, normal=input_normal)
                else:
                    raise NotImplementedError
            else:
                hash_tree.build_encoder_hierarchy_adaptive(density_depth=1, uniform_density=True)
        return hash_tree

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

        tree_size = hash_tree.get_coords_size(hash_tree.DECODER, 0)
        if tree_size > 50000 and self.trainer.training:
            exp.logger.warning(f"Skipping {batch[DS.SHAPE_NAME][0]} due to large tree size ({tree_size}).")
            return None

        if self.hparams.weighted.enabled:
            point_w = self.decoder(
                hash_tree, w_features,
                hash_tree.DECODER_TMP
            )
            out['point_w'] = point_w[:, 0]
        else:
            point_w = None

        normal_data = {
            feat_depth: normal_data / hash_tree.get_stride(hash_tree.DECODER, feat_depth)
            for feat_depth, normal_data in normal_features.items() if feat_depth < self.hparams.adaptive_depth}

        reconstructor = Reconstructor(hash_tree, self.basis, basis_features)
        reconstructor.solve_multigrid(
            start_depth=hash_tree.depth - 1,
            end_depth=0,
            normal_data=normal_data,
            screen_alpha=self.hparams.screening.alpha,
            screen_xyz=input_xyz,
            screen_delta=self.hparams.screening.delta * (2 * point_w if point_w is not None else 1),
            solver=self.hparams.solver,
            verbose=False,
        )

        reconstructor.set_fixed_level_set(self.hparams.screening.delta)
        out.update({'structure_features': structure_features, 'basis_features': basis_features,
                    'reconstructor': reconstructor, 'normal_data': normal_data})
        return out

    def compute_gt_chi(self, query_pos: torch.Tensor, ref_xyz: torch.Tensor, ref_normal: torch.Tensor,
                       do_smooth: bool = False):
        mc_query_sdf = sdf_from_points(query_pos, ref_xyz, ref_normal, 8, 0.02)
        if do_smooth:
            return torch.tanh(-mc_query_sdf / self.hparams.supervision.chi_gt_smoothing) * \
                              self.hparams.screening.delta
        else:
            return (mc_query_sdf < 0.0).float() * 2 - 1

    def compute_gt_hashtree(self, batch, out):
        if 'gt_tree' in out.keys():
            return out['gt_tree']
        gt_tree = self.build_hashtree(
            batch[DS.GT_DENSE_PC][0], is_gt=True, input_normal=batch[DS.GT_DENSE_NORMAL][0])
        out['gt_tree'] = gt_tree
        return gt_tree

    def compute_gt_normal_data(self, batch, out, global_scale=1.0):
        if 'gt_normal_data' in out.keys():
            return out['gt_normal_data']
        assert 'tree' in out.keys(), "Gt normal data has to be splatted onto predicted structure!"
        pd_hashtree = out['tree']

        # The only thing we require from gt_tree is its density, because the structure should follow prediction!
        gt_hashtree = self.compute_gt_hashtree(batch, out)
        gt_density = gt_hashtree.xyz_density

        normal_data = {}
        ref_xyz, ref_normal = batch[DS.GT_DENSE_PC][0], batch[DS.GT_DENSE_NORMAL][0] / gt_density[:, None]
        for d in range(self.hparams.adaptive_depth):
            normal_data_depth, failed_mask = pd_hashtree.splat_data(
                ref_xyz, pd_hashtree.DECODER, d, ref_normal, return_nf_mask=True, check_corr=False)
            normal_data[d] = normal_data_depth / (pd_hashtree.get_stride(pd_hashtree.ENCODER, d) ** 3)
            if global_scale != 1.0:
                normal_data[d] = normal_data[d] * global_scale
            if not torch.any(failed_mask):
                break
            ref_xyz, ref_normal = ref_xyz[failed_mask], ref_normal[failed_mask]

        out['gt_normal_data'] = normal_data
        return normal_data

    def compute_mesh(self, out, return_samples: bool = False):
        with torch.no_grad():
            reconstructor = out['reconstructor']
            mc_query_pos, mc_base_coords = reconstructor.hash_tree.get_test_grid(
                **self.hparams.marching_cubes)
            pd_chi = reconstructor.evaluate_chi(mc_query_pos, max_points=2 ** 20)
            mesh = reconstructor.extract_mesh(
                mc_base_coords, pd_chi, self.hparams.marching_cubes.depth)
        if return_samples:
            return mesh, mc_query_pos, mc_base_coords
        else:
            return mesh

    def compute_loss(self, batch, out, compute_metric: bool):
        loss_dict = exp.TorchLossMeter()
        metric_dict = exp.TorchLossMeter()

        # Learn structure
        hash_tree_gt = self.compute_gt_hashtree(batch, out)
        for feat_depth, struct_feat in out['structure_features'].items():
            base_coords = out['tree'].get_coords(HashTree.DECODER_TMP, feat_depth)
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

        reconstructor = out['reconstructor']

        # Learn geometry
        ref_xyz, ref_normal = batch[DS.GT_DENSE_PC][0], batch[DS.GT_DENSE_NORMAL][0]

        if self.hparams.supervision.surface_weight > 0.0:
            with exp.pt_profile_named("evaluate_chi"):
                pd_chi = reconstructor.evaluate_chi(ref_xyz)
            with exp.pt_profile_named("evaluate_raw_chi"):
                pd_normal = reconstructor.evaluate_raw_chi(ref_xyz, compute_grad=True)
            pd_normal = -pd_normal / (torch.linalg.norm(pd_normal, dim=-1, keepdim=True) + 1.0e-6)
            loss_dict.add_loss('chi', torch.abs(pd_chi).mean() / self.hparams.screening.delta,
                               self.hparams.supervision.surface_weight)
            if self.hparams.supervision.grad_mult > 0.0:
                loss_dict.add_loss('grad-chi', 1.0 - torch.sum(pd_normal * ref_normal, dim=-1).mean(),
                                   self.hparams.supervision.surface_weight * self.hparams.supervision.grad_mult)

        return loss_dict, metric_dict

    def log_visualizations(self, batch, out, batch_idx):
        if self.logger is None:
            return
        with torch.no_grad():
            reconstructor = out['reconstructor']
            if reconstructor is None:
                return
            self.log_geometry("pd_mesh", self.compute_mesh(out))

    def should_use_pd_structure(self, is_val):
        # In case this returns True:
        #   - The tree generation would completely rely on prediction, so does the supervision signal.
        prob = (self.trainer.global_step - self.hparams.structure_schedule.start_step) / \
               (self.hparams.structure_schedule.end_step - self.hparams.structure_schedule.start_step)
        prob = min(max(prob, 0.0), 1.0)
        if not is_val:
            self.log("pd_struct_prob", prob, prog_bar=True, on_step=True, on_epoch=False)
        return random.random() < prob

    def train_val_step(self, batch, batch_idx, is_val):
        out = {'idx': batch_idx}
        if not self.should_use_pd_structure(is_val):
            self.compute_gt_hashtree(batch, out)

        out = self(batch, out)

        # OOM Guard.
        if out is None and not is_val:
            return None

        loss_dict, metric_dict = self.compute_loss(batch, out, compute_metric=is_val)
        if not is_val:
            self.log_dict_prefix('train_loss', loss_dict)
            if batch_idx % 200 == 0:
                self.log_visualizations(batch, out, batch_idx)
        else:
            self.log_dict_prefix('val_metric', metric_dict)
            self.log_dict_prefix('val_loss', loss_dict)

        loss_sum = loss_dict.get_sum()
        self.log('val_loss' if is_val else 'train_loss/sum', loss_sum)
        return loss_sum

    def test_step(self, batch, batch_idx):
        if self.hparams.enable_timing:
            exp.global_timers.enable("main", cuda_sync=True, persistent=True)

        self.log('source', batch[DS.SHAPE_NAME][0], on_epoch=False)

        input_pc = batch[DS.INPUT_PC][0]
        out = {'idx': batch_idx}

        if self.hparams.test_use_gt_structure:
            self.compute_gt_hashtree(batch, out)

        out = self(batch, out)

        loss_dict, metric_dict = self.compute_loss(batch, out, compute_metric=True)
        self.log_dict(loss_dict)
        self.log_dict(metric_dict)

        exp.global_timers.toc("main", "Before mesh.")
        mesh, mc_query_pos, mc_base_coords = self.compute_mesh(out, return_samples=True)
        exp.global_timers.toc("main", "After mesh.")

        evaluator = MeshEvaluator(essential_only=True)
        eval_dict = evaluator.eval_mesh(mesh, batch[DS.GT_DENSE_PC][0], batch[DS.GT_DENSE_NORMAL][0])
        self.log_dict(eval_dict)

        exp.global_timers.finalize("main", merged=True)

        if self.record_folder is not None:
            self.test_log_data({
                'input': vis.pointcloud(input_pc, normal=batch[DS.TARGET_NORMAL][0]),
                'mesh': mesh
            })

        if self.hparams.visualize:
            print(pd.DataFrame([eval_dict]))

            reconstructor = out['reconstructor']
            ref_xyz, ref_normal = batch[DS.GT_DENSE_PC][0], batch[DS.GT_DENSE_NORMAL][0]
            best_mesh = reconstructor.extract_mesh(mc_base_coords,
                                                   self.compute_gt_chi(mc_query_pos, ref_xyz, ref_normal),
                                                   self.hparams.marching_cubes.depth)

            vis.show_3d(
                [vis.pointcloud(input_pc, is_sphere=False, sphere_radius=0.006,
                                **({'cfloat': out['point_w'], 'cfloat_normalize': True} if 'point_w' in out.keys()
                                else {'ucid': 6})), vis.colored_mesh(mesh)],
                [vis.pointcloud(input_pc), vis.colored_mesh(best_mesh)],
                point_size=2, use_new_api=True, auto_plane=False
            )

    def get_dataset_spec(self):
        return [DS.SHAPE_NAME, DS.INPUT_PC, DS.GT_DENSE_PC, DS.GT_DENSE_NORMAL, DS.TARGET_NORMAL]

    def get_collate_fn(self):
        return list_collate

    def get_hparams_metrics(self):
        return [('val_loss', True)]
