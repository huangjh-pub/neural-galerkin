import numpy as np
import torch
from pykdtree.kdtree import KDTree

# Worst-case metric, used if no prediction is generated.
EMPTY_PCL_DICT = {
    'completeness': np.sqrt(3),
    'accuracy': np.sqrt(3),
    'completeness2': 3,
    'accuracy2': 3,
    'chamfer': 6,
}

EMPTY_PCL_DICT_NORMALS = {
    'normals completeness': -1.,
    'normals accuracy': -1.,
    'normals': -1.,
}


def distance_p2p(points_src, normals_src, points_tgt, normals_tgt):
    kdtree = KDTree(points_tgt)
    dist, idx = kdtree.query(points_src)

    if normals_src is not None and normals_tgt is not None:
        normals_src = \
            normals_src / np.linalg.norm(normals_src, axis=-1, keepdims=True)
        normals_tgt = \
            normals_tgt / np.linalg.norm(normals_tgt, axis=-1, keepdims=True)

        normals_dot_product = (normals_tgt[idx] * normals_src).sum(axis=-1)
        # Handle normals that point into wrong direction gracefully
        # (mostly due to method not caring about this in generation)
        normals_dot_product = np.abs(normals_dot_product)
    else:
        normals_dot_product = np.array(
            [np.nan] * points_src.shape[0], dtype=np.float32)
    return dist, normals_dot_product


def get_threshold_percentage(dist, thresholds):
    in_threshold = [
        (dist <= t).mean() for t in thresholds
    ]
    return in_threshold


class MeshEvaluator:
    """
    Mesh evaluation class that handles the mesh evaluation process. Returned dict has meaning:
        - completeness:             mean distance from all gt to pd.
        - accuracy:                 mean distance from all pd to gt.
        - chamfer-l1/l2:            average of the above two. [Chamfer distance]
        - f-score(/-15/-20):        [F-score], computed at the threshold of 0.01, 0.015, 0.02.
        - normals completeness:     mean normal alignment (0-1) from all gt to pd.
        - normals accuracy:         mean normal alignment (0-1) from all pd to gt.
        - normals:                  average of the above two, i.e., [Normal Consistency Score] (0-1)
    Args:
        n_points (int): number of points to be used for evaluation
    """

    def __init__(self, n_points=100000, essential_only=False):
        self.n_points = n_points
        # self.thresholds = np.linspace(1. / 1000, 1, 1000)
        # self.fidx = [9, 14, 19]
        self.thresholds = np.array([0.01, 0.015, 0.02, 0.002])
        self.fidx = [0, 1, 2, 3]
        self.essential_only = essential_only

    def eval_mesh(self, mesh, pointcloud_tgt, normals_tgt):
        """
        Evaluates a mesh.
        :param mesh: (o3d.geometry.TriangleMesh) mesh which should be evaluated
        :param pointcloud_tgt: np (Nx3) ground-truth xyz
        :param normals_tgt: np (Nx3) ground-truth normals
        :return: metric-dict
        """
        if isinstance(pointcloud_tgt, torch.Tensor):
            pointcloud_tgt = pointcloud_tgt.detach().cpu().numpy().astype(float)

        if isinstance(normals_tgt, torch.Tensor):
            normals_tgt = normals_tgt.detach().cpu().numpy().astype(float)

        # Triangle normal is used to be consistent with SAP.
        try:
            sampled_pcd = mesh.sample_points_uniformly(
                number_of_points=self.n_points, use_triangle_normal=True)
            pointcloud = np.asarray(sampled_pcd.points)
            normals = np.asarray(sampled_pcd.normals)
        except RuntimeError:    # Sample error.
            pointcloud = np.zeros((0, 3))
            normals = np.zeros((0, 3))

        out_dict = self.eval_pointcloud(
            pointcloud, pointcloud_tgt, normals, normals_tgt)

        return out_dict

    def eval_pointcloud(self, pointcloud, pointcloud_tgt, normals=None, normals_tgt=None):
        """
        Evaluates a point cloud.
        :param pointcloud: np (Mx3) predicted xyz
        :param pointcloud_tgt:  np (Nx3) ground-truth xyz
        :param normals: np (Mx3) predicted normals
        :param normals_tgt: np (Nx3) ground-truth normals
        :return: metric-dict
        """
        # Return maximum losses if pointcloud is empty
        if pointcloud.shape[0] == 0:
            print('Empty pointcloud / mesh detected! Return max loss.')
            out_dict = EMPTY_PCL_DICT.copy()
            if normals is not None and normals_tgt is not None:
                out_dict.update(EMPTY_PCL_DICT_NORMALS)
            return out_dict

        # Completeness: how far are the points of the target point cloud
        # from thre predicted point cloud
        completeness, completeness_normals = distance_p2p(
            pointcloud_tgt, normals_tgt, pointcloud, normals
        )
        recall = get_threshold_percentage(completeness, self.thresholds)
        completeness2 = completeness ** 2

        completeness = completeness.mean()
        completeness2 = completeness2.mean()
        completeness_normals = completeness_normals.mean()

        # Accuracy: how far are th points of the predicted pointcloud
        # from the target pointcloud
        accuracy, accuracy_normals = distance_p2p(
            pointcloud, normals, pointcloud_tgt, normals_tgt
        )
        precision = get_threshold_percentage(accuracy, self.thresholds)
        accuracy2 = accuracy ** 2

        accuracy = accuracy.mean()
        accuracy2 = accuracy2.mean()
        accuracy_normals = accuracy_normals.mean()

        # Chamfer distance
        chamfer_l2 = 0.5 * (completeness2 + accuracy2)
        normals_correctness = (
                0.5 * completeness_normals + 0.5 * accuracy_normals
        )
        chamfer_l1 = 0.5 * (completeness + accuracy)

        # F-Score
        F = [
            2 * precision[i] * recall[i] / (precision[i] + recall[i])
            for i in range(len(precision))
        ]

        if self.essential_only:
            return {
                'chamfer-L1': chamfer_l1,
                'f-score': F[self.fidx[0]],
                'normals': normals_correctness
            }

        out_dict = {
            'completeness': completeness,
            'accuracy': accuracy,
            'normals completeness': completeness_normals,
            'normals accuracy': accuracy_normals,
            'normals': normals_correctness,
            'completeness2': completeness2,
            'accuracy2': accuracy2,
            'chamfer-L2': chamfer_l2,
            'chamfer-L1': chamfer_l1,
            'f-precision': precision[self.fidx[0]],
            'f-recall': recall[self.fidx[0]],
            'f-score': F[self.fidx[0]],  # threshold = 1.0%
            'f-score-15': F[self.fidx[1]],  # threshold = 1.5%
            'f-score-20': F[self.fidx[2]],  # threshold = 2.0%
            'f-score-strict': F[self.fidx[3]]
        }

        return out_dict
