import numpy as np
from numpy.random import RandomState

from dataset.base import DatasetSpec as DS


def pad_cloud(P: np.ndarray, n_in: int, return_inds=False, random_state=None):
    """
    Pad or subsample 3D Point cloud to n_in (fixed) number of points
    :param P: N x C numpy array
    :param n_in: number of points to truncate
    :return: n_in x C numpy array
    """
    if random_state is None:
        random_state = RandomState()

    N = P.shape[0]
    # https://github.com/charlesq34/pointnet/issues/41
    if N > n_in:  # need to subsample
        choice = random_state.choice(N, n_in, replace=False)
    elif N < n_in:  # need to pad by duplication
        ii = random_state.choice(N, n_in - N)
        choice = np.concatenate([range(N), ii])
    else:
        choice = np.arange(N)

    if return_inds:
        return choice
    else:
        return P[choice, :]


class PointcloudNoise:
    def __init__(self, stddev):
        self.stddev = stddev

    def __call__(self, data, rng):
        if self.stddev == 0.0:
            return data
        data_out = data.copy()
        if DS.INPUT_PC in data.keys():
            points = data[DS.INPUT_PC]
            noise = self.stddev * rng.randn(*points.shape)
            noise = noise.astype(np.float32)
            data_out[DS.INPUT_PC] = points + noise
        return data_out


class SubsamplePointcloud:
    """ Point cloud subsampling transformation class.

    It subsamples the point cloud data.

    Args:
        N (int): maximum number of points to be subsampled
        n_min (int): minimum number, default is None
    """

    def __init__(self, N, n_min=None):
        self.N = N
        self.n_min = n_min if n_min is not None else N
        assert self.n_min <= self.N

    def __call__(self, data, rng):
        # Will modify 'INPUT_PC' and 'TARGET_NORMAL'
        data_out = data.copy()
        assert DS.INPUT_PC in data.keys()

        points = data[DS.INPUT_PC]
        if points.shape[0] > self.N:
            indices = pad_cloud(points, self.N, return_inds=True, random_state=rng)
        elif points.shape[0] < self.n_min:
            indices = pad_cloud(points, self.n_min, return_inds=True, random_state=rng)
        else:
            indices = np.arange(points.shape[0])
        data_out[DS.INPUT_PC] = points[indices, :]

        if DS.TARGET_NORMAL in data.keys():
            data_out[DS.TARGET_NORMAL] = data[DS.TARGET_NORMAL][indices, :]

        return data_out


class BoundScale:
    """
    Centralize the point cloud and limit the bound to [-a,a], where min_a <= a <= max_a.
    """
    def __init__(self, min_a, max_a):
        assert min_a <= max_a
        self.min_a = min_a
        self.max_a = max_a

    def __call__(self, data, rng):
        # Will modify 'INPUT_PC', 'GT_DENSE_PC', 'GT_ONET_SAMPLE'
        data_out = data.copy()
        assert DS.INPUT_PC in data.keys()

        points = data[DS.INPUT_PC]
        p_max, p_min = np.max(points, axis=0), np.min(points, axis=0)
        center = (p_max + p_min) / 2.
        cur_scale = np.max(p_max - p_min) / 2.
        target_scale = max(min(cur_scale, self.max_a), self.min_a)

        data_out[DS.INPUT_PC] = (points - center[None, :]) * (target_scale / cur_scale)
        if DS.GT_DENSE_PC in data.keys():
            data_out[DS.GT_DENSE_PC] = (data[DS.GT_DENSE_PC] - center[None, :]) * (target_scale / cur_scale)

        if DS.GT_ONET_SAMPLE in data.keys():
            data_out[DS.GT_ONET_SAMPLE][0] = (data[DS.GT_ONET_SAMPLE][0] - center[None, :]) * \
                                             (target_scale / cur_scale)

        return data_out


class ComposedTransforms:
    def __init__(self, args):
        self.args = args
        self.transforms = []
        if self.args is not None:
            for t_spec in self.args:
                self.transforms.append(
                    globals()[t_spec.name](**t_spec.args)
                )

    def __call__(self, data, rng):
        for t in self.transforms:
            data = t(data, rng)
        return data
