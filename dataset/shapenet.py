import os
from pathlib import Path

import numpy as np

from dataset.base import DatasetSpec as DS
from dataset.base import RandomSafeDataset
from dataset.transforms import ComposedTransforms


class ShapeNetDataset(RandomSafeDataset):
    def __init__(self, onet_base_path, spec, split, categories=None, transforms=None,
                 random_seed=0, hparams=None, skip_on_error=False, custom_name="shapenet", **kwargs):
        if isinstance(random_seed, str):
            super().__init__(0, True, skip_on_error)
        else:
            super().__init__(random_seed, False, skip_on_error)
        self.skip_on_error = skip_on_error
        self.custom_name = custom_name

        self.split = split
        self.spec = spec
        self.transforms = ComposedTransforms(transforms)

        # If categories is None, use all sub-folders
        if categories is None:
            base_path = Path(onet_base_path)
            categories = os.listdir(base_path)
            categories = [c for c in categories if (base_path / c).is_dir()]
        self.categories = categories

        # Get all models
        self.models = []
        self.onet_base_paths = {}
        for c in categories:
            self.onet_base_paths[c] = Path(onet_base_path + "/" + c)
            split_file = self.onet_base_paths[c] / (split + '.lst')
            with split_file.open('r') as f:
                models_c = f.read().split('\n')
            if '' in models_c:
                models_c.remove('')
            self.models += [{'category': c, 'model': m} for m in models_c]
        self.hparams = hparams

    def __len__(self):
        return len(self.models)

    def get_name(self):
        return f"{self.custom_name}-cat{len(self.categories)}-{self.split}"

    def get_short_name(self):
        return self.custom_name

    def _get_item(self, data_id, rng):
        category = self.models[data_id]['category']
        model = self.models[data_id]['model']

        data = {}

        gt_data = np.load(self.onet_base_paths[category] / model / 'pointcloud.npz')
        gt_points = gt_data['points'].astype(np.float32)
        gt_normals = gt_data['normals'].astype(np.float32)

        if DS.SHAPE_NAME in self.spec:
            data[DS.SHAPE_NAME] = "/".join([category, model])

        if DS.GT_DENSE_PC in self.spec:
            data[DS.GT_DENSE_PC] = gt_points

        if DS.GT_DENSE_NORMAL in self.spec:
            data[DS.GT_DENSE_NORMAL] = gt_normals

        if DS.INPUT_PC in self.spec:
            data[DS.INPUT_PC] = gt_points

        if DS.TARGET_NORMAL in self.spec:
            data[DS.TARGET_NORMAL] = gt_normals

        if self.transforms is not None:
            data = self.transforms(data, rng)

        return data
