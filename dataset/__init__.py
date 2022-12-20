from .shapenet import ShapeNetDataset
from .matterport import MatterportDataset


def build_dataset(name: str, spec, hparams, kwargs: dict):
    return eval(name)(**kwargs, spec=spec, hparams=hparams)
