import argparse
import importlib
from pathlib import Path

import pytorch_lightning as pl
from pycg import exp


def get_default_parser():
    default_parser = argparse.ArgumentParser(add_help=False)
    default_parser = pl.Trainer.add_argparse_args(default_parser)
    return default_parser


if __name__ == '__main__':
    pl.seed_everything(0)
    parser = exp.ArgumentParserX(base_config_path='configs/default/test.yaml', parents=[get_default_parser()])
    parser.add_argument('--ckpt', type=str, required=False, help='Path to ckpt file.')
    parser.add_argument('--record', nargs='*', help='Whether or not to store evaluation data. add name to specify save path.')
    parser.add_argument('--focus', type=str, default="none", help='Sample to focus')

    known_args = parser.parse_known_args()[0]
    if known_args.ckpt is not None:
        model_yaml_path = Path(known_args.ckpt).parent.parent / "hparams.yaml"
        model_args = exp.parse_config_yaml(model_yaml_path)
    else:
        model_args = None

    args = parser.parse_args(additional_args=model_args)
    if args.gpus is None:
        args.gpus = 1
    args.max_epochs = 1

    trainer = pl.Trainer.from_argparse_args(argparse.Namespace(**args), logger=None)
    net_module = importlib.import_module("models." + args.model).Model
    ckpt_path = args.ckpt

    if ckpt_path is not None:
        net_model = net_module.load_from_checkpoint(ckpt_path, hparams=args)
    else:
        net_model = net_module(args)
    test_result = trainer.test(net_model)
    net_model.print_test_logs()
