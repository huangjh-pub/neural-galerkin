import importlib
import shutil
import tempfile
from pathlib import Path

from pycg import exp
import pytorch_lightning as pl


# Monkey-patch `extract_batch_size` to not raise warning from weird tensor sizes
def extract_bs(self, *args, **kwargs):
    batch_size = 1
    self.batch_size = batch_size
    return batch_size


pl.trainer.connectors.logger_connector.result._ResultCollection._extract_batch_size = extract_bs

from omegaconf import OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.callbacks.base import Callback
from pytorch_lightning.loggers import TensorBoardLogger


class CopyModelFileCallback(Callback):
    """Copy the model definition file to checkpoint directory"""

    def __init__(self):
        self.source_path = None
        self.target_path = None

    def on_train_start(self, trainer, pl_module):
        if self.source_path is not None and self.target_path is not None:
            if self.target_path.parent.exists():
                shutil.move(self.source_path, self.target_path)


if __name__ == '__main__':
    pl.seed_everything(0)

    # PROGRAM args will NOT be saved for a checkpoints.
    program_parser = exp.argparse.ArgumentParser()
    program_parser.add_argument('--resume', action='store_true', help='Continue training. Use hparams.yaml file.')
    program_parser.add_argument('--validate_first', action='store_true',
                                help='Do a full validation with logging before training starts.')
    program_parser = pl.Trainer.add_argparse_args(program_parser)
    program_args, other_args = program_parser.parse_known_args()
    if program_args.max_epochs is None:
        program_args.max_epochs = 50

    # MODEL args include: --lr, --num_layers, etc. (everything defined in YAML)
    #   These use AP-X module, which accepts CLI and YAML inputs.
    #   These args will be saved as hyper-params.
    model_parser = exp.ArgumentParserX(base_config_path='configs/default/train.yaml')
    model_args = model_parser.parse_args(other_args)
    hyper_path = model_args.hyper
    del model_args["hyper"]

    # Resuming stuff.
    last_ckpt_path = Path(hyper_path).parent / "checkpoints" / "last.ckpt"
    if not last_ckpt_path.exists() or not program_args.resume:
        last_ckpt_path = None
    logger_version_num = Path(hyper_path).parent.name if program_args.resume else None

    # Set checkpoint auto-save options.
    logger = TensorBoardLogger('checkpoints/', name=model_args.name,
                               version=logger_version_num, default_hp_metric=False)
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        filename='{epoch:02d}-{val_loss:.2f}',
        save_last=True, save_top_k=1,
        mode='min'
    )
    lr_record_callback = LearningRateMonitor(logging_interval='step')
    copy_model_file_callback = CopyModelFileCallback()

    # Build trainer
    trainer = pl.Trainer.from_argparse_args(
        program_args,
        callbacks=[checkpoint_callback, lr_record_callback, copy_model_file_callback],
        logger=logger,
        log_every_n_steps=20,
        resume_from_checkpoint=last_ckpt_path,
        check_val_every_n_epoch=1,
        accelerator='gpu',
        accumulate_grad_batches=model_args.get('accumulate_grad_batches', 1))
    net_module = importlib.import_module("models." + model_args.model).Model
    net_model = net_module(model_args)
    trainer_log_dir = trainer.logger.log_dir

    print(" >>>> ======= MODEL HYPER-PARAMETERS ======= <<<< ")
    print("Checkpoint Directory is in:", trainer_log_dir)
    print(OmegaConf.to_yaml(net_model.hparams, resolve=True))
    print(" >>>> ====================================== <<<< ")

    # Copy model file to a temporary location.
    temp_py_path = Path(tempfile._get_default_tempdir()) / next(tempfile._get_candidate_names())
    if trainer_log_dir:
        shutil.copy(f"models/{model_args.model.replace('.', '/')}.py", temp_py_path)
        copy_model_file_callback.source_path = temp_py_path
        copy_model_file_callback.target_path = Path(trainer_log_dir) / "model.py"

    # Start training.
    if program_args.validate_first:
        if last_ckpt_path is not None:
            net_model = net_module.load_from_checkpoint(last_ckpt_path, hparams=model_args)
        trainer.validate(net_model)
    trainer.fit(net_model)
