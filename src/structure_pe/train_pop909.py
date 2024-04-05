import argparse
import logging
import os

import neptune.new as neptune
import pytorch_lightning as pl
import yaml
from confugue import Configuration

from structure_pe.datasets import PianoRollPop909DM
from structure_pe.neptune_logger import NeptuneWrapperLogger
from structure_pe.structureformer import StructureFormer

logging.getLogger("neptune.new.internal.operation_processors.async_operation_processor").setLevel(logging.CRITICAL)

LOGGER = logging.getLogger('train')


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', type=str, required=True)
    parser.add_argument('--name', type=str, default=None)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # when running multiple experiments with different LRs
    parser.add_argument('--lr', type=float)
    parser.add_argument('--root', type=str)
    # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    args = parser.parse_args()

    cfg_path = os.path.join(args.model_dir, 'config.yaml')
    cfg = Configuration.from_yaml_file(cfg_path)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # set the correct LR
    cfg["model"]["lr"] = args.lr
    cfg['datamodule']['root'] = args.root
    # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    neptune_run = neptune.init_run(
        # change this to your project
        project = "manvi/binarization",
        # project = "username/project",
        name = args.name or args.model_dir,
        mode="debug",
        source_files = []
    )
    neptune_run['config'] = cfg.get()
    neptune_logger = NeptuneWrapperLogger(neptune_run)

    seed = cfg.get('seed', 0)
    pl.seed_everything(seed)

    print("\nLoading data...")
    dm = cfg['datamodule'].configure(PianoRollPop909DM)

    print("\nInitializing system...")
    model = cfg['model'].configure(StructureFormer)

    run_path = os.path.join(args.model_dir, neptune_run['sys/id'].fetch())

    trainer = cfg['trainer'].configure(
        pl.Trainer,
        gpus=-1,
        deterministic=True,
        logger=neptune_logger,
        callbacks=[
            # checkpoint at best val loss and at the end of training epochs
            pl.callbacks.ModelCheckpoint(
                monitor='val/loss',
                filename="val_{epoch}-{step}",
                save_weights_only=True,
                mode="min",
                dirpath=run_path),
            pl.callbacks.ModelCheckpoint(
                save_weights_only=True,
                filename="latest_{epoch}-{step}",
                save_on_train_epoch_end=True,
                dirpath=run_path)
        ],
    )
    cfg.get_unused_keys(warn=True)

    print("\n--------------------------------------------------")
    print("\t\tConfiguration")
    print("--------------------------------------------------")
    print(yaml.dump(cfg.get()))
    print("\n--------------------------------------------------")
    print("--------------------------------------------------\n\n")

    trainer.fit(model=model, datamodule=dm)


if __name__ == '__main__':
    main()
