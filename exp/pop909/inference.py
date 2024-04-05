import os
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from confugue import Configuration

from structure_pe.datasets import PianoRollPop909DM
from structure_pe.structureformer import StructureFormer
import argparse

task = "next_note_prediction"
train_seqlen = 512
test_seqlen = 1024
best_models = {
    'rpe_structure': 245,
    'rpe_structure_sine': 484,
    'rpe_structure_sine_nonstationary/chords': 496
}

exp_config_files = {
    'rpe_structure_sine_nonstationary/chords': ("rpe_structure_sine_nonstationary", "chords")
}

def main():
    base_dir = Path("/path/to/structurePE/exp/pop909")

    for exp in list(best_models.keys()):

        print("\n\nModel: ", exp)
        model_name = "SPE-" + str(best_models[exp])
        print("Name: ", model_name)

        sample_dir = base_dir / "samples" / task / ("train_" + str(train_seqlen) + "-test_" + str(test_seqlen)) / model_name

        if "/" in exp:
            exp_, config_ = exp_config_files[exp]
            cfg_dir = base_dir / exp_ / task / (str(train_seqlen) + "_" + config_)
        else:
            cfg_dir = base_dir / exp / task / str(train_seqlen)
        cfg = Configuration.from_yaml_file(str(cfg_dir / "config.yaml"))

        cfg['datamodule']['root'] = "/path/to/structurePE/data/pop909/data"
        if test_seqlen == 512:
            sample_lens = "0.5,0.5,0.5"
        elif test_seqlen == 1024:
            sample_lens = '1,1,1'
        else:
            sample_lens = '3,3,3'
        cfg['datamodule']['sample_lengths'] = sample_lens
        cfg['datamodule']['use_test_set'] = True
        cfg['datamodule']['batch_size'] = 1
        cfg['model']['savedir'] = str(sample_dir)

        model_dir = Path("/path/to/structurePE/ckpts") / task / model_name
        ckpt_path = list(model_dir.glob('val*.ckpt'))[0]
        print("Checkpoint path: ", ckpt_path)
        ckpt = torch.load(ckpt_path)

        model = cfg['model'].configure(StructureFormer)
        model.load_state_dict(ckpt['state_dict'])
        device = torch.device("cuda")
        model.to(device)

        dm = cfg['datamodule'].configure(
            PianoRollPop909DM
        )
        dl = dm.test_dataloader()

        sample_dir.mkdir(parents=True, exist_ok=True)
        print("Saving to: ", sample_dir)

        trainer = cfg['trainer'].configure(
            pl.Trainer,
            gpus=-1,
            deterministic=True,
        )
        trainer.test(model=model, dataloaders=dl)

    print("Finished!")


if __name__ == '__main__':
    main()