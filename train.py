#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from typing import Tuple

import pytorch_lightning
import hydra
import torch.utils.data
from omegaconf import DictConfig
# from omegaconf import OmegaConf
# from hydra.utils import instantiate
from torch import nn
from torch.utils.data import DataLoader

from data import label_sets
from wav2letter import Wav2Letter
from jasper import Jasper
from data.data_loader import SpectrogramDataset, BatchAudioDataLoader

name_to_model = {
    "jasper": Jasper,
    "wav2letter": Wav2Letter
}


def get_data_loaders(labels, cfg):
    train_dataset = SpectrogramDataset(cfg.train_manifest, cfg.audio_conf, labels, mel_spec=cfg.mel_spec)
    train_batch_loader = BatchAudioDataLoader(train_dataset, batch_size=cfg.batch_size, num_workers=cfg.num_workers)
    eval_dataset = SpectrogramDataset(cfg.val_manifest, cfg.audio_conf, labels, mel_spec=cfg.mel_spec)
    val_batch_loader = BatchAudioDataLoader(eval_dataset, batch_size=cfg.batch_size, num_workers=cfg.num_workers)
    return train_batch_loader, val_batch_loader


rval: Tuple[pytorch_lightning.Trainer, nn.Module, DataLoader, DataLoader] = None


@hydra.main(config_path='configuration', config_name='config', version_base='1.1')
def main(cfg: DictConfig):
    if type(cfg.model.labels) is str:
        cfg.model.labels = label_sets.labels_map[cfg.model.labels]
    train_loader, val_loader = get_data_loaders(cfg.model.labels, cfg.data)
    model = name_to_model[cfg.model.name](cfg.model)
    trainer = pytorch_lightning.Trainer(**cfg.trainer)
    global rval
    rval = (trainer, model, train_loader, val_loader)
    trainer.fit(model, train_loader, val_loader)


if __name__ == '__main__':
    main()
