#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from typing import Tuple, Optional

import pytorch_lightning
import hydra
from omegaconf import DictConfig

from torch import nn
from torch.utils.data import DataLoader

from data import label_sets
from wav2letter import Wav2Letter
from jasper import Jasper
from data.data_loader import WavDataset, BatchWavDataLoader
# from torch.profiler import profile, record_function, ProfilerActivity

name_to_model = {
    "jasper": Jasper,
    "wav2letter": Wav2Letter
}


def get_data_loaders(labels, cfg):
    sample_rate = cfg.audio_conf.sample_rate
    train_dataset = WavDataset(cfg.train_manifest, sample_rate, labels)
    train_batch_loader = BatchWavDataLoader(train_dataset, batch_size=cfg.batch_size, num_workers=cfg.num_workers)
    eval_dataset = WavDataset(cfg.val_manifest, sample_rate, labels)
    val_batch_loader = BatchWavDataLoader(eval_dataset, batch_size=cfg.batch_size, num_workers=cfg.num_workers)
    return train_batch_loader, val_batch_loader


_rval: Optional[Tuple[pytorch_lightning.Trainer, nn.Module, DataLoader, DataLoader]] = None
_cfg = None


@hydra.main(config_path='configuration', config_name='config', version_base='1.1')
def main(cfg: DictConfig):
    if type(cfg.model.labels) is str:
        cfg.model.labels = label_sets.labels_map[cfg.model.labels]
    global _cfg, _rval
    _cfg = cfg
    train_loader, val_loader = get_data_loaders(cfg.model.labels, cfg.data)
    model = name_to_model[cfg.model.name](cfg.model)
    trainer = pytorch_lightning.Trainer(**cfg.trainer)
    _rval = (trainer, model, train_loader, val_loader)
    trainer.fit(model, train_loader, val_loader)


if __name__ == '__main__':
    main()
