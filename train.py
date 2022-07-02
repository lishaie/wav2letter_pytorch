#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from typing import Tuple, Optional, Dict, Union

import pandas as pd
import pytorch_lightning
from pytorch_lightning import loggers as pl_loggers
import hydra
from omegaconf import DictConfig, OmegaConf

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


def get_hparams(cfg: DictConfig) -> Dict[str, Union[str, int, float]]:
    cfg = OmegaConf.to_object(cfg.copy())
    cfg['model'].pop('layers')
    cfg['model'].pop('labels')
    cfg['model']['decoder'].pop('labels')
    cfg = pd.json_normalize(cfg, sep='.').to_dict()
    return cfg


@hydra.main(config_path='configuration', config_name='config', version_base='1.1')
def main(cfg: DictConfig):
    if type(cfg.model.labels) is str:
        cfg.model.labels = label_sets.labels_map[cfg.model.labels]
    global _cfg, _rval
    _cfg = cfg

    log_dir = f'lightning_logs'
    opt_name = cfg.model.optimizer._target_.split(".")[-1]
    exp_name = f'bs{cfg.data.batch_size:03d}_{opt_name}{cfg.model.optimizer.lr:1.3f}'
    tb_logger = pl_loggers.TensorBoardLogger(save_dir=log_dir, name=exp_name, version=None, default_hp_metric=True)
    hparams = get_hparams(cfg)
    tb_logger.log_hyperparams(params=hparams)

    train_loader, val_loader = get_data_loaders(cfg.model.labels, cfg.data)
    model = name_to_model[cfg.model.name](cfg.model)
    trainer = pytorch_lightning.Trainer(**cfg.trainer, logger=tb_logger)
    _rval = (trainer, model, train_loader, val_loader)
    trainer.fit(model, train_loader, val_loader)


if __name__ == '__main__':
    main()
