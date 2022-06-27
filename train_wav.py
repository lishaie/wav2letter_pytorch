#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from typing import Tuple, Optional

import pytorch_lightning
import hydra
from omegaconf import DictConfig

from torch import nn
from torch.utils.data import DataLoader, Dataset

from data import label_sets
from wav2letter import Wav2Letter, RawWav2Letter
from jasper import Jasper

from data.wav_data_loader import WavDataset, WavDataLoader
from torch.profiler import profile, record_function, ProfilerActivity

name_to_model = {
    "jasper": Jasper,
    "wav2letter": Wav2Letter
}


def get_data_loaders(labels, cfg):
    sample_rate = cfg.audio_conf.sample_rate
    train_dataset = WavDataset(cfg.train_manifest, sample_rate, labels)
    train_batch_loader = WavDataLoader(train_dataset, batch_size=cfg.batch_size, num_workers=cfg.num_workers)
    eval_dataset = WavDataset(cfg.val_manifest, sample_rate, labels)
    val_batch_loader = WavDataLoader(eval_dataset, batch_size=cfg.batch_size, num_workers=cfg.num_workers)
    return train_batch_loader, val_batch_loader


rval: Optional[Tuple[pytorch_lightning.Trainer, nn.Module, DataLoader, DataLoader]] = None
_cfg = None


@hydra.main(config_path='configuration', config_name='config', version_base='1.1')
def main(cfg: DictConfig):
    if type(cfg.model.labels) is str:
        cfg.model.labels = label_sets.labels_map[cfg.model.labels]
    global _cfg
    _cfg = cfg
    print(cfg.keys())
    train_loader, val_loader = get_data_loaders(cfg.model.labels, cfg.data)
    model = RawWav2Letter(cfg.model)
    trainer = pytorch_lightning.Trainer(**cfg.trainer)
    global rval
    rval = (trainer, model, train_loader, val_loader)
    # with profile(
    #         activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    #         with_stack=True,
    # ) as prof:
    #     trainer.fit(model, train_loader)
    #
    # # Print aggregated stats
    # print(prof.key_averages(group_by_stack_n=10).table(sort_by="self_cpu_time_total", row_limit=20))
    trainer.fit(model, train_loader, val_loader)


if __name__ == '__main__':
    main()
