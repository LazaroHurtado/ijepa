# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from logging import getLogger
from typing import Optional, List, Dict

import torch
import random
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.functional import F
from torchaudio.functional import resample

import torchaudio.transforms as T
from datasets import load_dataset

_GLOBAL_SEED = 0
logger = getLogger()


def make_voxceleb2(
    batch_size,
    mask_collator_fn,
    pin_mem=True,
    num_workers=8,
    world_size=1,
    rank=0,
    split="train",
    drop_last=True,
):
    dataset = VoxCeleb(
        split=split
    )
    logger.info('VoxCeleb2 dataset created')
    dist_sampler = DistributedSampler(
        dataset=dataset,
        num_replicas=world_size,
        shuffle=(split=="train"),
        rank=rank)
    data_loader = DataLoader(
        dataset,
        shuffle=False,
        collate_fn=mask_collator_fn,
        sampler=dist_sampler,
        batch_size=batch_size,
        drop_last=drop_last,
        pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=False)
    logger.info('VoxCeleb2 unsupervised data loader created')

    return dataset, data_loader, dist_sampler


class VoxCeleb(Dataset):
    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 1024,
        hop_length: int = 256,
        n_mels: int = 80,
        f_min: float = 0.0,
        f_max: Optional[float] = None,
        crop_timesteps: int = 96,
        split: str = "train",
    ):
        super().__init__()
        ds = load_dataset("acul3/voxceleb2", data_dir="data/", num_proc=6)["train"]
        ds = ds.remove_columns(["transcription", "gender"])
        
        if split in ["train", "validation", "test"]:
            split = ds.train_test_split(test_size=0.1, seed=42)
            if split == "train":
                self.ds = split["train"]
            else:
                self.ds = split["test"]
        elif split == "all":
            self.ds = ds

        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.f_min = f_min
        self.f_max = f_max or (self.sample_rate / 2.0)
        self.crop_timesteps = crop_timesteps

        # Mel + log transform
        self.mel_transform = T.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            f_min=self.f_min,
            f_max=self.f_max,
            power=2.0)
        self.db_transform = T.AmplitudeToDB(top_db=80.0)

        all_speaker_ids = self.ds["speaker_id"]
        self.id2idx, self.num_speakers = self.build_one_hot_map(all_speaker_ids)

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.ds[idx]

        # Decode audio_path column (should be AudioDecoder or similar)
        audio_decoder = item["audio_path"]
        # audio_decoder is e.g. a TorchCodec AudioDecoder object
        waveform, sr = audio_decoder.get_all_samples().data, audio_decoder.get_all_samples().sample_rate
        # If waveform is numpy, convert to tensor
        if isinstance(waveform, torch.Tensor):
            wav = waveform.float()
        else:
            wav = torch.from_numpy(waveform).float()

        # If multi-channel, mix to mono
        if wav.ndim > 1:
            wav = wav.mean(dim=0, keepdim=True)
        else:
            wav = wav.unsqueeze(0)

        # Resample if needed
        if sr != self.sample_rate:
            wav = resample(wav, orig_freq=sr, new_freq=self.sample_rate)

        # Compute mel spectrogram
        mel = self.mel_transform(wav)             # shape: (1, n_mels, T_frames)
        log_mel = self.db_transform(mel).squeeze(0)  # shape: (n_mels, T_frames)

        T_frames = log_mel.shape[1]
        if T_frames < self.crop_timesteps:
            # If too short, pad time dimension with zeros
            pad_amount = self.crop_timesteps - T_frames
            log_mel = F.pad(log_mel, (0, pad_amount), mode='constant', value=0.0)
            T_frames = self.crop_timesteps
        
        start = random.randint(0, T_frames - self.crop_timesteps)
        log_mel = log_mel[:, start:start + self.crop_timesteps]  # shape: (n_mels, crop_timesteps)

        speaker_id = self.id2idx[item["speaker_id"]]

        return {
            "mel": log_mel.unsqueeze(0),
            "speaker_id": speaker_id,
        }
    
    def build_one_hot_map(self, speaker_id_list):
        unique_ids = set(speaker_id_list)
        id2idx = {spid: idx for idx, spid in enumerate(unique_ids)}
        num_classes = len(unique_ids)
        return id2idx, num_classes
    

def mel_collate_fn(batch: List[Dict[str, torch.Tensor]]):
    mels = torch.stack([b["mel"] for b in batch], dim=0)
    speaker_ids = torch.tensor([b["speaker_id"] for b in batch], dtype=torch.long)

    collated = {
        "mel": mels,
        "speaker_id": speaker_ids,
    }
    return collated
