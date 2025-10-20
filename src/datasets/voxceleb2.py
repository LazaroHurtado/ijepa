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
        rank=rank)
    data_loader = DataLoader(
        dataset,
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
        audio_col: str = "audio_path",
        transcription_col: str = "transcription",
        speaker_col: str = "speaker_id",
        gender_col: str = "gender",
        split: str = "train",
    ):
        super().__init__()
        self.ds = load_dataset("acul3/voxceleb2", data_dir="data/", num_proc=8)[split]
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.f_min = f_min
        self.f_max = f_max or (self.sample_rate / 2.0)
        self.crop_timesteps = crop_timesteps
        self.audio_col = audio_col
        self.transcription_col = transcription_col
        self.speaker_col = speaker_col
        self.gender_col = gender_col

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

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.ds[idx]

        # Decode audio_path column (should be AudioDecoder or similar)
        audio_decoder = item[self.audio_col]
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

        # Random crop of length crop_timesteps
        start = random.randint(0, T_frames - self.crop_timesteps)
        mel_crop = log_mel[:, start:start + self.crop_timesteps].unsqueeze(0)  # shape: (1, n_mels, crop_timesteps)

        speaker_id = int(item[self.speaker_col].replace("id", ""))
        gender = item[self.gender_col]
        transcription = item[self.transcription_col]

        return {
            "mel": mel_crop,                  # Tensor (n_mels, crop_timesteps)
            "speaker_id": torch.tensor(speaker_id, dtype=torch.long),  # Tensor scalar
            "gender": gender,                # leave as string or convert to tensor if encoded
            "transcription": transcription   # leave as string
        }


def mel_collate_fn(batch: List[Dict[str, torch.Tensor]]):
    mels = torch.stack([b["mel"] for b in batch], dim=0)              # (B, n_mels, crop_timesteps)
    speaker_ids = torch.stack([b["speaker_id"] for b in batch], dim=0)  # (B,)
    genders = [b["gender"] for b in batch]
    transcriptions = [b["transcription"] for b in batch]

    collated = {
        "mel": mels,
        "speaker_id": speaker_ids,
        "gender": genders,
        "transcription": transcriptions
    }
    return collated
