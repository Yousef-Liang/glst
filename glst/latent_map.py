# Copyright (c) 2026, Yousef Liang, Ryan Ardalan, Yarden Goraly, Chenhao Zhu, Lizzi (Yue) Yin, Jonas Frey, Claire Tomlin, Marco Pavone, and Shankar Sastry.
# SPDX-License-Identifier: BSD-3-Clause

"""Latent global map: local height observations fused into a spatial-temporal memory."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .online_map import DifferentiableOnlineGlobalMapBuilder, OnlineMapSpec


@dataclass
class LatentMapConfig:
    """Sizes and device for :class:`LatentGlobalMapModel`.

    ``device`` selects where the online map canvas and the LSTM state live.
    Set it here before construction. ``module.to()`` moves the network weights
    and does not move that canvas.
    """

    batch_size: int = 4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    local_latent_size: int = 8
    global_latent_size: int = 32
    global_map_size: int = 128
    latent_channels: int = 8

    input_resolution: float = 0.1
    downsample_factor: int = 4
    mask_weight_threshold: float = 1e-4
    backproplimit: int = 2


class LocalEncoder(nn.Module):
    def __init__(self, latent_channels: int = 1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(2, 8, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, latent_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GlobalDecoderBackbone(nn.Module):
    def __init__(self, latent_channels: int = 1, global_map_size: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(latent_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(size=(64, 64), mode="bilinear", align_corners=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MapHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.head = nn.Conv2d(8, 1, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


class MaskHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(8, 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 1, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


class ConfidenceHead(nn.Module):
    def __init__(self, in_channels: int = 8) -> None:
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 1, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.head(x)).squeeze(1)


class LatentGlobalMapModel(nn.Module):
    """Online latent global map.

    Each step takes a local height/mask pair and a pose ``(x, y, z, qw, qx, qy, qz)``.
    The returned ``latent_global`` tensor has shape ``(B, latent_channels + 1, 32, 32)``:
    eight spatial channels plus one temporal-memory channel. That tensor is the
    latent-global block of the navigation policy observation.
    """

    def __init__(self, cfg: LatentMapConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.encoder = LocalEncoder(latent_channels=cfg.latent_channels)

        self.decoder_backbone = GlobalDecoderBackbone(
            latent_channels=cfg.latent_channels + 1,
            global_map_size=cfg.global_map_size,
        )
        self.map_head = MapHead()
        self.mask_head = MaskHead()
        self.conf_head = ConfidenceHead(in_channels=8)

        self.latent_resolution = cfg.input_resolution * cfg.downsample_factor

        self.latent_specs = OnlineMapSpec(
            local_size=cfg.local_latent_size,
            global_size=cfg.global_latent_size,
            resolution=self.latent_resolution,
            mask_weight_threshold=cfg.mask_weight_threshold,
            device=cfg.device,
            dtype=torch.float32,
            local_height_is_robot_relative=False,
            output_robot_relative=False,
        )
        self.builder = DifferentiableOnlineGlobalMapBuilder(
            batch_size=cfg.batch_size * cfg.latent_channels,
            spec=self.latent_specs,
        )

        self.temporal_embed = nn.Sequential(
            nn.Flatten(),
            nn.Linear(cfg.latent_channels * cfg.local_latent_size * cfg.local_latent_size, 256),
            nn.ReLU(inplace=True),
        )
        self.temporal_lstm = nn.LSTM(
            input_size=256,
            hidden_size=512,
            batch_first=True,
        )
        self.temporal_to_map = nn.Sequential(
            nn.Linear(512, cfg.global_latent_size * cfg.global_latent_size),
            nn.ReLU(inplace=True),
        )
        self.mem_scale_logit = nn.Parameter(torch.tensor(-2.3))

        self.h_mem_infer: torch.Tensor | None = None
        self.c_mem_infer: torch.Tensor | None = None
        self.infer_batch_size = cfg.batch_size

    def _make_memory_map(
        self,
        latent_feat_t: torch.Tensor,
        h_mem: torch.Tensor,
        c_mem: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = latent_feat_t.shape[0]
        grid = self.cfg.global_latent_size
        emb = self.temporal_embed(latent_feat_t)
        out, (h_mem, c_mem) = self.temporal_lstm(emb.unsqueeze(1), (h_mem, c_mem))
        mem = self.temporal_to_map(out[:, 0]).view(batch, 1, grid, grid)
        return mem, h_mem, c_mem

    def forward(
        self,
        input_local_seq: torch.Tensor,
        local_mask_seq: torch.Tensor,
        pose_seq: torch.Tensor,
        valid_steps: torch.Tensor,
        seq_lens: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        batch, steps = input_local_seq.shape[:2]
        channels = self.cfg.latent_channels
        local_size = self.cfg.local_latent_size
        grid = self.cfg.global_latent_size

        expanded_init_pose = pose_seq[:, 0].unsqueeze(1).expand(batch, channels, 7).reshape(batch * channels, 7)
        self.builder.reset(expanded_init_pose)

        mask_threshold = 0.5 if self.training else 0.2

        all_pred_maps = []
        all_pred_mask_logits = []
        all_pred_mask_probs = []
        all_pred_mask_bins = []
        all_pred_confs = []

        h_mem = torch.zeros(1, batch, 512, device=pose_seq.device)
        c_mem = torch.zeros(1, batch, 512, device=pose_seq.device)
        latent_global = None

        for t in range(steps):
            valid_t = valid_steps[:, t].float().view(batch, 1, 1, 1)
            pose_t = pose_seq[:, t]

            z_cur_4d = pose_t[:, 2].view(batch, 1, 1, 1)
            height_world = input_local_seq[:, t, 0:1] + z_cur_4d
            input_local_world = torch.cat([height_world, input_local_seq[:, t, 1:2]], dim=1)

            latent_feat_t = self.encoder(input_local_world * valid_t)
            mem_t, h_mem, c_mem = self._make_memory_map(latent_feat_t, h_mem, c_mem)
            mem_t = mem_t * valid_t

            latent_mask_t = (
                F.interpolate(local_mask_seq[:, t], size=(local_size, local_size), mode="nearest") > 0.5
            ).float() * valid_t

            flat_feat = latent_feat_t.reshape(batch * channels, local_size, local_size)
            flat_mask = latent_mask_t.expand(batch, channels, local_size, local_size).reshape(batch * channels, local_size, local_size)
            flat_pose = pose_t.unsqueeze(1).expand(batch, channels, 7).reshape(batch * channels, 7)

            self.builder.update(flat_feat, flat_mask, flat_pose)
            feat_global_flat, _, _ = self.builder.get_current_global(flat_pose)

            feat_global = feat_global_flat.reshape(batch, channels, grid, grid)
            scale = torch.sigmoid(self.mem_scale_logit)
            feat_global_aug = torch.cat([feat_global, scale * mem_t], dim=1)
            latent_global = feat_global_aug

            decoded_feat = self.decoder_backbone(feat_global_aug)
            pred_map_t = self.map_head(decoded_feat).squeeze(1)

            detached_feat = decoded_feat.detach()
            pred_mask_logits_t = self.mask_head(detached_feat).squeeze(1)
            pred_mask_prob_t = torch.sigmoid(pred_mask_logits_t)
            pred_mask_bin_t = (pred_mask_prob_t > mask_threshold).float()
            pred_conf_t = self.conf_head(detached_feat)

            z_cur_3d = pose_t[:, 2].view(batch, 1, 1)
            pred_map_t = pred_map_t - z_cur_3d

            all_pred_maps.append(pred_map_t.unsqueeze(1))
            all_pred_mask_logits.append(pred_mask_logits_t.unsqueeze(1))
            all_pred_mask_probs.append(pred_mask_prob_t.unsqueeze(1))
            all_pred_mask_bins.append(pred_mask_bin_t.unsqueeze(1))
            all_pred_confs.append(pred_conf_t.unsqueeze(1))

            if (t + 1) % self.cfg.backproplimit == 0:
                self.builder.world_sum.detach_()
                self.builder.world_weight.detach_()
                h_mem = h_mem.detach()
                c_mem = c_mem.detach()

        if latent_global is None:
            raise RuntimeError("Latent global map was not produced. The sequence length is zero.")
        return {
            "latent_global": latent_global,
            "pred_map": torch.cat(all_pred_maps, dim=1),
            "pred_mask_logits": torch.cat(all_pred_mask_logits, dim=1),
            "pred_mask_prob": torch.cat(all_pred_mask_probs, dim=1),
            "pred_mask_bin": torch.cat(all_pred_mask_bins, dim=1),
            "pred_conf": torch.cat(all_pred_confs, dim=1),
        }

    def reset_inference(
        self,
        initial_pose: torch.Tensor,
        env_ids: torch.Tensor | None = None,
    ) -> None:
        initial_pose = initial_pose.to(device=self.cfg.device, dtype=torch.float32)
        channels = self.cfg.latent_channels

        if env_ids is None:
            batch = initial_pose.shape[0]
            expanded_init_pose = initial_pose.unsqueeze(1).expand(batch, channels, 7).reshape(batch * channels, 7)
            self.builder.reset(expanded_init_pose)
            self.h_mem_infer = torch.zeros(1, batch, 512, device=initial_pose.device)
            self.c_mem_infer = torch.zeros(1, batch, 512, device=initial_pose.device)
            return

        env_ids = torch.as_tensor(env_ids, device=initial_pose.device, dtype=torch.long)
        if env_ids.numel() == 0:
            return

        batch_sub = initial_pose.shape[0]
        expanded_init_pose = initial_pose.unsqueeze(1).expand(batch_sub, channels, 7).reshape(batch_sub * channels, 7)
        expanded_env_ids = env_ids.unsqueeze(1).expand(env_ids.shape[0], channels).reshape(-1)
        self.builder.reset(expanded_init_pose, env_ids=expanded_env_ids)

        full_batch = self.cfg.batch_size
        if self.h_mem_infer is None or self.c_mem_infer is None or self.h_mem_infer.shape[1] != full_batch:
            self.h_mem_infer = torch.zeros(1, full_batch, 512, device=initial_pose.device)
            self.c_mem_infer = torch.zeros(1, full_batch, 512, device=initial_pose.device)

        self.h_mem_infer[:, env_ids, :] = 0.0
        self.c_mem_infer[:, env_ids, :] = 0.0

    @torch.no_grad()
    def forward_step(
        self,
        input_local: torch.Tensor,
        local_mask: torch.Tensor,
        pose: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        input_local = input_local.to(device=self.cfg.device, dtype=torch.float32)
        local_mask = local_mask.to(device=self.cfg.device, dtype=torch.float32)
        pose = pose.to(device=self.cfg.device, dtype=torch.float32)

        batch = input_local.shape[0]
        channels = self.cfg.latent_channels
        local_size = self.cfg.local_latent_size
        grid = self.cfg.global_latent_size

        if self.h_mem_infer is None or self.c_mem_infer is None or self.h_mem_infer.shape[1] != batch:
            raise RuntimeError(
                f"Temporal LSTM state is not initialized for batch {batch}. Call reset_inference(...) first."
            )

        mask_threshold = 0.2
        valid_t = torch.ones(batch, 1, 1, 1, device=input_local.device, dtype=torch.float32)

        z_cur_4d = pose[:, 2].view(batch, 1, 1, 1)
        height_world = input_local[:, 0:1] + z_cur_4d
        input_local_world = torch.cat([height_world, input_local[:, 1:2]], dim=1)

        latent_feat_t = self.encoder(input_local_world * valid_t)
        mem_t, self.h_mem_infer, self.c_mem_infer = self._make_memory_map(
            latent_feat_t, self.h_mem_infer, self.c_mem_infer
        )

        latent_mask_t = (F.interpolate(local_mask, size=(local_size, local_size), mode="nearest") > 0.5).float() * valid_t

        flat_feat = latent_feat_t.reshape(batch * channels, local_size, local_size)
        flat_mask = latent_mask_t.expand(batch, channels, local_size, local_size).reshape(
            batch * channels, local_size, local_size
        )
        flat_pose = pose.unsqueeze(1).expand(batch, channels, 7).reshape(batch * channels, 7)

        self.builder.update(flat_feat, flat_mask, flat_pose)
        feat_global_flat, _, _ = self.builder.get_current_global(flat_pose)
        feat_global = feat_global_flat.reshape(batch, channels, grid, grid)

        scale = torch.sigmoid(self.mem_scale_logit)
        feat_global_aug = torch.cat([feat_global, scale * mem_t], dim=1)

        decoded_feat = self.decoder_backbone(feat_global_aug)
        pred_map = self.map_head(decoded_feat).squeeze(1)

        detached_feat = decoded_feat.detach()
        pred_mask_logits = self.mask_head(detached_feat).squeeze(1)
        pred_mask_prob = torch.sigmoid(pred_mask_logits)
        pred_mask_bin = (pred_mask_prob > mask_threshold).float()
        pred_conf = self.conf_head(detached_feat)

        z_cur_3d = pose[:, 2].view(batch, 1, 1)
        pred_map = pred_map - z_cur_3d

        return {
            "latent_global": feat_global_aug,
            "pred_map": pred_map,
            "pred_mask_logits": pred_mask_logits,
            "pred_mask_prob": pred_mask_prob,
            "pred_mask_bin": pred_mask_bin,
            "pred_conf": pred_conf,
        }
