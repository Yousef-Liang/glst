# Copyright (c) 2026, Yousef Liang, Ryan Ardalan, Yarden Goraly, Chenhao Zhu, Lizzi (Yue) Yin, Jonas Frey, Claire Tomlin, Marco Pavone, and Shankar Sastry.
# SPDX-License-Identifier: BSD-3-Clause

"""Navigation policy: local height map, latent global map, and proprioception."""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn

from .actor_critic import ActorCritic


@dataclass(frozen=True)
class ObservationConfig:
    """Flat observation layout consumed by the policy.

    The vector is the concatenation, in this order, of:

    * local height map and mask, each ``local_h`` by ``local_w``
    * latent global map, ``latent_global_c`` by ``latent_global_h`` by ``latent_global_w``
    * base linear velocity (3), base angular velocity (3), goal (4), past positions
    """

    local_h: int = 32
    local_w: int = 32

    latent_global_c: int = 9
    latent_global_h: int = 32
    latent_global_w: int = 32
    latent_resolution_m: float = 0.4

    base_lin_vel_dim: int = 3
    base_ang_vel_dim: int = 3
    goal_dim: int = 4
    past_positions_dim: int = 82 * 2

    @property
    def local_pair_dim(self) -> int:
        return 2 * self.local_h * self.local_w

    @property
    def latent_global_dim(self) -> int:
        return self.latent_global_c * self.latent_global_h * self.latent_global_w

    @property
    def vector_dim(self) -> int:
        return self.base_lin_vel_dim + self.base_ang_vel_dim + self.goal_dim + self.past_positions_dim

    @property
    def total_obs_dim(self) -> int:
        return self.local_pair_dim + self.latent_global_dim + self.vector_dim


@dataclass(frozen=True)
class CNNEncoderConfig:
    input_h: int
    input_w: int
    out_dim: int
    channels: Sequence[int]


@dataclass(frozen=True)
class AttentionGlobalEncoderConfig:
    input_c: int = 9
    input_h: int = 32
    input_w: int = 32
    goal_dim: int = 4
    latent_resolution_m: float = 0.4
    sigma_m: float = 1.0

    d_model: int = 128
    out_dim: int = 256
    nhead: int = 4
    num_layers: int = 1
    ff_dim: int = 512


OBS_CFG = ObservationConfig()

LOCAL_CNN_CFG = CNNEncoderConfig(
    input_h=OBS_CFG.local_h,
    input_w=OBS_CFG.local_w,
    out_dim=128,
    channels=(16, 32),
)

GLOBAL_ATTENTION_CFG = AttentionGlobalEncoderConfig(
    input_c=OBS_CFG.latent_global_c,
    input_h=OBS_CFG.latent_global_h,
    input_w=OBS_CFG.latent_global_w,
    goal_dim=OBS_CFG.goal_dim,
    latent_resolution_m=OBS_CFG.latent_resolution_m,
    sigma_m=1.5,
    d_model=64,
    out_dim=128,
    nhead=4,
    num_layers=1,
    ff_dim=128,
)


class HeightMapTensorHelper:
    @staticmethod
    def split_flat_pair(flat_pair: torch.Tensor, h: int, w: int):
        k = h * w
        if flat_pair.ndim != 2 or flat_pair.shape[1] != 2 * k:
            raise ValueError(f"Expected [B, {2 * k}], got {tuple(flat_pair.shape)}")
        b = flat_pair.shape[0]
        height = flat_pair[:, :k].view(b, 1, h, w)
        mask = flat_pair[:, k:].view(b, 1, h, w)
        return height, mask


class ObservationParser:
    def __init__(self, cfg: ObservationConfig):
        i = 0
        self.local_pair_slice = slice(i, i + cfg.local_pair_dim)
        i += cfg.local_pair_dim

        self.latent_global_slice = slice(i, i + cfg.latent_global_dim)
        i += cfg.latent_global_dim

        self.base_lin_vel_slice = slice(i, i + cfg.base_lin_vel_dim)
        i += cfg.base_lin_vel_dim

        self.base_ang_vel_slice = slice(i, i + cfg.base_ang_vel_dim)
        i += cfg.base_ang_vel_dim

        self.goal_slice = slice(i, i + cfg.goal_dim)
        i += cfg.goal_dim

        self.past_positions_slice = slice(i, i + cfg.past_positions_dim)
        i += cfg.past_positions_dim

        self.total_dim = i

    def parse(self, obs_flat: torch.Tensor) -> dict[str, torch.Tensor]:
        if obs_flat.ndim != 2:
            raise ValueError(f"Expected [B, N], got {tuple(obs_flat.shape)}")

        return {
            "local_pair": obs_flat[:, self.local_pair_slice],
            "latent_global": obs_flat[:, self.latent_global_slice],
            "base_lin_vel": obs_flat[:, self.base_lin_vel_slice],
            "base_ang_vel": obs_flat[:, self.base_ang_vel_slice],
            "goal": obs_flat[:, self.goal_slice],
            "past_positions": obs_flat[:, self.past_positions_slice],
        }


class CNNEncoder(nn.Module):
    """Encode a local height map and its visibility mask."""

    def __init__(self, cfg: CNNEncoderConfig):
        super().__init__()
        self.h = cfg.input_h
        self.w = cfg.input_w

        layers = []
        in_ch = 2
        for out_ch in cfg.channels:
            layers += [
                nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1),
                nn.ELU(),
            ]
            in_ch = out_ch

        self.backbone = nn.Sequential(*layers, nn.Flatten())

        with torch.no_grad():
            dummy = torch.zeros(1, 2, self.h, self.w)
            flat_dim = self.backbone(dummy).shape[1]

        self.head = nn.Linear(flat_dim, cfg.out_dim)

    def forward(self, flat_pair: torch.Tensor) -> torch.Tensor:
        height, mask = HeightMapTensorHelper.split_flat_pair(flat_pair, self.h, self.w)
        x = torch.cat([height, mask], dim=1)
        return self.head(self.backbone(x))


class LatentGlobalAttentionEncoder(nn.Module):
    """Goal-conditioned attention over the latent global map.

    The latent map is ``[B, 9, 32, 32]``. A goal heatmap is concatenated as an
    extra channel, a CNN produces a ``16 x 16`` token grid, and one transformer
    layer plus attention pooling returns a feature of size ``out_dim``.
    """

    def __init__(self, cfg: AttentionGlobalEncoderConfig):
        super().__init__()
        self.cfg = cfg
        self.c = cfg.input_c
        self.h = cfg.input_h
        self.w = cfg.input_w
        self.d_model = cfg.d_model

        self.cnn = nn.Sequential(
            nn.Conv2d(cfg.input_c + 1, 64, kernel_size=3, stride=1, padding=1),
            nn.ELU(),
            nn.Conv2d(64, cfg.d_model, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(cfg.d_model, cfg.d_model, kernel_size=3, stride=1, padding=1),
            nn.ELU(),
        )

        self.token_h = cfg.input_h // 2
        self.token_w = cfg.input_w // 2
        self.num_tokens = self.token_h * self.token_w

        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_tokens, cfg.d_model))

        enc_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            dim_feedforward=cfg.ff_dim,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.self_attn = nn.TransformerEncoder(enc_layer, num_layers=cfg.num_layers)

        self.pool_query = nn.Parameter(torch.zeros(1, 1, cfg.d_model))
        self.out = nn.Sequential(
            nn.LayerNorm(cfg.d_model),
            nn.Linear(cfg.d_model, cfg.out_dim),
            nn.ELU(),
        )

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.pool_query, std=0.02)

    def _goal_heatmap(self, goal: torch.Tensor) -> torch.Tensor:
        """Render ``goal[:, :2]`` (meters, map-centered) as a Gaussian on the grid."""
        batch = goal.shape[0]
        device = goal.device
        dtype = goal.dtype

        goal_xy_m = goal[:, :2]

        cx = (self.w - 1) / 2.0
        cy = (self.h - 1) / 2.0

        goal_x = goal_xy_m[:, 0] / self.cfg.latent_resolution_m + cx
        goal_y = goal_xy_m[:, 1] / self.cfg.latent_resolution_m + cy

        xs = torch.arange(self.w, device=device, dtype=dtype)
        ys = torch.arange(self.h, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")

        dx = xx.unsqueeze(0) - goal_x.view(batch, 1, 1)
        dy = yy.unsqueeze(0) - goal_y.view(batch, 1, 1)

        sigma_cells = self.cfg.sigma_m / self.cfg.latent_resolution_m
        heatmap = torch.exp(-(dx * dx + dy * dy) / (2.0 * sigma_cells * sigma_cells))

        return heatmap.unsqueeze(1)

    def forward(self, flat_latent: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        expected_dim = self.c * self.h * self.w
        if flat_latent.ndim != 2 or flat_latent.shape[1] != expected_dim:
            raise ValueError(f"Expected [B, {expected_dim}], got {tuple(flat_latent.shape)}")

        batch = flat_latent.shape[0]

        latent = flat_latent.view(batch, self.c, self.h, self.w)
        goal_map = self._goal_heatmap(goal)

        x = torch.cat([latent, goal_map], dim=1)
        x = self.cnn(x)

        x = x.flatten(2).transpose(1, 2)
        x = x + self.pos_embed

        x = self.self_attn(x)

        query = self.pool_query.expand(batch, -1, -1)
        attn = torch.softmax(
            torch.matmul(query, x.transpose(1, 2)) / math.sqrt(self.d_model),
            dim=-1,
        )

        pooled = torch.matmul(attn, x).squeeze(1)
        return self.out(pooled)


class ActorCriticSpatioTemporalMLPWithLatentGlobal(ActorCritic):
    """GLST navigation policy.

    ``num_actor_obs`` and ``num_critic_obs`` are accepted so this class can be
    constructed the same way as an rsl_rl actor-critic. The network replaces
    those sizes with the fused encoder width from :class:`ObservationConfig`.

    The default MLP widths are the trained policy: ``[512, 256, 128]``.
    """

    is_recurrent = False

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        actor_hidden_dims=(512, 256, 128),
        critic_hidden_dims=(512, 256, 128),
        activation="elu",
        init_noise_std=0.5,
        **kwargs,
    ):
        if kwargs:
            warnings.warn(f"Ignored kwargs: {kwargs.keys()}")

        self.obs_cfg = OBS_CFG
        self.parser = ObservationParser(self.obs_cfg)

        local_feat_dim = LOCAL_CNN_CFG.out_dim
        latent_global_feat_dim = GLOBAL_ATTENTION_CFG.out_dim
        vector_feat_dim = self.obs_cfg.vector_dim
        fused_dim = local_feat_dim + latent_global_feat_dim + vector_feat_dim

        super().__init__(
            num_actor_obs=fused_dim,
            num_critic_obs=fused_dim,
            num_actions=num_actions,
            actor_hidden_dims=actor_hidden_dims,
            critic_hidden_dims=critic_hidden_dims,
            activation=activation,
            init_noise_std=init_noise_std,
        )

        self.local_encoder = CNNEncoder(LOCAL_CNN_CFG)
        self.latent_global_encoder = LatentGlobalAttentionEncoder(GLOBAL_ATTENTION_CFG)

    def _encode(self, observations: torch.Tensor) -> torch.Tensor:
        parts = self.parser.parse(observations)
        vector_obs = torch.cat(
            [
                parts["base_lin_vel"],
                parts["base_ang_vel"],
                parts["goal"],
                parts["past_positions"],
            ],
            dim=-1,
        )
        z_local = self.local_encoder(parts["local_pair"])
        z_global = self.latent_global_encoder(parts["latent_global"], parts["goal"])
        return torch.cat([z_local, z_global, vector_obs], dim=-1)

    def act(self, observations, masks=None, hidden_states=None):
        return super().act(self._encode(observations))

    def act_inference(self, observations):
        return super().act_inference(self._encode(observations))

    def evaluate(self, critic_observations, masks=None, hidden_states=None):
        return super().evaluate(self._encode(critic_observations))

    def get_hidden_states(self):
        return None, None

    def reset(self, dones=None):
        return


def build_policy(num_actions: int = 3, **kwargs) -> ActorCriticSpatioTemporalMLPWithLatentGlobal:
    """Construct the navigation policy. ``num_actions`` is 3 for an SE(2) velocity command."""
    obs_dim = OBS_CFG.total_obs_dim
    return ActorCriticSpatioTemporalMLPWithLatentGlobal(obs_dim, obs_dim, num_actions, **kwargs)
