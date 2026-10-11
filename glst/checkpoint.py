# Copyright (c) 2026, Yousef Liang, Ryan Ardalan, Yarden Goraly, Chenhao Zhu, Lizzi (Yue) Yin, Jonas Frey, Claire Tomlin, Marco Pavone, and Shankar Sastry.
# SPDX-License-Identifier: BSD-3-Clause

"""Checkpoint loading for GLST modules."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn


def load_checkpoint(
    module: nn.Module,
    path: str | Path,
    map_location: str | torch.device = "cpu",
    strict: bool = False,
    zero_missing: bool = False,
):
    """Load a ``.pt`` file into ``module``.

    Accepts a raw state dict or a dict with ``model_state_dict`` (policy
    checkpoints) or ``state_dict`` (latent-map checkpoints). ``zero_missing``
    sets parameters that were absent from the file to zero.
    """
    payload = torch.load(path, map_location=map_location, weights_only=False)
    state_dict = _unwrap_state_dict(payload)
    incompatible = module.load_state_dict(state_dict, strict=strict)
    if zero_missing:
        loaded = set(state_dict.keys())
        with torch.no_grad():
            for name, param in module.named_parameters():
                if name not in loaded:
                    param.zero_()
            for name, buf in module.named_buffers():
                if name not in loaded:
                    buf.zero_()
    return incompatible


def _unwrap_state_dict(payload) -> dict:
    if isinstance(payload, dict):
        if isinstance(payload.get("model_state_dict"), dict):
            return payload["model_state_dict"]
        if isinstance(payload.get("state_dict"), dict):
            return payload["state_dict"]
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a state dict, got {type(payload).__name__}.")
    return payload
