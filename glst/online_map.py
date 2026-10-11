# Copyright (c) 2026, Yousef Liang, Ryan Ardalan, Yarden Goraly, Chenhao Zhu, Lizzi (Yue) Yin, Jonas Frey, Claire Tomlin, Marco Pavone, and Shankar Sastry.
# SPDX-License-Identifier: BSD-3-Clause

"""Rolling global map. The stored canvas shifts by integer cells."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class OnlineMapSpec:
    local_size: int = 32
    global_size: int = 128
    resolution: float = 0.1
    mask_weight_threshold: float = 1e-4
    device: str = "cpu"
    dtype: torch.dtype = torch.float32

    local_height_is_robot_relative: bool = True
    output_robot_relative: bool = True


def quat_to_yaw_batch(quat_wxyz: torch.Tensor) -> torch.Tensor:
    qw = quat_wxyz[:, 0]
    qx = quat_wxyz[:, 1]
    qy = quat_wxyz[:, 2]
    qz = quat_wxyz[:, 3]
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return torch.atan2(siny_cosp, cosy_cosp)


def pose_to_xyyaw_batch(pose: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if pose.ndim != 2 or pose.shape[1] != 7:
        raise ValueError(f"Expected pose shape (B, 7), got {tuple(pose.shape)}")
    xy = pose[:, :2]
    yaw = quat_to_yaw_batch(pose[:, 3:7])
    return xy, yaw


def extract_pose_z_batch(pose: torch.Tensor) -> torch.Tensor:
    if pose.ndim != 2 or pose.shape[1] != 7:
        raise ValueError(f"Expected pose shape (B, 7), got {tuple(pose.shape)}")
    return pose[:, 2]


def make_world_index_grid_xy(size: int, resolution: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    coords_1d = torch.arange(size, device=device, dtype=dtype) * resolution
    ys, xs = torch.meshgrid(coords_1d, coords_1d, indexing="ij")
    return torch.stack([xs, ys], dim=-1)


def world_to_local_frame_grid(world_xy: torch.Tensor, frame_xy: torch.Tensor, frame_yaw: torch.Tensor) -> torch.Tensor:
    dx = world_xy[..., 0] - frame_xy[:, None, None, 0]
    dy = world_xy[..., 1] - frame_xy[:, None, None, 1]
    c = torch.cos(frame_yaw)[:, None, None]
    s = torch.sin(frame_yaw)[:, None, None]
    x_local = c * dx + s * dy
    y_local = -s * dx + c * dy
    return torch.stack([x_local, y_local], dim=-1)


def metric_xy_to_grid_sample_norm(xy_metric: torch.Tensor, size: int, resolution: float) -> torch.Tensor:
    pixel = xy_metric / resolution + (size // 2)
    return 2.0 * pixel / (size - 1) - 1.0


def make_centered_grid_xy(size: int, resolution: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    coords_1d = (torch.arange(size, device=device, dtype=dtype) - (size // 2)) * resolution
    ys, xs = torch.meshgrid(coords_1d, coords_1d, indexing="ij")
    return torch.stack([xs, ys], dim=-1)


def local_to_world_frame_batch(points_local: torch.Tensor, frame_xy: torch.Tensor, frame_yaw: torch.Tensor) -> torch.Tensor:
    c = torch.cos(frame_yaw)
    s = torch.sin(frame_yaw)
    rot = torch.stack(
        [
            torch.stack([c, -s], dim=-1),
            torch.stack([s, c], dim=-1),
        ],
        dim=-2,
    )
    return torch.bmm(points_local, rot.transpose(1, 2)) + frame_xy[:, None, :]


class DifferentiableOnlineGlobalMapBuilder:
    """Rolling robot-centered global map.

    The stored canvas is shifted by integer cells only, which avoids repeated
    bilinear resampling of the world memory.
    """

    def __init__(self, batch_size: int, spec: OnlineMapSpec = OnlineMapSpec()) -> None:
        self.batch_size = batch_size
        self.spec = spec
        self.device = torch.device(spec.device)
        self.dtype = spec.dtype

        self.world_index_xy_grid = make_world_index_grid_xy(spec.global_size, spec.resolution, self.device, self.dtype)
        self.query_centered_xy_grid = make_centered_grid_xy(
            spec.global_size, spec.resolution, self.device, self.dtype
        )

        self.world_sum: torch.Tensor | None = None
        self.world_weight: torch.Tensor | None = None
        self.origin_xy: torch.Tensor | None = None
        self.z_ref: torch.Tensor | None = None

    def _center_origin_from_pose(self, pose: torch.Tensor) -> torch.Tensor:
        center_xy, _ = pose_to_xyyaw_batch(pose)
        half_extent = 0.5 * (self.spec.global_size - 1) * self.spec.resolution
        return torch.stack([center_xy[:, 0] - half_extent, center_xy[:, 1] - half_extent], dim=-1)

    def _sample_world_onto_query(self, world_tensor: torch.Tensor, pose: torch.Tensor) -> torch.Tensor:
        self._require_reset()
        frame_xy, frame_yaw = pose_to_xyyaw_batch(pose)
        batch = pose.shape[0]

        query_local = self.query_centered_xy_grid.reshape(1, -1, 2).expand(batch, -1, 2)
        query_world = local_to_world_frame_batch(query_local, frame_xy, frame_yaw)
        query_world = query_world.reshape(batch, self.spec.global_size, self.spec.global_size, 2)

        world_grid = (query_world - self.origin_xy[:, None, None, :]) / self.spec.resolution
        world_grid = 2.0 * world_grid / (self.spec.global_size - 1) - 1.0

        return F.grid_sample(
            world_tensor,
            world_grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )

    def reset(self, initial_pose: torch.Tensor, env_ids: torch.Tensor | None = None) -> None:
        initial_pose = initial_pose.to(device=self.device, dtype=self.dtype)
        new_origin = self._center_origin_from_pose(initial_pose)
        z_init = extract_pose_z_batch(initial_pose)

        if self.world_sum is None or self.world_weight is None or self.origin_xy is None or self.z_ref is None:
            self.world_sum = torch.zeros(
                self.batch_size, self.spec.global_size, self.spec.global_size, device=self.device, dtype=self.dtype
            )
            self.world_weight = torch.zeros_like(self.world_sum)
            self.origin_xy = torch.zeros(self.batch_size, 2, device=self.device, dtype=self.dtype)
            self.z_ref = torch.zeros(self.batch_size, device=self.device, dtype=self.dtype)

            if env_ids is None:
                if initial_pose.shape[0] != self.batch_size:
                    raise ValueError("When env_ids is None, initial_pose must have batch_size rows.")
                self.origin_xy[:] = new_origin
                self.z_ref[:] = z_init
            else:
                env_ids = env_ids.to(device=self.device, dtype=torch.long)
                self.origin_xy[env_ids] = new_origin
                self.z_ref[env_ids] = z_init
            return

        if env_ids is None:
            if initial_pose.shape[0] != self.batch_size:
                raise ValueError("When env_ids is None, initial_pose must have batch_size rows.")
            self.world_sum = torch.zeros_like(self.world_sum)
            self.world_weight = torch.zeros_like(self.world_weight)
            self.origin_xy = new_origin
            self.z_ref = z_init
        else:
            env_ids = env_ids.to(device=self.device, dtype=torch.long)
            self.world_sum[env_ids] = 0.0
            self.world_weight[env_ids] = 0.0
            self.origin_xy[env_ids] = new_origin
            self.z_ref[env_ids] = z_init

    def _require_reset(self) -> None:
        if self.world_sum is None or self.world_weight is None or self.origin_xy is None or self.z_ref is None:
            raise RuntimeError("Call reset(initial_pose) before update/get_current_global.")

    def _shift_tensor_int(self, x: torch.Tensor, shift_x: torch.Tensor, shift_y: torch.Tensor) -> torch.Tensor:
        """shift_x > 0 moves content right. shift_y > 0 moves content up."""
        batch, height, width = x.shape
        out = torch.zeros_like(x)
        for b in range(batch):
            sx = int(shift_x[b].item())
            sy = int(shift_y[b].item())

            src_x0 = max(0, -sx)
            src_x1 = min(width, width - sx) if sx >= 0 else width
            dst_x0 = max(0, sx)
            dst_x1 = min(width, width + sx) if sx < 0 else width

            src_y0 = max(0, -sy)
            src_y1 = min(height, height - sy) if sy >= 0 else height
            dst_y0 = max(0, sy)
            dst_y1 = min(height, height + sy) if sy < 0 else height

            if src_x0 < src_x1 and src_y0 < src_y1 and dst_x0 < dst_x1 and dst_y0 < dst_y1:
                out[b, dst_y0:dst_y1, dst_x0:dst_x1] = x[b, src_y0:src_y1, src_x0:src_x1]
        return out

    def _recenter_canvas(self, pose: torch.Tensor) -> None:
        self._require_reset()
        pose = pose.to(device=self.device, dtype=self.dtype)
        target_origin = self._center_origin_from_pose(pose)
        delta_xy = target_origin - self.origin_xy

        shift_cells_x = -torch.round(delta_xy[:, 0] / self.spec.resolution).to(torch.long)
        shift_cells_y = -torch.round(delta_xy[:, 1] / self.spec.resolution).to(torch.long)

        if torch.all((shift_cells_x == 0) & (shift_cells_y == 0)):
            return

        self.world_sum = self._shift_tensor_int(self.world_sum, shift_cells_x, shift_cells_y)
        self.world_weight = self._shift_tensor_int(self.world_weight, shift_cells_x, shift_cells_y)
        self.origin_xy = self.origin_xy - torch.stack(
            [
                shift_cells_x.to(self.dtype) * self.spec.resolution,
                shift_cells_y.to(self.dtype) * self.spec.resolution,
            ],
            dim=-1,
        )

    def _sample_local_onto_world(self, local_tensor: torch.Tensor, pose: torch.Tensor) -> torch.Tensor:
        self._require_reset()
        frame_xy, frame_yaw = pose_to_xyyaw_batch(pose)
        world_xy = self.origin_xy[:, None, None, :] + self.world_index_xy_grid[None, :, :, :]
        local_xy = world_to_local_frame_grid(world_xy, frame_xy, frame_yaw)
        local_grid = metric_xy_to_grid_sample_norm(local_xy, size=local_tensor.shape[-1], resolution=self.spec.resolution)
        return F.grid_sample(
            local_tensor,
            local_grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )

    def update(self, local_height: torch.Tensor, local_mask: torch.Tensor, pose: torch.Tensor) -> None:
        self._require_reset()
        local_height = local_height.to(device=self.device, dtype=self.dtype)
        local_mask = local_mask.to(device=self.device, dtype=self.dtype)
        pose = pose.to(device=self.device, dtype=self.dtype)

        if local_height.ndim != 3 or local_mask.ndim != 3:
            raise ValueError("local_height and local_mask must have shape (B, S, S).")
        if local_height.shape != local_mask.shape:
            raise ValueError("local_height and local_mask must have the same shape.")
        if local_height.shape[0] != self.batch_size or pose.shape[0] != self.batch_size:
            raise ValueError("Batch size mismatch.")

        self._recenter_canvas(pose)

        sampled_height_sum = self._sample_local_onto_world((local_height * local_mask).unsqueeze(1), pose).squeeze(1)
        sampled_height_weight = self._sample_local_onto_world(local_mask.unsqueeze(1), pose).squeeze(1)

        sampled_height = sampled_height_sum / torch.clamp(sampled_height_weight, min=1e-12)
        sampled_mask = (sampled_height_weight > 0.5).to(self.dtype)

        if self.spec.local_height_is_robot_relative:
            z_cur = extract_pose_z_batch(pose)
            sampled_height = sampled_height + (z_cur - self.z_ref)[:, None, None]

        self.world_sum = self.world_sum + sampled_height * sampled_mask
        self.world_weight = self.world_weight + sampled_mask

    def get_current_global(self, pose: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self._require_reset()
        pose = pose.to(device=self.device, dtype=self.dtype)
        self._recenter_canvas(pose)

        query_sum = self._sample_world_onto_query(self.world_sum.unsqueeze(1), pose).squeeze(1)
        query_weight = self._sample_world_onto_query(self.world_weight.unsqueeze(1), pose).squeeze(1)
        query_map = query_sum / torch.clamp(query_weight, min=1e-12)

        if self.spec.local_height_is_robot_relative and self.spec.output_robot_relative:
            z_cur = extract_pose_z_batch(pose)
            query_map = query_map - (z_cur - self.z_ref)[:, None, None]

        query_mask = (query_weight > self.spec.mask_weight_threshold).to(self.dtype)
        return query_map, query_mask, query_weight

    def step(self, local_height: torch.Tensor, local_mask: torch.Tensor, pose: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        self.update(local_height=local_height, local_mask=local_mask, pose=pose)
        global_map, global_mask, _ = self.get_current_global(pose)
        return global_map, global_mask

    def get_world_state(self) -> Dict[str, torch.Tensor]:
        self._require_reset()
        return {
            "world_sum": self.world_sum.clone(),
            "world_weight": self.world_weight.clone(),
            "origin_xy": self.origin_xy.clone(),
            "z_ref": self.z_ref.clone(),
        }
