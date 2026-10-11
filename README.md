# GLST: Global-Local Spatio-Temporal Memory Decomposition Framework for Online Chained-Goal Navigation

**Authors:** Yousef Liang, Ryan Ardalan, Yarden Goraly, Chenhao Zhu, Lizzi (Yue) Yin, Jonas Frey, Claire Tomlin, Marco Pavone, and Shankar Sastry.

This repository is the model release of GLST. It contains two PyTorch modules:

* **Latent global map.** An online spatial-temporal memory. At each step it reads a local height map, a visibility mask, and the robot pose, and it writes a latent global map.
* **Navigation policy.** A local CNN and a goal-conditioned attention encoder over that latent map, followed by an MLP actor-critic. The action is a 3D SE(2) velocity command.

Training and robot deployment code are planned for a later release. Checkpoints are not included.

## Install

```bash
pip install torch
pip install -e .
```

Python 3.10 or newer is required.

## Latent global map

`input_local` is `(B, 2, 32, 32)`: channel 0 is height, channel 1 is the visibility mask. `local_mask` is `(B, 1, 32, 32)`. `pose` is `(B, 7)` in the order `x, y, z, qw, qx, qy, qz`.

`latent_global` is `(B, 9, 32, 32)`: 8 spatial channels and 1 temporal-memory channel, at 0.4 m per cell. That tensor is the latent-global block of the policy observation.

Set `device` on `LatentMapConfig` before constructing the model. `module.to()` moves the network weights and does not move the map canvas.

```python
import torch
from glst import LatentGlobalMapModel, LatentMapConfig

cfg = LatentMapConfig(batch_size=1, device="cpu")
model = LatentGlobalMapModel(cfg)

pose = torch.tensor([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]])
model.reset_inference(pose)
out = model.forward_step(
    input_local=torch.zeros(1, 2, 32, 32),
    local_mask=torch.zeros(1, 1, 32, 32),
    pose=pose,
)
latent_global = out["latent_global"]  # (1, 9, 32, 32)
```

The trained weights are in `checkpoints/latent_global_map.pt`. `load_checkpoint(..., zero_missing=True)` matches the loader used when this map is frozen inside navigation training: tensors missing from the file are set to zero.

```python
from glst import load_checkpoint

load_checkpoint(model, "checkpoints/latent_global_map.pt", map_location="cpu", zero_missing=True)
```

## Navigation policy

The policy reads one flat vector per environment. With the default :class:`glst.ObservationConfig` its length is 11438:

| Block | Shape | Values |
| --- | --- | --- |
| Local height map, then mask | `2 x 32 x 32` | 2048 |
| Latent global map | `9 x 32 x 32` | 9216 |
| Base linear velocity | 3 | 3 |
| Base angular velocity | 3 | 3 |
| Goal | 4 | 4 |
| Past positions | `82 x 2` | 164 |

`goal[:, :2]` is the goal position in meters, in the robot-centered latent map. The attention encoder turns those two values into a Gaussian heatmap with `sigma = 1.5 m`.

```python
import torch
from glst import build_policy, load_checkpoint

policy = build_policy(num_actions=3)
observation = torch.zeros(1, policy.obs_cfg.total_obs_dim)
action = policy.act_inference(observation)  # (1, 3)

load_checkpoint(policy, "policy.pt", map_location="cpu")
```

`build_policy` uses the trained MLP widths `[512, 256, 128]` and `init_noise_std=0.5`. A policy checkpoint is the `model_state_dict` written by rsl_rl. Module names in that file are `local_encoder`, `latent_global_encoder`, `actor`, `critic`, and `std`.

## License

BSD-3-Clause. See [LICENSE](LICENSE).

`glst/actor_critic.py` is derived from [rsl_rl](https://github.com/leggedrobotics/rsl_rl), copyright ETH Zurich and NVIDIA CORPORATION, and keeps that copyright header. See [NOTICE](NOTICE).

## Acknowledgements

This research was supported in part by the Defense Advanced Research Projects Agency (DARPA) under Agreement No. HR00112490425. The views, opinions, and/or findings expressed are those of the author(s) and should not be interpreted as representing the official views or policies of the Defense Advanced Research Projects Agency (DARPA), or the U.S. Government.

## Citation

```bibtex
@software{liang2026glst,
  author = {Liang, Yousef and Ardalan, Ryan and Goraly, Yarden and Zhu, Chenhao and Yin, Lizzi (Yue) and Frey, Jonas and Tomlin, Claire and Pavone, Marco and Sastry, Shankar},
  title = {GLST: Global-Local Spatio-Temporal Memory Decomposition Framework for Online Chained-Goal Navigation},
  year = {2026},
  url = {https://github.com/Yousef-Liang/glst},
  version = {0.1.0}
}
```
