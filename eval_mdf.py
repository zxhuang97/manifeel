"""
Evaluate a trained MDF (Multi-modal Diffusion Forcing) model inside the
manifeel IsaacGym container.

The MDF model is loaded from a force_tool_tactile checkpoint. A thin
ManifeelObsManager subclass replaces PolicyObsManager's IsaacLab-specific
data collection so that the real DFMMPolicy can be reused unchanged.

Usage (inside the manifeel container):
    python eval_mdf.py \
        --checkpoint /path/to/mdf/checkpoints/epoch=xx.ckpt \
        --config-dir /path/to/mdf/run_dir \
        --output-dir data/outputs/mdf_eval \
        --isaacgym-cfg isaacgym_config_power_plug.yaml \
        --gpu 0
"""

import isaacgym  # must be first

import sys
import os
import pathlib
import click
import json
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

# ─────────────────────────────────────────────────────────────────────────────
# Manifeel / diffusion-policy imports (available inside container)
# ─────────────────────────────────────────────────────────────────────────────
import hydra
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.common.pytorch_util import dict_apply


# ─────────────────────────────────────────────────────────────────────────────
# force_tool_tactile imports (injected via PYTHONPATH)
# ─────────────────────────────────────────────────────────────────────────────
from algorithms.diffusion_forcing.df_multi_modal import DiffusionForcingMultiModal
from force_tool.utils.data_utils import SmartDict, SmartQueue
from force_tool.policy.mmdf_policy import DFMMPolicy


# ─────────────────────────────────────────────────────────────────────────────
# ManifeelObsManager
# ─────────────────────────────────────────────────────────────────────────────

class ManifeelObsManager:
    """
    Drop-in replacement for PolicyObsManager that works with manifeel's
    IsaacGym (v1) env instead of IsaacLab.

    Replicates the PolicyObsManager public interface used by DFMMPolicy:
        reset(action, reward, cam_encode_func)
        process_step(action, reward, cam_encode_func)
        obs_cur  (SmartDict of shape (B, his_len, D) per modality)

    Data comes from the manifeel obs dict (pushed in from predict_action)
    rather than reading IsaacLab env sensors.

    Modalities handled:
        state  : (B, 7)  ee_pos + ee_quat
        taxim  : (B, 3, H, W) float32 [0-1] — right_tactile_camera_taxim
        wrist  : (B, 3, H, W) float32 [0-1]
        action : (B, 6)  — last action sent to env
    """

    def __init__(
        self,
        his_len: int,
        modalities: list,
        taxim_frame_stride: int = 0,
        taxim_size: tuple = (224, 224),
        wrist_size: tuple = (224, 224),
        gray_taxim: bool = False,
    ):
        self.his_len = his_len
        self.modalities = modalities
        self.use_action = "action" in modalities
        self.use_tactile = "taxim" in modalities
        self.taxim_frame_stride = taxim_frame_stride
        self.taxim_size = taxim_size
        self.wrist_size = wrist_size
        self.gray_taxim = gray_taxim
        self.taxim_frame_history = None

        self.obs_cur = None
        self._pending_obs = None   # set by predict_action before reset/process_step

    # ── called by predict_action to inject the current manifeel obs dict ──

    def set_obs(self, obs_dict: dict):
        """Store the latest manifeel obs dict for use in the next reset/process_step."""
        self._pending_obs = obs_dict

    # ── taxim frame-stride logic (mirrors PolicyObsManager._prepare_taxim) ──

    def _prepare_taxim(self, taxim_raw: torch.Tensor) -> torch.Tensor:
        """taxim_raw: (B, 3, H, W) float32 [0-1]"""
        import torchvision.transforms as transforms

        if taxim_raw.max() > 1.0:
            taxim_raw = taxim_raw / 255.0
        if self.gray_taxim:
            taxim_raw = F.rgb_to_grayscale(taxim_raw).repeat(1, 3, 1, 1)

        cur = transforms.Resize(self.taxim_size, antialias=True)(taxim_raw)

        if self.taxim_frame_stride == 0:
            return cur  # single-frame mode

        if self.taxim_frame_history is None:
            self.taxim_frame_history = [taxim_raw.clone() for _ in range(self.taxim_frame_stride)]

        prev_raw = self.taxim_frame_history[0]
        self.taxim_frame_history = self.taxim_frame_history[1:] + [taxim_raw.clone()]
        prev = transforms.Resize(self.taxim_size, antialias=True)(prev_raw)
        return torch.cat([prev, cur], dim=1)

    # ── extract state/camera from pending obs dict ────────────────────────

    def _get_state_data(self) -> dict:
        obs = self._pending_obs
        # obs["state"] shape: (B, T_obs, 7) — take newest step
        state = obs["state"][:, -1].float()   # (B, 7)
        return {"state": state}

    def _get_camera_data(self) -> dict:
        obs = self._pending_obs
        camera = {}
        if self.use_tactile:
            taxim_raw = obs["right_tactile_camera_taxim"][:, -1].float()  # (B, 3, H, W)
            if taxim_raw.shape[-2:] != tuple(self.taxim_size):
                taxim_raw = F.interpolate(taxim_raw, size=self.taxim_size, mode="bilinear", align_corners=False)
            camera["taxim"] = self._prepare_taxim(taxim_raw)
        if "wrist" in obs:
            wrist_raw = obs["wrist"][:, -1].float()
            if wrist_raw.dtype == torch.uint8 or wrist_raw.max() > 1.0:
                wrist_raw = wrist_raw / 255.0
            if wrist_raw.shape[-2:] != tuple(self.wrist_size):
                wrist_raw = F.interpolate(wrist_raw, size=self.wrist_size, mode="bilinear", align_corners=False)
            camera["wrist"] = wrist_raw
        return camera

    # ── PolicyObsManager public interface ────────────────────────────────

    def reset(self, action, reward, cam_encode_func=None):
        self.taxim_frame_history = None
        state_data = self._get_state_data()
        camera_data = self._get_camera_data()

        if self.use_action:
            state_data["action"] = action.float()

        if cam_encode_func is not None:
            for k, v in camera_data.items():
                camera_data[k] = v.unsqueeze(1)
            camera_data = cam_encode_func(camera_data)
            for k, v in camera_data.items():
                camera_data[k] = v.squeeze(1)

        obs_data = SmartQueue(max_size=self.his_len, reverse_queue=False, backend="torch")
        for k, v in camera_data.items():
            obs_data.add(k, v)
        for k, v in state_data.items():
            obs_data.add(k, v)

        self.obs_cur = obs_data.apply(lambda x: torch.stack(x, 1))
        return self.obs_cur

    def process_step(self, action, reward, cam_encode_func=None):
        state_data = self._get_state_data()
        camera_data = self._get_camera_data()

        if self.use_action:
            state_data["action"] = action.float()

        if cam_encode_func is not None:
            for k, v in camera_data.items():
                camera_data[k] = v.unsqueeze(1)
            camera_data = cam_encode_func(camera_data)
            for k, v in camera_data.items():
                camera_data[k] = v.squeeze(1)

        for k, v in camera_data.items():
            self.obs_cur[k] = torch.roll(self.obs_cur[k], -1, dims=1)
            self.obs_cur[k][:, -1] = v
        for k, v in state_data.items():
            self.obs_cur[k] = torch.roll(self.obs_cur[k], -1, dims=1)
            self.obs_cur[k][:, -1] = v

        return self.obs_cur


# ─────────────────────────────────────────────────────────────────────────────
# MDF Policy wrapper  ── speaks BaseImagePolicy so ManifeelRunner can call it
# ─────────────────────────────────────────────────────────────────────────────

class MDFManifeel(BaseImagePolicy):
    """
    Wraps DFMMPolicy (unchanged) + ManifeelObsManager so the real
    force_tool_tactile inference pipeline runs inside the manifeel container.

    ManifeelRunner calls:
        policy.reset()
        policy.predict_action(obs_dict)  → {'action': (B, n_action_steps, 6)}
    """

    def __init__(
        self,
        model: DiffusionForcingMultiModal,
        his_len: int = 4,
        n_action_steps: int = 1,
        inference_mode: str = "policy",
        sampling_timesteps: int = 10,
        ddim_sampling_eta: float = 0.0,
        clip_noise: float = 6.0,
        scheduling_mode: str = "full_sequence",
        taxim_size: tuple = (224, 224),
        wrist_size: tuple = (224, 224),
        taxim_frame_stride: int = 0,
        precision: str = "bf16",
    ):
        super().__init__()
        self.n_action_steps = n_action_steps
        self._action_queue = None

        # Build ManifeelObsManager first (doesn't need DFMMPolicy yet)
        self._obs_mgr = ManifeelObsManager(
            his_len=his_len,
            modalities=model.modalities,
            taxim_frame_stride=taxim_frame_stride,
            taxim_size=taxim_size,
            wrist_size=wrist_size,
        )

        # Monkeypatch PolicyObsManager so DFMMPolicy.__init__ doesn't call
        # BaseRecorder.__init__ (which requires an IsaacLab env_uw).
        import force_tool.utils.isaac_utils as _iu
        _orig_pom = _iu.PolicyObsManager

        class _StubPOM:
            def __init__(self, *a, **kw):
                pass

        _iu.PolicyObsManager = _StubPOM
        self._dfmm = DFMMPolicy(
            model=model,
            env=None,
            inference_mode=inference_mode,
            his_len=his_len,
            his_mod=model.obs_names + model.action_names,
            clip_noise=clip_noise,
            sampling_timesteps=sampling_timesteps,
            ddim_sampling_eta=ddim_sampling_eta,
            scheduling_mode=scheduling_mode,
            ddim_mode="ddim",
            obs_cfg={},
            precision=precision,
            enable_state_estimation=False,
        )
        _iu.PolicyObsManager = _orig_pom  # restore

        # Wire our obs manager into the real DFMMPolicy
        self._dfmm.obs_manager = self._obs_mgr

    @property
    def device(self):
        return next(self._dfmm.model.parameters()).device

    @property
    def dtype(self):
        return self._dfmm.dtype

    def reset(self):
        self._action_queue = None

    @torch.inference_mode()
    def predict_action(self, obs_dict: dict) -> dict:
        """
        obs_dict from ManifeelRunner (tensors on CPU or device):
            wrist                      : (B, T_obs, 3, H, W)
            right_tactile_camera_taxim : (B, T_obs, 3, H, W) float32 [0-1]
            state                      : (B, T_obs, 7)

        Returns:
            {'action': (B, n_action_steps, 6)}
        """
        device = self.device
        obs_on_device = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                         for k, v in obs_dict.items()}

        self._obs_mgr.set_obs(obs_on_device)

        zero_action = torch.zeros(
            obs_on_device["state"].shape[0], self._dfmm.model.modalities_dim["action"],
            device=device,
        )
        zero_reward = torch.zeros(obs_on_device["state"].shape[0], device=device)

        if self._action_queue is None:
            # First call: reset history
            self._dfmm.reset(zero_action, zero_reward)
            self._action_queue = []
        else:
            # Subsequent calls: push latest obs into history
            self._dfmm.update_obs(zero_action, zero_reward)

        if len(self._action_queue) == 0:
            policy_out = self._dfmm(obs=None, extras=None)
            action_seq = policy_out["action_seq"]      # (plan_horizon, B, 6)
            action_seq = action_seq.permute(1, 0, 2)   # (B, plan_horizon, 6)

            for i in range(0, action_seq.shape[1], self.n_action_steps):
                chunk = action_seq[:, i: i + self.n_action_steps]
                if chunk.shape[1] == self.n_action_steps:
                    self._action_queue.append(chunk)

            if not self._action_queue:
                self._action_queue.append(action_seq)

        action_chunk = self._action_queue.pop(0)   # (B, n_action_steps, 6)
        return {"action": action_chunk.float()}


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint loader
# ─────────────────────────────────────────────────────────────────────────────

def load_mdf_model(checkpoint_path: str, config_dir: str) -> DiffusionForcingMultiModal:
    """
    Load a DiffusionForcingMultiModal model from a lightning checkpoint.

    checkpoint_path : path to the .ckpt file
    config_dir      : directory containing config.yaml (one level above checkpoints/)
    """
    from force_tool.utils.normalizer import LinearNormalizer

    cfg_path = pathlib.Path(config_dir) / "config.yaml"
    cfg = OmegaConf.load(cfg_path)

    algo_name = cfg.algorithm.get("_name", "tacmdf_q")

    from experiments.exp_planning import PlanningExperiment
    AlgoCls = PlanningExperiment.compatible_algorithms.get(algo_name, DiffusionForcingMultiModal)

    normalizer = LinearNormalizer()
    model = AlgoCls(cfg.algorithm, normalizer=normalizer)

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["state_dict"]

    # Lightning wraps the algo under self.algo, so keys are prefixed "algo."
    new_sd = {}
    for k, v in state_dict.items():
        new_k = k.removeprefix("algo.")
        new_sd[new_k] = v

    missing, unexpected = model.load_state_dict(new_sd, strict=False)
    if missing:
        print(f"Missing keys ({len(missing)}): {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"Unexpected keys ({len(unexpected)}): {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

@click.command()
@click.option("-c", "--checkpoint", required=True, help="Path to MDF .ckpt file")
@click.option("--config-dir", default=None, help="Dir containing config.yaml (default: parent of checkpoints/)")
@click.option("-o", "--output-dir", required=True)
@click.option("--isaacgym-cfg", default="isaacgym_config_power_plug.yaml",
              help="IsaacGym config name (relative to manifeel/config/)")
@click.option("--n-envs", default=50, show_default=True)
@click.option("--n-test-vis", default=2, show_default=True)
@click.option("--max-steps", default=200, show_default=True)
@click.option("--n-obs-steps", default=2, show_default=True)
@click.option("--n-action-steps", default=1, show_default=True)
@click.option("--his-len", default=4, show_default=True)
@click.option("--sampling-steps", default=10, show_default=True)
@click.option("--taxim-frame-stride", default=0, show_default=True,
              help="0 = single-frame taxim (ViT); 5 = two-frame (Sparsh)")
@click.option("--inference-mode", default="policy", show_default=True)
@click.option("--precision", default="bf16", type=click.Choice(["bf16", "fp16", "fp32"]))
@click.option("--gpu", default=0, show_default=True)
@click.option("--seed", default=100000, show_default=True)
def main(
    checkpoint, config_dir, output_dir, isaacgym_cfg,
    n_envs, n_test_vis, max_steps, n_obs_steps, n_action_steps,
    his_len, sampling_steps, taxim_frame_stride, inference_mode, precision, gpu, seed,
):
    os.environ["GPU"] = str(gpu)
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    if config_dir is None:
        ckpt = pathlib.Path(checkpoint)
        if "checkpoints" in ckpt.parts:
            idx = ckpt.parts.index("checkpoints")
            config_dir = str(pathlib.Path(*ckpt.parts[:idx]))
        else:
            config_dir = str(ckpt.parent)

    print(f"Loading MDF model from: {checkpoint}")
    print(f"Config dir: {config_dir}")

    ft_root = os.environ.get("FORCE_TOOL_ROOT", "")
    if ft_root and ft_root not in sys.path:
        sys.path.insert(0, ft_root)

    model = load_mdf_model(checkpoint, config_dir)
    device = torch.device(f"cuda:{gpu}")
    model.to(device)

    policy = MDFManifeel(
        model=model,
        his_len=his_len,
        n_action_steps=n_action_steps,
        inference_mode=inference_mode,
        sampling_timesteps=sampling_steps,
        taxim_frame_stride=taxim_frame_stride,
        precision=precision,
    )

    hydra_cfg_dir = str(pathlib.Path(__file__).parent / "manifeel" / "config")
    hydra.initialize(config_path=hydra_cfg_dir, version_base=None)

    shape_meta = {
        "obs": {
            "wrist": {"shape": [3, 256, 256], "type": "rgb"},
            "right_tactile_camera_taxim": {"shape": [3, 256, 256], "type": "rgb"},
            "state": {"shape": [7], "type": "low_dim"},
        },
        "action": {"shape": [6]},
    }

    from manifeel.env_runner.vistac_pih_runner_unit import ManifeelRunner
    runner = ManifeelRunner(
        output_dir=output_dir,
        shape_meta=shape_meta,
        isaacgym_cfg_name=isaacgym_cfg,
        n_test=n_envs,
        n_test_vis=n_test_vis,
        test_start_seed=seed,
        max_steps=max_steps,
        n_obs_steps=n_obs_steps,
        n_action_steps=n_action_steps,
        fps=10,
        past_action=False,
        tactile_size=[224, 224],
    )

    print("Starting evaluation...")
    log_data = runner.run(policy)

    results = {k: (v._path if hasattr(v, "_path") else v) for k, v in log_data.items()}
    out_path = pathlib.Path(output_dir) / "eval_log.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {out_path}")
    print(log_data)


if __name__ == "__main__":
    main()
