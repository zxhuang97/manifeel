"""
ManiFeel eval entry point for MMDF socket policy.

Runs inside the Apptainer container (manifeel conda env).
Connects to the MMDF policy server and evaluates using ManifeelRunner.

Usage:
    python manifeel/policy/run_mmdf_eval.py \
        --task-cfg vistac_wrist \
        --isaacgym-cfg isaacgym_config_power_plug.yaml \
        --output-dir data/manifeel_eval/... \
        --ckpt-name epoch_10 \
        --port 5556 \
        --n-test 50 \
        --n-test-vis 2 \
        --max-steps 500 \
        --seed 100000
"""

import argparse
import json
import os

import isaacgym  # must be imported before torch
import hydra
from omegaconf import OmegaConf

from manifeel.env_runner.vistac_pih_runner_unit import ManifeelRunner
from manifeel.policy.mmdf_socket_policy import MMDFSocketPolicy
from task_info import get_by_isaacgym_cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-cfg", default="vistac_wrist")
    parser.add_argument("--isaacgym-cfg", default="isaacgym_config_power_plug.yaml")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--ckpt-name", required=True)
    parser.add_argument("--server-host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5556)
    parser.add_argument("--connect-timeout", type=float, default=180.0)
    parser.add_argument("--n-test", type=int, default=50)
    parser.add_argument("--n-test-vis", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Episode step budget; defaults to task_info value if not set")
    parser.add_argument("--n-obs-steps", type=int, default=1)
    parser.add_argument("--n-action-steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=100000)
    parser.add_argument("--tactile-size", type=int, nargs=2, default=[224, 224])
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    config_dir = os.path.realpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "config")
    )
    # Load shape_meta directly from the task yaml (no interpolations needed)
    task_cfg_path = os.path.join(config_dir, "task", f"{args.task_cfg}.yaml")
    task_cfg = OmegaConf.load(task_cfg_path)
    shape_meta = OmegaConf.to_container(task_cfg.shape_meta, resolve=True) if "shape_meta" in task_cfg else {}
    action_dim = get_by_isaacgym_cfg(args.isaacgym_cfg)["action_dim"]
    max_steps  = args.max_steps if args.max_steps is not None else get_by_isaacgym_cfg(args.isaacgym_cfg)["max_steps"]
    if "action" in shape_meta:
        shape_meta["action"]["shape"] = [action_dim]
    print(f"[eval] task={args.task_cfg}  action_dim={action_dim}  max_steps={max_steps}")

    # ManifeelRunner needs GlobalHydra initialized to load the isaacgym config
    hydra.initialize_config_dir(config_dir=config_dir, version_base=None)

    policy = MMDFSocketPolicy(
        server_host=args.server_host,
        server_port=args.port,
        action_dim=action_dim,
        connect_timeout=args.connect_timeout,
        n_obs_steps=args.n_obs_steps,
        n_action_steps=args.n_action_steps,
    )
    runner = ManifeelRunner(
        output_dir=args.output_dir,
        shape_meta=shape_meta,
        isaacgym_cfg_name=args.isaacgym_cfg,
        n_test=args.n_test,
        n_test_vis=args.n_test_vis,
        test_start_seed=args.seed,
        max_steps=max_steps,
        n_obs_steps=args.n_obs_steps,
        n_action_steps=args.n_action_steps,
        fps=10,
        tactile_size=args.tactile_size,
    )

    wandb_mode = os.environ.get("WANDB_MODE", "disabled")
    import wandb
    wandb.init(mode=wandb_mode)
    log_data = runner.run(policy)

    mean_score = log_data.get("test/mean_score", 0.0)
    results = {
        "ckpt_name": args.ckpt_name,
        "summary": {
            "success_rate_mean": mean_score,
        },
        "log_data": {k: v for k, v in log_data.items() if not hasattr(v, "_path")},
    }
    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"mean_score = {mean_score:.3f}")


if __name__ == "__main__":
    main()
