if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import hydra
import torch
from omegaconf import OmegaConf
import pathlib
from torch.utils.data import DataLoader
import copy
import random
import wandb
import tqdm
import numpy as np
import shutil
import dill
import threading
import re
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from manifeel.policy.diffusion_transformer_image_policy import DiffusionTransformerImagePolicy
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy.model.diffusion.ema_model import EMAModel
from diffusion_policy.model.common.lr_scheduler import get_scheduler

OmegaConf.register_new_resolver("eval", eval, replace=True)

class TrainDiffusionTransformerHybridWorkspace(BaseWorkspace):
    include_keys = ['global_step', 'epoch']

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        self.model: DiffusionTransformerImagePolicy = hydra.utils.instantiate(cfg.policy)

        _print_model_summary(self.model)

        self.ema_model: DiffusionTransformerImagePolicy = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)

        # Transformer policy has its own get_optimizer with separate weight decay
        self.optimizer = self.model.get_optimizer(**cfg.optimizer)

        self.global_step = 0
        self.epoch = 0

    def run(self):
        cfg = copy.deepcopy(self.cfg)

        self.resume_training(cfg)

        dataset: BaseImageDataset
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        assert isinstance(dataset, BaseImageDataset)
        train_dataloader = DataLoader(dataset, **cfg.dataloader)
        normalizer = dataset.get_normalizer()

        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)

        self.model.set_normalizer(normalizer)
        if cfg.training.use_ema:
            self.ema_model.set_normalizer(normalizer)

        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(
                len(train_dataloader) * cfg.training.num_epochs) \
                    // cfg.training.gradient_accumulate_every,
            last_epoch=self.global_step-1
        )

        ema: EMAModel = None
        if cfg.training.use_ema:
            ema = hydra.utils.instantiate(
                cfg.ema,
                model=self.ema_model)

        env_runner: BaseImageRunner
        env_runner = hydra.utils.instantiate(
            cfg.task.env_runner,
            output_dir=self.output_dir)
        assert isinstance(env_runner, BaseImageRunner)

        wandb_run = wandb.init(
            dir=str(self.output_dir),
            config=OmegaConf.to_container(cfg, resolve=True),
            **cfg.logging
        )
        wandb.config.update(
            {
                "output_dir": self.output_dir,
            }
        )

        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, 'checkpoints'),
            **cfg.checkpoint.topk
        )

        device = torch.device(cfg.training.device)
        self.model.to(device)
        if self.ema_model is not None:
            self.ema_model.to(device)
        optimizer_to(self.optimizer, device)

        train_sampling_batch = None

        if cfg.training.debug:
            cfg.training.num_epochs = 2
            cfg.training.max_train_steps = 3
            cfg.training.max_val_steps = 3
            cfg.training.rollout_every = 1
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1

        log_path = os.path.join(self.output_dir, 'logs.json.txt')
        with JsonLogger(log_path) as json_logger:
            for local_epoch_idx in range(cfg.training.num_epochs):
                step_log = dict()

                if cfg.training.freeze_encoder:
                    self.model.obs_encoder.eval()
                    self.model.obs_encoder.requires_grad_(False)

                train_losses = list()
                with tqdm.tqdm(train_dataloader, desc=f"Training epoch {self.epoch}", 
                        leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                    for batch_idx, batch in enumerate(tepoch):
                        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                        if train_sampling_batch is None:
                            train_sampling_batch = batch

                        raw_loss = self.model.compute_loss(batch)
                        loss = raw_loss / cfg.training.gradient_accumulate_every
                        loss.backward()

                        if self.global_step % cfg.training.gradient_accumulate_every == 0:
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                            lr_scheduler.step()
                        
                        if cfg.training.use_ema:
                            ema.step(self.model)

                        raw_loss_cpu = raw_loss.item()
                        tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
                        train_losses.append(raw_loss_cpu)
                        step_log = {
                            'train_loss': raw_loss_cpu,
                            'global_step': self.global_step,
                            'epoch': self.epoch,
                            'lr': lr_scheduler.get_last_lr()[0]
                        }

                        is_last_batch = (batch_idx == (len(train_dataloader)-1))
                        if not is_last_batch:
                            wandb_run.log(step_log, step=self.global_step)
                            json_logger.log(step_log)
                            self.global_step += 1

                        if (cfg.training.max_train_steps is not None) \
                            and batch_idx >= (cfg.training.max_train_steps-1):
                            break

                train_loss = np.mean(train_losses)
                step_log['train_loss'] = train_loss

                policy = self.model
                if cfg.training.use_ema:
                    policy = self.ema_model
                policy.eval()

                if (self.epoch % cfg.training.rollout_every) == 0:
                    runner_log = env_runner.run(policy)
                    step_log.update(runner_log)

                if (self.epoch % cfg.training.val_every) == 0:
                    with torch.no_grad():
                        val_losses = list()
                        with tqdm.tqdm(val_dataloader, desc=f"Validation epoch {self.epoch}", 
                                leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                            for batch_idx, batch in enumerate(tepoch):
                                batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                                loss = self.model.compute_loss(batch)
                                val_losses.append(loss)
                                if (cfg.training.max_val_steps is not None) \
                                    and batch_idx >= (cfg.training.max_val_steps-1):
                                    break
                        if len(val_losses) > 0:
                            val_loss = torch.mean(torch.tensor(val_losses)).item()
                            step_log['val_loss'] = val_loss

                if (self.epoch % cfg.training.sample_every) == 0:
                    with torch.no_grad():
                        batch = dict_apply(train_sampling_batch, lambda x: x.to(device, non_blocking=True))
                        obs_dict = batch['obs']
                        gt_action = batch['action']
                        
                        result = policy.predict_action(obs_dict)
                        pred_action = result['action_pred']
                        mse = torch.nn.functional.mse_loss(pred_action, gt_action)
                        step_log['train_action_mse_error'] = mse.item()
                        del batch
                        del obs_dict
                        del gt_action
                        del result
                        del pred_action
                        del mse
                
                if (self.epoch % cfg.training.checkpoint_every) == 0:
                    if cfg.checkpoint.save_last_ckpt:
                        self.save_checkpoint(epoch=self.epoch)
                    if cfg.checkpoint.save_last_snapshot:
                        self.save_snapshot()

                policy.train()

                wandb_run.log(step_log, step=self.global_step)
                json_logger.log(step_log)
                self.global_step += 1
                self.epoch += 1

    def save_checkpoint(self, path=None, tag='latest', epoch=0,
                        exclude_keys=None,
                        include_keys=None,
                        use_thread=True):
        if path is None:
            path = pathlib.Path(self.output_dir).joinpath('checkpoints', f'{tag}_epoch{epoch}.ckpt')
        else:
            path = pathlib.Path(path)
            path = path.with_name(f'{path.stem}_epoch{epoch}.ckpt')
        if exclude_keys is None:
            exclude_keys = tuple(self.exclude_keys)
        if include_keys is None:
            include_keys = tuple(self.include_keys) + ('_output_dir',)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            'cfg': self.cfg,
            'state_dicts': dict(),
            'pickles': dict()
        }
        for key, value in self.__dict__.items():
            if hasattr(value, 'state_dict') and hasattr(value, 'load_state_dict'):
                if key not in exclude_keys:
                    if use_thread:
                        payload['state_dicts'][key] = _copy_to_cpu(value.state_dict())
                    else:
                        payload['state_dicts'][key] = value.state_dict()
            elif key in include_keys:
                payload['pickles'][key] = dill.dumps(value)
        if use_thread:
            self._saving_thread = threading.Thread(
                target=lambda : torch.save(payload, path.open('wb'), pickle_module=dill))
            self._saving_thread.start()
        else:
            torch.save(payload, path.open('wb'), pickle_module=dill)
        _prune_checkpoints(path.parent)
        return str(path.absolute())
    
    def resume_training(self, cfg):
        if cfg.training.resume:
            print("Resuming training...")
            latest_ckpt_path = self.get_latest_checkpoint_path()
            if latest_ckpt_path is not None:
                print(f"Attempting to resume from checkpoint {latest_ckpt_path}")
                while latest_ckpt_path:
                    try:
                        self.load_checkpoint(path=latest_ckpt_path)
                        print(f"Successfully resumed from checkpoint {latest_ckpt_path}")
                        break
                    except Exception as e:
                        print(f"Failed to load checkpoint {latest_ckpt_path}: {e}")
                        latest_ckpt_path = self.get_previous_checkpoint_path(latest_ckpt_path)
                else:
                    print("No valid checkpoints found. Starting from scratch.")
            else:
                print("No checkpoints found. Starting from scratch.")
    
    def get_latest_checkpoint_path(self):
        checkpoint_dir = pathlib.Path(self.output_dir).joinpath('checkpoints')
        checkpoint_files = list(checkpoint_dir.glob('*.ckpt'))
        epoch_files = []
        for file in checkpoint_files:
            match = re.search(r'_epoch(\d+)\.ckpt$', file.name)
            if match:
                epoch = int(match.group(1))
                epoch_files.append((epoch, file))
        if epoch_files:
            epoch_files.sort(key=lambda x: x[0], reverse=True)
            return epoch_files[0][1]
        return None
    
    def get_previous_checkpoint_path(self, current_path):
        checkpoint_dir = pathlib.Path(self.output_dir).joinpath('checkpoints')
        checkpoint_files = list(checkpoint_dir.glob('*.ckpt'))
        epoch_files = []
        for file in checkpoint_files:
            match = re.search(r'_epoch(\d+)\.ckpt$', file.name)
            if match:
                epoch = int(match.group(1))
                epoch_files.append((epoch, file))
        epoch_files.sort(key=lambda x: x[0], reverse=True)
        current_match = re.search(r'_epoch(\d+)\.ckpt$', current_path.name)
        if current_match:
            current_epoch = int(current_match.group(1))
            for epoch, file in epoch_files:
                if epoch < current_epoch:
                    return file
        return None

def _print_model_summary(model):
    print("\n" + "=" * 60)
    print("MODEL ARCHITECTURE")
    print("=" * 60)
    print(model)
    print("=" * 60)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters:     {total:,}")
    print(f"Trainable parameters: {trainable:,}")
    if hasattr(model, 'obs_encoder'):
        enc_params = sum(p.numel() for p in model.obs_encoder.parameters())
        print(f"  obs_encoder:        {enc_params:,}")
    if hasattr(model, 'model'):
        backbone_params = sum(p.numel() for p in model.model.parameters())
        backbone_name = model.model.__class__.__name__
        print(f"  backbone ({backbone_name}): {backbone_params:,}")
    print("=" * 60 + "\n")

def _prune_checkpoints(checkpoint_dir):
    checkpoint_files = list(checkpoint_dir.glob('*.ckpt'))
    epoch_files = []
    for file in checkpoint_files:
        match = re.search(r'_epoch(\d+)\.ckpt$', str(file.name))
        if match:
            epoch = int(match.group(1))
            epoch_files.append((epoch, file))
    epoch_files.sort(key=lambda x: x[0], reverse=True)
    if len(epoch_files) > 3:
        print(f"Keeping checkpoints: {[f[0] for f in epoch_files[:3]]}")
        print(f"Removing checkpoints: {[f[0] for f in epoch_files[3:]]}")
        for _, file in epoch_files[3:]:
            try:
                file.unlink()
                print(f"Deleted: {file}")
            except Exception as e:
                print(f"Error deleting {file}: {e}")

def _copy_to_cpu(x):
    if isinstance(x, torch.Tensor):
        return x.detach().to('cpu')
    elif isinstance(x, dict):
        result = dict()
        for k, v in x.items():
            result[k] = _copy_to_cpu(v)
        return result
    elif isinstance(x, list):
        return [_copy_to_cpu(k) for k in x]
    else:
        return copy.deepcopy(x)

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")), 
    config_name=pathlib.Path(__file__).stem)
def main(cfg):
    workspace = TrainDiffusionTransformerHybridWorkspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()
