from gym import spaces

import isaacgym
# import isaacgymenvs
# from isaacgymenvs.utils.utils import set_seed
from isaacgymenvs.tasks.tacsl.tacsl_task_insertion import TacSLTaskInsertion
from isaacgymenvs.tasks.tacsl.tacsl_task_power import TacSLTaskPowerInsertion
from isaacgymenvs.tasks.tacsl.tacsl_task_gear import TacSLTaskGear
from isaacgymenvs.tasks.tacsl.tacsl_task_pick_in_box import TacSLTaskPickInBox
from isaacgymenvs.tasks.tacsl.tacsl_task_pick_in_box import TacSLTaskSearchkInBox
from isaacgymenvs.tasks.tacsl.tacsl_task_class_ball import TacSLTaskClassBall

from isaacgymenvs.utils.reformat import omegaconf_to_dict

import torch
# import cv2
import numpy as np
import random
import os
import hydra
from omegaconf import DictConfig, OmegaConf

"""
TODO: Design the DONE signal for the step function
"""

class SingleIsaacEnvWrapper():

    metadata = {"render.modes": ["rgb_array"], "video.frames_per_second": 10}

    def __init__(self, env):
        # self.cfg = cfg
        self.envs = env
        self.render_cache = []
        self.task_obs_keys = ['left_tactile_camera_taxim', 'state']
        # self.task_obs_keys = ['wrist', 'left_tactile_camera_taxim', 'state']

    # def __init__(self, cfg):
    #     self.cfg = cfg
    #     self.render_cache = []
    #     self._start_task(self.cfg)

    @property
    def observation_space(self):
        obs_space = self.envs.observation_space
        obs_space = self._update_observation_space(obs_space)
        return obs_space
    
    @property
    def action_space(self):
        return self.envs.action_space

    @property
    def num_acts(self) -> int:
        """Get the number of actions in the environment."""
        return self.envs.num_actions

    @property
    def num_obs(self) -> int:
        """Get the number of observations in the environment."""
        return self.envs.num_observations

    @property
    def num_envs(self) -> int:
        """Get the number of environments."""
        return self.envs.num_environments

    # def _start_task(self, cfg):
    #     self.envs = isaacgymenvs.make(
    #                         cfg.seed, 
    #                         cfg.task_name, 
    #                         cfg.task.env.numEnvs, 
    #                         cfg.sim_device,
    #                         cfg.rl_device,
    #                         cfg.graphics_device_id,
    #                         cfg.headless,
    #                         cfg.multi_gpu,
    #                         cfg.capture_video,
    #                         cfg.force_render,
    #                         cfg,
    #     )

    def _start_task(self, cfg):
        rl_device = cfg.rl_device
        sim_device = cfg.sim_device
        graphics_device_id = cfg.graphics_device_id
        headless = cfg.headless
        virtual_screen_capture = cfg.capture_video
        force_render = cfg.force_render

        cfg_task = cfg['task']
        cfg_dict = omegaconf_to_dict(cfg_task)

        self.envs = TacSLTaskInsertion(cfg=cfg_dict,
                                       rl_device=rl_device,
                                       sim_device=sim_device,
                                       graphics_device_id=graphics_device_id,
                                       headless=headless,
                                       virtual_screen_capture=virtual_screen_capture,
                                       force_render=force_render,
                                       )

    def seed(self, seed=None):
        if seed is None:
            seed = np.random.randint(0,25536)
        # global rank of the GPU
        global_rank = int(os.getenv("RANK", "0"))
        # sets seed. if seed is -1 will pick a random one
        seed = set_seed(seed, torch_deterministic=False, rank=global_rank)
        # seed = set_seed(seed)
        self._seed = seed
        
    def step(self, actions):
        actions = torch.from_numpy(actions).to(dtype=torch.float32).unsqueeze(0)
        obs, reward, reset, info = self.envs.step(actions)
        
        obs = self._transform_obs_data(obs['obs'])
        self.render_cache = obs
        obs = self._apply_obs_by_keys(obs)

        reward = reward[0].cpu().numpy()
        reset = reset[0].cpu().numpy()
        info = {key: value.cpu().numpy() for key, value in info.items() if isinstance(value, torch.Tensor)}

        return obs, reward, reset, info

    def reset(self):
        self.envs.reset_idx(torch.arange(self.num_envs))
        self.envs.compute_observations()
        obs = self.envs.reset()
        obs = self._transform_obs_data(obs['obs'])
        self.render_cache = obs
        obs = self._apply_obs_by_keys(obs)
        return obs
    
    def render(self, mode="rgb_array"):
        obs_view = 'front'
        if self.render_cache is None:
            raise RuntimeError('Must run reset or step before render.')
        if len(self.render_cache[obs_view].shape) > 3:
            img = self.render_cache[obs_view][0]
        else:
            img = self.render_cache[obs_view]
        img = np.moveaxis(img, 0, -1)
        img = (img * 255).astype(np.uint8)

        return img
        # if self.envs.virtual_display and mode == "rgb_array":
        #     img = self.envs.virtual_display.grab()
        #     return np.array(img)

    def _transform_obs_data(self, obs):
        tf_obs = dict()
        for key, value in obs.items():
            # Squeeze singleton dimensions
            tf_obs[key] = value.cpu().numpy().squeeze()

            # Check if the last dimension has size 3 and move it to the front
            # check if it is the image data
            if len(tf_obs[key].shape) >= 3 and tf_obs[key].shape[-1] == 3:
                tf_obs[key] = np.moveaxis(tf_obs[key], -1, 0)

        assert all(key in tf_obs for key in ('ee_pos', 'ee_quat')), \
                    "Keys 'ee_pos' and 'ee_quat' are missing in tf_obs"
        
        ee_pos = tf_obs['ee_pos']
        ee_quat = tf_obs['ee_quat']
        state = np.concatenate([ee_pos, ee_quat])
        tf_obs['state'] = state

        return tf_obs

    def _apply_obs_by_keys(self, obs):
        task_obs = {key: obs[key] for key in self.task_obs_keys if key in obs}
        return task_obs

    def _update_observation_space(self, observation_space):
        updated_observation_space = {}
        for key, space in observation_space.spaces.items():
            if key in self.task_obs_keys: 
                if isinstance(space, spaces.Box):
                    # Check if the shape ends with (W, H, 3) and modify to (3, W, H)
                    if len(space.shape) == 3 and space.shape[-1] == 3:
                        updated_observation_space[key] = spaces.Box(
                            low=np.float32(-np.inf), high=np.float32(np.inf), 
                            shape=(3, space.shape[0], space.shape[1]), 
                            dtype=space.dtype
                        )
                    else:
                        # Keep the original space for non-image data
                        updated_observation_space[key] = space
                else:
                    # Keep the original space for non-Box types
                    updated_observation_space[key] = space

        if 'state' in self.task_obs_keys:
            updated_observation_space['state'] = spaces.Box(
                            low=np.float32(-np.inf), high=np.float32(np.inf), 
                            shape=(7,), 
                            dtype=np.float32
                        )

        # Return the updated Dict observation space
        return spaces.Dict(updated_observation_space)

    def close(self):
        pass
        # return self.envs.close()

def get_isaacgym_env(cfg):
    rl_device = cfg.rl_device
    sim_device = cfg.sim_device
    graphics_device_id = cfg.graphics_device_id
    headless = cfg.headless
    virtual_screen_capture = cfg.capture_video
    force_render = cfg.force_render

    cfg_task = cfg['task']
    cfg_dict = omegaconf_to_dict(cfg_task)

    env = TacSLTaskInsertion(cfg=cfg_dict,
                            rl_device=rl_device,
                            sim_device=sim_device,
                            graphics_device_id=graphics_device_id,
                            headless=headless,
                            virtual_screen_capture=virtual_screen_capture,
                            force_render=force_render,
                                    )
    return env

def set_seed(seed, torch_deterministic=False, rank=0):
    """ set seed across modules """
    if seed == -1 and torch_deterministic:
        seed = 42 + rank
    elif seed == -1:
        seed = np.random.randint(0, 10000)
    else:
        seed = seed + rank

    print("Setting seed: {}".format(seed))

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    return seed

if __name__ == '__main__':
    @hydra.main(version_base="1.1", 
                config_path="../config", 
                config_name="isaacgym_config")
    def main(cfg: DictConfig):
        env = get_isaacgym_env(cfg)
        # Pass the config explicitly
        wrapped_env = SingleIsaacEnvWrapper(env)
        print("Observation space is", wrapped_env.observation_space)
        print("Action space is", wrapped_env.action_space)
        wrapped_env.seed(0)
        _ = wrapped_env.reset()
        for _ in range(5):
            random_actions = 2.0 * np.random.rand(wrapped_env.action_space.shape[0]) - 1.0
            obs, reward, reset, _ = wrapped_env.step(random_actions)
            # print(reward, reset)
            # tactile_rgb_image = observations['left_tactile_camera_taxim'][0]
            # wrist_rgb_image = observations['wrist_2'][0]

    main()