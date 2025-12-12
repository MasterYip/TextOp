import gymnasium as gym

from . import agents, diffusion_env_cfg

##
# Register Gym environments.
##

# G1 Diffusion Policy (Direct environment - for DiffuseCLOC evaluation)
gym.register(
    id="Isaac-TextOp-Diffusion-G1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": diffusion_env_cfg.G1DiffusionEnvCfg,
    },
)
