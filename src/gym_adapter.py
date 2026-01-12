import gymnasium as gym
import numpy as np
from gymnasium import spaces

class GymAdapter(gym.Env):
    """
    Adapts our BaseEnv to Gymnasium interface for Stable-Baselines3.
    """
    def __init__(self, env_class, **kwargs):
        super().__init__()
        self.internal_env = env_class(**kwargs)
        
        # Infer spaces from a reset/step cycle
        obs = self.internal_env.reset()
        if hasattr(self.internal_env, 'num_boids'):
            # Boids: (N, 4) flattened or kept as matrix?
            # SB3 MLPs usually expect flat vectors.
            self.flat = True
            n = self.internal_env.num_boids * 4
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(n,), dtype=np.float32
            )
            # Dummy action space for boids (they just simulate)
            self.action_space = spaces.Discrete(1)
        else:
            # CartPole
            self.flat = False
            self.observation_space = spaces.Box(
                low=np.array([-4.8, -np.inf, -0.42, -np.inf]), 
                high=np.array([4.8, np.inf, 0.42, np.inf]), 
                dtype=np.float32
            )
            self.action_space = spaces.Discrete(2)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        obs = self.internal_env.reset()
        if self.flat: obs = obs.flatten()
        return obs, {}

    def step(self, action):
        # CartPole expects int action
        if hasattr(action, 'item'): action = action.item()
        
        obs, reward, terminated, truncated, info = self.internal_env.step(action)
        if self.flat: obs = obs.flatten()
        
        # SB3 expects terminated/truncated
        return obs, float(reward), bool(terminated), bool(truncated), info
