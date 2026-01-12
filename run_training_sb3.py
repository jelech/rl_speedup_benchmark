import time
import argparse
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

from src.gym_adapter import GymAdapter
import src.envs.cartpole as cartpole_pkg
import src.envs.boids as boids_pkg

def get_env_class(env_name, backend):
    if env_name == 'cartpole':
        if backend == 'python': return cartpole_pkg.PythonCartPoleEnv
        if backend == 'cython': return cartpole_pkg.CythonCartPoleEnv
        # C/Go skipped for simplicity in training benchmark unless wrapped carefully
    elif env_name == 'boids':
        if backend == 'python': return boids_pkg.BoidsEnv
        if backend == 'cython': return boids_pkg.CythonBoidsEnv
    return None

def run_sb3_benchmark(env_name, backend, total_timesteps=100_000, num_envs=1, num_boids=100):
    print(f"--- SB3 Training Benchmark: {env_name} [{backend}] ({num_envs} envs) ---")
    
    EnvClass = get_env_class(env_name, backend)
    if EnvClass is None:
        print("Backend not supported for SB3 benchmark.")
        return

    # Create VecEnv
    # We use a lambda to delay instantiation
    env_kwargs = {'num_boids': num_boids} if env_name == 'boids' else {}
    
    def make_env():
        return GymAdapter(EnvClass, **env_kwargs)

    # Use Subproc for parallelism if num_envs > 1 (fair comparison with JAX batch)
    if num_envs > 1:
        vec_env = make_vec_env(make_env, n_envs=num_envs, vec_env_cls=SubprocVecEnv)
    else:
        vec_env = DummyVecEnv([make_env])

    model = PPO("MlpPolicy", vec_env, verbose=0, device="cpu") # Force CPU for baseline fairness? Or GPU?
    # Usually users compare:
    # 1. SB3 (CPU Env + GPU Train) vs JAX (GPU Env + GPU Train)
    # Let's try auto device (likely MPS on Mac)
    model = PPO("MlpPolicy", vec_env, verbose=0, device="auto") 

    start_time = time.perf_counter()
    model.learn(total_timesteps=total_timesteps)
    duration = time.perf_counter() - start_time
    
    sps = total_timesteps / duration
    print(f"Training finished in {duration:.2f}s")
    print(f"Training SPS: {sps:.2f}")
    
    return sps

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='cartpole')
    parser.add_argument('--backend', type=str, default='python')
    parser.add_argument('--steps', type=int, default=10000)
    parser.add_argument('--num_envs', type=int, default=1)
    parser.add_argument('--num_boids', type=int, default=100)
    
    args = parser.parse_args()
    
    run_sb3_benchmark(args.env, args.backend, args.steps, args.num_envs, args.num_boids)
