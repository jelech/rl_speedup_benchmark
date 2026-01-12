import argparse
import sys
import os
import numpy as np
from src.common import BenchmarkTimer
from src.agents.simple import RandomAgent, SimplePolicyAgent

def get_env_class(env_name, backend):
    if env_name == 'cartpole':
        import src.envs.cartpole as pkg
        if backend == 'python': return pkg.PythonCartPoleEnv
        if backend == 'c': return pkg.CCartPoleEnv
        if backend == 'go': return pkg.GoCartPoleEnv
        if backend == 'cython': return pkg.CythonCartPoleEnv
        if backend == 'jax': return pkg.JaxCartPoleEnv
    elif env_name == 'boids':
        import src.envs.boids as pkg
        if backend == 'python': return pkg.BoidsEnv
        if backend == 'cython': return pkg.CythonBoidsEnv
        if backend == 'jax': return pkg.JaxBoidsEnv
        if backend in ['c', 'go']: return None # Not implemented
        
    raise ValueError(f"Unknown env/backend combo: {env_name}/{backend}")

def run_session(env_name, backend, episodes=1000, max_steps=500, num_boids=100):
    print(f"--- Starting Benchmark: {env_name} [{backend}] ---")
    
    EnvClass = get_env_class(env_name, backend)
    if EnvClass is None:
        print(f"Skipping {backend} for {env_name} (Not Implemented)")
        return None
        
    # Instantiate
    if env_name == 'boids':
        env = EnvClass(num_boids=num_boids)
    else:
        env = EnvClass()

    # Agent (Simple dummy agent)
    agent = SimplePolicyAgent()
    
    timer = BenchmarkTimer()
    timer.start_total()

    for ep in range(episodes):
        with timer.time_block('reset'):
            state = env.reset()
        
        terminated = False
        truncated = False
        steps = 0
        
        while not (terminated or truncated) and steps < max_steps:
            with timer.time_block('agent'):
                # For boids, action might be ignored or dummy
                action = 0 
                # agent.act(state) # Skip actual agent logic for boids to focus on env physics?
                # Actually let's keep it minimal
                pass
            
            with timer.time_block('step'):
                next_state, reward, terminated, truncated, _ = env.step(action)
            
            # Skip learning for pure env benchmark
            # with timer.time_block('agent'):
            #    agent.learn(...)
            
            state = next_state
            steps += 1
            timer.stats.total_steps += 1

    timer.stop_total()
    timer.stats.episodes = episodes
    
    return timer.stats

def print_stats(stats, name):
    if stats is None: return
    print(f"\nResult for [{name}]:")
    print(f"  Total Time: {stats.total_time:.4f}s")
    print(f"  Total Steps: {stats.total_steps}")
    print(f"  Env Step Time: {stats.env_step_time:.4f}s")
    print(f"  Env Reset Time: {stats.env_reset_time:.4f}s")
    print(f"  Env Only FPS: {stats.env_fps:.2f}")
    print("-" * 30)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='cartpole', choices=['cartpole', 'boids'])
    parser.add_argument('--backend', type=str, default='all', 
                        choices=['python', 'c', 'cython', 'go', 'jax', 'all'])
    parser.add_argument('--episodes', type=int, default=1000)
    parser.add_argument('--steps', type=int, default=500)
    parser.add_argument('--num_boids', type=int, default=100) # Only for boids
    args = parser.parse_args()

    if args.backend == 'all':
        if args.env == 'cartpole':
            targets = ['python', 'c', 'cython', 'go', 'jax']
        else:
            targets = ['python', 'cython', 'jax']
    else:
        targets = [args.backend]
    
    results = {}
    
    for target in targets:
        try:
            stats = run_session(args.env, target, 
                              episodes=args.episodes, 
                              max_steps=args.steps,
                              num_boids=args.num_boids)
            print_stats(stats, target)
            results[target] = stats
        except Exception as e:
            print(f"Error running {target}: {e}")
            import traceback
            traceback.print_exc()
