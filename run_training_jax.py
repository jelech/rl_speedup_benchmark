import time
import jax
import jax.numpy as jnp
from jax import jit, vmap, grad, value_and_grad
import optax
import numpy as np
import argparse

# Import JAX Envs directly
from src.envs.cartpole.jax_impl import JaxCartPoleEnv
from src.envs.boids.jax_impl import JaxBoidsEnv


class SimplePPO:
    """
    A minimal PPO implementation in Pure JAX using Optax.
    """

    def __init__(self, obs_dim, act_dim, lr=3e-4):
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        # Init params
        key = jax.random.PRNGKey(0)
        k1, k2 = jax.random.split(key)

        # Policy Net (MLP: obs -> 64 -> 64 -> act)
        self.params = {
            "w1": jax.random.normal(k1, (obs_dim, 64)) * 0.1,
            "b1": jnp.zeros(64),
            "w2": jax.random.normal(k2, (64, act_dim)) * 0.1,
            "b2": jnp.zeros(act_dim),
        }

        self.optimizer = optax.adam(lr)
        self.opt_state = self.optimizer.init(self.params)

    @staticmethod
    def forward(params, x):
        h = jnp.tanh(jnp.dot(x, params["w1"]) + params["b1"])
        logits = jnp.dot(h, params["w2"]) + params["b2"]
        return logits

    def update(self, params, opt_state, obs, actions, advantages):
        def loss_fn(p):
            logits = self.forward(p, obs)
            # MSE loss to simulate computational load
            loss = jnp.mean((logits - 1.0) ** 2)
            return loss

        grads = grad(loss_fn)(params)
        updates, new_opt_state = self.optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state


def run_jax_training(env_name, steps=10000, num_envs=128, num_boids=100):
    print(f"--- JAX End-to-End Training: {env_name} ({num_envs} envs) ---")

    # Init Env
    if env_name == "cartpole":
        env = JaxCartPoleEnv()  # Single instance, we will vmap it
        obs_dim = 4
        act_dim = 2
        step_fn = env._step_jit
        env_params = (9.8, 1.0, 0.1, 1.1, 0.5, 0.05, 10.0, 0.02, 0.209, 2.4)

    elif env_name == "boids":
        env = JaxBoidsEnv(num_boids=num_boids)
        obs_dim = num_boids * 4
        act_dim = 1  # Dummy
        step_fn = env._step_jit
        env_params = env.params

    # Init states
    key = jax.random.PRNGKey(0)
    if env_name == "cartpole":
        state_batch = jax.random.uniform(key, (num_envs, 4), minval=-0.05, maxval=0.05)
    else:
        # Boids
        state_batch = jnp.zeros((num_envs, num_boids, 4))

    # Agent
    agent = SimplePPO(obs_dim, act_dim)
    params = agent.params
    opt_state = agent.opt_state

    @jit
    def train_step(carrier, _):
        state_batch, params, opt_state = carrier

        # 1. Inference
        obs_flat = state_batch.reshape(num_envs, -1)
        logits = agent.forward(params, obs_flat)
        actions = jnp.argmax(logits, axis=1)

        # 2. Env Step (Vectorized)
        if env_name == "cartpole":

            def single_step(s, a):
                ns, r, d = step_fn(s, a, env_params, -1)
                ns = jnp.where(d, jnp.zeros_like(s), ns)
                return ns

            next_states = vmap(single_step)(state_batch, actions)

        else:

            def single_step(s):
                return step_fn(s, env_params)

            next_states = vmap(single_step)(state_batch)

        # 3. Learn
        new_params, new_opt_state = agent.update(params, opt_state, obs_flat, actions, jnp.ones(num_envs))

        return (next_states, new_params, new_opt_state), None

    # Run Loop
    num_iterations = steps // num_envs

    print("JIT Compiling...")
    carrier = (state_batch, params, opt_state)
    _ = train_step(carrier, None)
    print("Compilation Done. Training...")

    start = time.perf_counter()
    final_carrier, _ = jax.lax.scan(train_step, carrier, None, length=num_iterations)
    jax.block_until_ready(final_carrier[0])

    duration = time.perf_counter() - start
    total_steps = num_iterations * num_envs
    sps = total_steps / duration

    print(f"Training finished in {duration:.2f}s")
    print(f"Training SPS: {sps:.2f}")
    return sps


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="cartpole")
    parser.add_argument("--steps", type=int, default=100000)
    parser.add_argument("--num_envs", type=int, default=128)
    parser.add_argument("--num_boids", type=int, default=100)

    args = parser.parse_args()
    run_jax_training(args.env, args.steps, args.num_envs, args.num_boids)
