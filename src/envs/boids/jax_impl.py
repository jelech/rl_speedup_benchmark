import jax
import jax.numpy as jnp
from jax import jit, vmap, random
import numpy as np
from src.common import BaseEnv


class JaxBoidsEnv(BaseEnv):
    def __init__(self, num_boids=100, width=800, height=600):
        self.num_boids = num_boids
        self.width = width
        self.height = height
        # Force key and initial state to be on CPU to avoid Metal allocator issues
        # JAX Metal is experimental and fails on default allocations
        try:
            cpu_device = jax.devices("cpu")[0]
        except:
            cpu_device = None

        if cpu_device:
            with jax.default_device(cpu_device):
                self.key = random.PRNGKey(42)
                self.state = jnp.zeros((num_boids, 4))
        else:
            # Fallback if no CPU device found (unlikely)
            self.key = random.PRNGKey(42)
            self.state = jnp.zeros((num_boids, 4))

        # Params as a tuple for JIT
        self.params = (
            5.0,  # max_speed
            0.1,  # max_force
            50.0,  # perception_radius
            20.0,  # crowding_radius
            float(width),
            float(height),
        )

        # Warmup
        print("Warming up JAX Boids...")
        self.reset()
        self.step()
        print("JAX Boids Warmup complete.")

    def reset(self) -> np.ndarray:
        self.key, k1, k2 = random.split(self.key, 3)
        pos = random.uniform(k1, (self.num_boids, 2)) * jnp.array([self.width, self.height])
        vel = (random.uniform(k2, (self.num_boids, 2)) - 0.5) * 5.0
        self.state = jnp.concatenate([pos, vel], axis=1)
        return np.array(self.state)

    @staticmethod
    @jit
    def _step_jit(state, params):
        max_speed, max_force, perception_radius, crowding_radius, width, height = params
        pos = state[:, :2]
        vel = state[:, 2:]

        # Compute pairwise differences
        # shape: (N, N, 2)
        # To optimize memory, we can use vmap over single boid logic,
        # but full matrix is fine for N=100-1000.

        delta = pos[:, None, :] - pos[None, :, :]
        dist_sq = jnp.sum(delta**2, axis=2)
        dist = jnp.sqrt(dist_sq)

        # Avoid self (diagonal is 0)
        # We can just add eye * inf
        dist = dist + jnp.eye(dist.shape[0]) * 1e5

        neighbors_mask = dist < perception_radius
        crowding_mask = dist < crowding_radius

        # 1. Separation
        safe_dist = jnp.maximum(dist, 1e-5)[:, :, None]
        sep_vecs = delta / safe_dist
        sep_force = jnp.sum(sep_vecs * crowding_mask[:, :, None], axis=1)

        # 2. Alignment
        count = jnp.sum(neighbors_mask, axis=1)[:, None]
        safe_count = jnp.maximum(count, 1.0)

        avg_vel = jnp.sum(vel[None, :, :] * neighbors_mask[:, :, None], axis=1)
        align_force = (avg_vel / safe_count) - vel

        # 3. Cohesion
        avg_pos = jnp.sum(pos[None, :, :] * neighbors_mask[:, :, None], axis=1)
        center_force = (avg_pos / safe_count) - pos

        # Combine
        total_acc = (sep_force * 1.5) + (align_force * 1.0) + (center_force * 1.0)

        # Limit force
        force_norm = jnp.linalg.norm(total_acc, axis=1, keepdims=True)
        scale = jnp.where(force_norm > max_force, max_force / force_norm, 1.0)
        total_acc *= scale

        # Update
        vel += total_acc

        # Limit speed
        speed = jnp.linalg.norm(vel, axis=1, keepdims=True)
        scale = jnp.where(speed > max_speed, max_speed / speed, 1.0)
        vel *= scale

        pos += vel

        # Wrap
        pos = jnp.stack([pos[:, 0] % width, pos[:, 1] % height], axis=1)

        return jnp.concatenate([pos, vel], axis=1)

    def step(self, action=None):
        self.state = self._step_jit(self.state, self.params)
        return np.array(self.state), 0.0, False, False, {}
