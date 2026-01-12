import jax
import jax.numpy as jnp
import numpy as np
from src.common import BaseEnv
from jax import jit, random

class JaxCartPoleEnv(BaseEnv):
    def __init__(self):
        # JAX needs a PRNG key
        self.key = random.PRNGKey(0)
        
        # Constants
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = (self.masspole + self.masscart)
        self.length = 0.5
        self.polemass_length = (self.masspole * self.length)
        self.force_mag = 10.0
        self.tau = 0.02
        self.theta_threshold_radians = 12 * 2 * jnp.pi / 360
        self.x_threshold = 2.4
        
        self.state = jnp.zeros(4)
        self.steps_beyond_terminated = -1
        
        # Warmup JIT
        print("Warming up JAX JIT...")
        self.reset()
        self.step(0)
        self.reset()
        print("JAX Warmup complete.")

    def reset(self):
        self.key, subkey = random.split(self.key)
        self.state = random.uniform(subkey, shape=(4,), minval=-0.05, maxval=0.05)
        self.steps_beyond_terminated = -1
        return np.array(self.state)

    @staticmethod
    @jit
    def _step_jit(state, action, params, steps_beyond_terminated):
        # Unpack params
        gravity, masscart, masspole, total_mass, length, polemass_length, force_mag, tau, theta_threshold, x_threshold = params
        
        x, x_dot, theta, theta_dot = state
        
        # JAX doesn't like Python control flow based on values, but here action is scalar
        # We can use jnp.where or lax.cond. 
        # Since action is passed as int, we can use simple arithmetic or jnp.where
        force = jnp.where(action == 1, force_mag, -force_mag)
        
        costheta = jnp.cos(theta)
        sintheta = jnp.sin(theta)

        temp = (force + polemass_length * theta_dot**2 * sintheta) / total_mass
        thetaacc = (gravity * sintheta - costheta * temp) / (length * (4.0 / 3.0 - masspole * costheta**2 / total_mass))
        xacc = temp - polemass_length * thetaacc * costheta / total_mass

        x = x + tau * x_dot
        x_dot = x_dot + tau * xacc
        theta = theta + tau * theta_dot
        theta_dot = theta_dot + tau * thetaacc

        state = jnp.array([x, x_dot, theta, theta_dot])

        terminated = (
            (x < -x_threshold) | (x > x_threshold) |
            (theta < -theta_threshold) | (theta > theta_threshold)
        )
        
        # Reward logic
        # If not terminated: 1.0
        # If terminated and steps_beyond == -1: 1.0, set steps_beyond = 0
        # If terminated and steps_beyond >= 0: 0.0, inc steps_beyond
        
        # This stateful logic 'steps_beyond_terminated' is tricky in pure functional JAX without passing it through
        # We simplify for benchmark: standard CartPole behavior usually terminates immediately.
        # So we just return 1.0 if not terminated.
        # If the runner keeps calling step() after done=True, we return 0.0.
        
        reward = jnp.where(terminated, 0.0, 1.0)
        # Fix: The original gym logic gives 1.0 on the step it fails.
        # My python logic: if not terminated: 1.0. If just failed: 1.0. If failed before: 0.0.
        # We need to pass in 'steps_beyond_terminated' to know.
        
        # Actually for speed benchmark, let's simplify slightly:
        # Just return 1.0 if not done, 0.0 if done. 
        # (This is standard behavior for many envs, Gym's steps_beyond is a warning mechanism mostly)
        
        return state, reward, terminated

    def step(self, action):
        # We need to pass params as a tuple/array to be JIT-friendly or closure
        params = (
            self.gravity, self.masscart, self.masspole, self.total_mass, 
            self.length, self.polemass_length, self.force_mag, self.tau, 
            self.theta_threshold_radians, self.x_threshold
        )
        
        # JAX arrays are immutable, so we get new state back
        # We must cast action to jnp array if not already, usually scalars are fine
        
        new_state, reward, terminated = self._step_jit(
            self.state, action, params, self.steps_beyond_terminated
        )
        
        self.state = new_state
        
        # To avoid side effects in JIT, we update the steps_beyond logic outside or passed through
        # Here we do it outside for simplicity as it's trivial logic
        if terminated:
             if self.steps_beyond_terminated == -1:
                 reward = 1.0 # First fail
                 self.steps_beyond_terminated = 0
             else:
                 reward = 0.0
                 self.steps_beyond_terminated += 1
        
        # Convert to numpy for the interface
        return np.array(self.state), float(reward), bool(terminated), False, {}
