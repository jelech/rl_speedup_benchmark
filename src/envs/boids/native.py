import numpy as np
from src.common import BaseEnv

class BoidsEnv(BaseEnv):
    def __init__(self, num_boids=100, width=800, height=600):
        self.num_boids = num_boids
        self.width = width
        self.height = height
        
        # Boids parameters
        self.max_speed = 5.0
        self.max_force = 0.1
        self.perception_radius = 50.0
        self.crowding_radius = 20.0
        
        # Initial state
        self.state = None # (N, 4) -> x, y, vx, vy

    def reset(self) -> np.ndarray:
        # Random position and velocity
        pos = np.random.rand(self.num_boids, 2) * np.array([self.width, self.height])
        vel = (np.random.rand(self.num_boids, 2) - 0.5) * self.max_speed
        self.state = np.hstack([pos, vel]).astype(np.float32)
        return self.state

    def step(self, action=None):
        # Action is ignored in this benchmark (simulation only), 
        # or can be a global perturbation. We focus on simulation cost.
        
        # Unpack
        pos = self.state[:, :2]
        vel = self.state[:, 2:]
        
        # Compute distances (Broadcasting, Heavy Op)
        # Shape: (N, 1, 2) - (1, N, 2) -> (N, N, 2)
        delta = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dist_sq = np.sum(delta**2, axis=2)
        dist = np.sqrt(dist_sq)
        
        # Rules accumulation
        acc = np.zeros_like(vel)
        
        # We use simple loops for Python baseline to represent "Naive Python"
        # Or vectorized NumPy for "Optimized Python". 
        # Benchmarks usually compare "Best Practice Python (NumPy)" vs Others.
        # Let's use NumPy vectorization, because naive python loops are too slow to be a fair baseline.
        
        # Mask for neighbors (dist < radius and dist > 0)
        # Fill diagonal with inf to avoid self-influence
        np.fill_diagonal(dist, np.inf)
        
        neighbors_mask = (dist < self.perception_radius)
        crowding_mask = (dist < self.crowding_radius)
        
        # 1. Separation
        # Vector away from neighbors: -delta
        # We want sum of (pos - neighbor_pos) / dist
        # delta is (pos_i - pos_j). If crowding, we want to move away from j.
        # So we want sum(delta_ij / dist_ij) where crowding_mask is True
        
        # Avoid div by zero
        safe_dist = np.where(dist < 1e-5, 1e-5, dist)
        sep_vecs = delta / safe_dist[:, :, np.newaxis] # Normalized direction
        
        # Filter by crowding
        sep_force = np.sum(sep_vecs * crowding_mask[:, :, np.newaxis], axis=1)
        
        # 2. Alignment
        # Avg velocity of neighbors
        avg_vel = np.sum(vel[np.newaxis, :, :] * neighbors_mask[:, :, np.newaxis], axis=1)
        count = np.sum(neighbors_mask, axis=1)[:, np.newaxis]
        safe_count = np.where(count == 0, 1, count)
        
        align_force = avg_vel / safe_count
        # Steer towards that velocity
        align_force = align_force - vel
        
        # 3. Cohesion
        # Avg position of neighbors
        avg_pos = np.sum(pos[np.newaxis, :, :] * neighbors_mask[:, :, np.newaxis], axis=1)
        center_force = (avg_pos / safe_count) - pos
        
        # Combine forces (weights)
        total_acc = (sep_force * 1.5) + (align_force * 1.0) + (center_force * 1.0)
        
        # Limit Force
        # Norm
        force_norm = np.linalg.norm(total_acc, axis=1, keepdims=True)
        scale = np.where(force_norm > self.max_force, self.max_force / force_norm, 1.0)
        total_acc *= scale
        
        # Update
        vel += total_acc
        
        # Limit Speed
        speed = np.linalg.norm(vel, axis=1, keepdims=True)
        scale = np.where(speed > self.max_speed, self.max_speed / speed, 1.0)
        vel *= scale
        
        pos += vel
        
        # Wrap around
        pos[:, 0] = pos[:, 0] % self.width
        pos[:, 1] = pos[:, 1] % self.height
        
        self.state = np.hstack([pos, vel]).astype(np.float32)
        
        return self.state, 0.0, False, False, {}
