# distutils: language=c
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False

import numpy as np
cimport numpy as np
from libc.math cimport sqrt

# We need a proper random generator in C or just call back to python for reset (slow reset is fine)
# For step, we want pure C speed.

cdef class CythonBoidsEnv:
    cdef int num_boids
    cdef double width
    cdef double height
    cdef double max_speed
    cdef double max_force
    cdef double perception_radius
    cdef double crowding_radius
    
    # State storage: Contiguous memory for speed
    cdef double[:, :] state_view
    cdef np.ndarray state_arr
    
    # Pre-allocated buffers for calculations to avoid malloc in loop
    cdef double[:, :] acc_view
    
    def __init__(self, int num_boids=100, double width=800.0, double height=600.0):
        self.num_boids = num_boids
        self.width = width
        self.height = height
        self.max_speed = 5.0
        self.max_force = 0.1
        self.perception_radius = 50.0
        self.crowding_radius = 20.0
        
        self.state_arr = np.zeros((num_boids, 4), dtype=np.float64)
        self.state_view = self.state_arr
        
        self.acc_view = np.zeros((num_boids, 2), dtype=np.float64)

    def reset(self):
        # Python random is fine for reset
        pos = np.random.rand(self.num_boids, 2) * np.array([self.width, self.height])
        vel = (np.random.rand(self.num_boids, 2) - 0.5) * self.max_speed
        self.state_arr = np.hstack([pos, vel]).astype(np.float64)
        self.state_view = self.state_arr
        return np.array(self.state_arr, dtype=np.float32)

    def step(self, action=None):
        cdef int i, j
        cdef double dx, dy, dist_sq, dist
        cdef double sx, sy # separation sum
        cdef double ax, ay # alignment sum
        cdef double cx, cy # cohesion sum
        cdef double count
        cdef double norm, scale
        
        # O(N^2) loop, but optimized in C
        for i in range(self.num_boids):
            sx = 0; sy = 0
            ax = 0; ay = 0
            cx = 0; cy = 0
            count = 0
            
            for j in range(self.num_boids):
                if i == j:
                    continue
                
                dx = self.state_view[i, 0] - self.state_view[j, 0]
                dy = self.state_view[i, 1] - self.state_view[j, 1]
                dist_sq = dx*dx + dy*dy
                
                if dist_sq < self.perception_radius * self.perception_radius:
                    dist = sqrt(dist_sq)
                    count += 1
                    
                    # Cohesion
                    cx += self.state_view[j, 0]
                    cy += self.state_view[j, 1]
                    
                    # Alignment
                    ax += self.state_view[j, 2]
                    ay += self.state_view[j, 3]
                    
                    # Separation
                    if dist < self.crowding_radius and dist > 1e-5:
                        sx += dx / dist
                        sy += dy / dist
            
            if count > 0:
                # Alignment steering
                ax /= count
                ay /= count
                ax -= self.state_view[i, 2]
                ay -= self.state_view[i, 3]
                
                # Cohesion steering
                cx /= count
                cy /= count
                cx -= self.state_view[i, 0]
                cy -= self.state_view[i, 1]
            
            # Combine
            self.acc_view[i, 0] = (sx * 1.5) + (ax * 1.0) + (cx * 1.0)
            self.acc_view[i, 1] = (sy * 1.5) + (ay * 1.0) + (cy * 1.0)
            
            # Limit Force
            norm = sqrt(self.acc_view[i, 0]**2 + self.acc_view[i, 1]**2)
            if norm > self.max_force:
                scale = self.max_force / norm
                self.acc_view[i, 0] *= scale
                self.acc_view[i, 1] *= scale
                
        # Update
        for i in range(self.num_boids):
            self.state_view[i, 2] += self.acc_view[i, 0]
            self.state_view[i, 3] += self.acc_view[i, 1]
            
            # Limit Speed
            norm = sqrt(self.state_view[i, 2]**2 + self.state_view[i, 3]**2)
            if norm > self.max_speed:
                scale = self.max_speed / norm
                self.state_view[i, 2] *= scale
                self.state_view[i, 3] *= scale
                
            self.state_view[i, 0] += self.state_view[i, 2]
            self.state_view[i, 1] += self.state_view[i, 3]
            
            # Wrap
            while self.state_view[i, 0] < 0: self.state_view[i, 0] += self.width
            while self.state_view[i, 0] >= self.width: self.state_view[i, 0] -= self.width
            while self.state_view[i, 1] < 0: self.state_view[i, 1] += self.height
            while self.state_view[i, 1] >= self.height: self.state_view[i, 1] -= self.height
            
        return np.array(self.state_arr, dtype=np.float32), 0.0, False, False, {}
