# distutils: language=c
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False

import numpy as np
cimport numpy as np
from libc.math cimport sin, cos, M_PI
from libc.stdlib cimport rand, RAND_MAX

cdef class CythonCartPoleEnv:
    # Define typed attributes for speed
    cdef double gravity
    cdef double masscart
    cdef double masspole
    cdef double total_mass
    cdef double length
    cdef double polemass_length
    cdef double force_mag
    cdef double tau
    cdef double theta_threshold_radians
    cdef double x_threshold
    
    # State
    cdef double x
    cdef double x_dot
    cdef double theta
    cdef double theta_dot
    cdef int steps_beyond_terminated

    def __init__(self):
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = (self.masspole + self.masscart)
        self.length = 0.5
        self.polemass_length = (self.masspole * self.length)
        self.force_mag = 10.0
        self.tau = 0.02
        self.theta_threshold_radians = 12 * 2 * M_PI / 360
        self.x_threshold = 2.4
        
        self.steps_beyond_terminated = -1

    cdef double _random_double(self, double min_val, double max_val):
        cdef double scale = rand() / <double>RAND_MAX
        return min_val + scale * (max_val - min_val)

    def reset(self):
        self.x = self._random_double(-0.05, 0.05)
        self.x_dot = self._random_double(-0.05, 0.05)
        self.theta = self._random_double(-0.05, 0.05)
        self.theta_dot = self._random_double(-0.05, 0.05)
        self.steps_beyond_terminated = -1
        
        return np.array([self.x, self.x_dot, self.theta, self.theta_dot], dtype=np.float32)

    def step(self, int action):
        cdef double force
        cdef double costheta
        cdef double sintheta
        cdef double temp
        cdef double thetaacc
        cdef double xacc
        cdef double reward = 0.0
        cdef bint terminated = False
        
        force = self.force_mag if action == 1 else -self.force_mag
        
        costheta = cos(self.theta)
        sintheta = sin(self.theta)

        temp = (force + self.polemass_length * self.theta_dot**2 * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (self.length * (4.0 / 3.0 - self.masspole * costheta**2 / self.total_mass))
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        self.x += self.tau * self.x_dot
        self.x_dot += self.tau * xacc
        self.theta += self.tau * self.theta_dot
        self.theta_dot += self.tau * thetaacc

        if (self.x < -self.x_threshold or self.x > self.x_threshold or
            self.theta < -self.theta_threshold_radians or self.theta > self.theta_threshold_radians):
            terminated = True

        if not terminated:
            reward = 1.0
        elif self.steps_beyond_terminated == -1:
            self.steps_beyond_terminated = 0
            reward = 1.0
        else:
            self.steps_beyond_terminated += 1
            reward = 0.0
            
        return np.array([self.x, self.x_dot, self.theta, self.theta_dot], dtype=np.float32), reward, terminated, False, {}
