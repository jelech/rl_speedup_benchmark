try:
    from .cy_impl import CythonBoidsEnv as _CythonBoidsEnv
except ImportError:
    # Fallback/Dev path
    try:
        from src.envs.boids.cy_impl import CythonBoidsEnv as _CythonBoidsEnv
    except ImportError:
        pass

class CythonBoidsWrapper:
    def __init__(self, num_boids=100):
        self.num_boids = num_boids # <--- Added this property
        self.env = _CythonBoidsEnv(num_boids=num_boids)
        
    def reset(self):
        return self.env.reset()
        
    def step(self, action=None):
        return self.env.step(action)
