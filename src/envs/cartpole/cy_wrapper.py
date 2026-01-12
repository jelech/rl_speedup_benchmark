try:
    from .cy_impl import CythonCartPoleEnv as _CythonCartPoleEnv
except ImportError:
    # During dev, it might need compilation
    try:
        from src.envs.cartpole.cy_impl import CythonCartPoleEnv as _CythonCartPoleEnv
    except ImportError:
        raise ImportError("Cython module not built. Run 'make' or 'python setup.py build_ext --inplace'")

class CythonCartPoleWrapper:
    def __init__(self):
        self.env = _CythonCartPoleEnv()
        
    def reset(self):
        return self.env.reset()
        
    def step(self, action):
        return self.env.step(int(action))
