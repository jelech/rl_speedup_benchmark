import ctypes
import os
import numpy as np
from src.common import BaseEnv

# Load the shared library
# We assume it's compiled to cartpole.so in the same directory or lib path
# For the runner, we will likely compile it to src/envs/c_src/cartpole.so
LIB_PATH = os.path.join(os.path.dirname(__file__), 'c_src/libcartpole.so')

class CCartPoleEnv(BaseEnv):
    def __init__(self):
        if not os.path.exists(LIB_PATH):
            raise RuntimeError(f"C Library not found at {LIB_PATH}. Please run 'make'.")
            
        self.lib = ctypes.CDLL(LIB_PATH)
        
        # Define Struct placeholder (we handle it as void* mostly or specific struct)
        # But here we interact via wrapper functions that take the pointer
        
        # CartPole_new
        self.lib.CartPole_new.restype = ctypes.c_void_p
        
        # CartPole_free
        self.lib.CartPole_free.argtypes = [ctypes.c_void_p]
        
        # CartPole_reset
        self.lib.CartPole_reset.argtypes = [ctypes.c_void_p]
        
        # CartPole_step
        # void CartPole_step(CartPoleState* s, int action, double* out_state, double* out_reward, bool* out_terminated)
        self.lib.CartPole_step.argtypes = [
            ctypes.c_void_p, 
            ctypes.c_int, 
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_bool)
        ]
        
        self.obj = self.lib.CartPole_new()
        
        # Pre-allocate buffers for ctypes to write into to avoid re-allocation overhead
        self.out_state = (ctypes.c_double * 4)()
        self.out_reward = ctypes.c_double()
        self.out_terminated = ctypes.c_bool()
        
    def reset(self) -> np.ndarray:
        self.lib.CartPole_reset(self.obj)
        # We need to manually get the state out or just replicate the structure access
        # For Reset, we usually want the state returned. 
        # But wait, my C reset doesn't return state, it just resets internal. 
        # I should probably just peek at the struct or modify reset to return it.
        # Let's just use a trick: Reset then call a getter? 
        # Actually, simpler: My C implementation initializes values. 
        # Let's just read them. 
        # Hack for speed: access struct fields directly? No, opaque pointer is safer.
        # Let's fix C code or just accept I need to query it?
        # Actually, let's just cheat and assume the caller calls step() immediately? 
        # No, reset must return state.
        # Let's just read the memory directly since we know the struct layout (4 doubles).
        
        state_ptr = ctypes.cast(self.obj, ctypes.POINTER(ctypes.c_double))
        # state is first 4 doubles
        return np.array([state_ptr[0], state_ptr[1], state_ptr[2], state_ptr[3]], dtype=np.float32)

    def step(self, action: int):
        self.lib.CartPole_step(
            self.obj, 
            int(action), # Force cast to python int for ctypes
            self.out_state, 
            ctypes.byref(self.out_reward), 
            ctypes.byref(self.out_terminated)
        )
        
        # Convert to numpy
        # buffer interface is faster than list conversion
        s = np.frombuffer(self.out_state, dtype=np.float64).astype(np.float32)
        r = float(self.out_reward.value)
        d = bool(self.out_terminated.value)
        
        return s, r, d, False, {}

    def __del__(self):
        if hasattr(self, 'lib') and hasattr(self, 'obj'):
            self.lib.CartPole_free(self.obj)
