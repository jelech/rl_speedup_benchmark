import ctypes
import os
import numpy as np
from src.common import BaseEnv

# Load the shared library
LIB_PATH = os.path.join(os.path.dirname(__file__), 'go_src/cartpole.so')

class GoCartPoleEnv(BaseEnv):
    def __init__(self):
        if not os.path.exists(LIB_PATH):
            raise RuntimeError(f"Go Library not found at {LIB_PATH}. Please run 'make'.")
            
        self.lib = ctypes.CDLL(LIB_PATH)
        
        # NewCartPole -> returns pointer (which is actually an ID disguised as pointer)
        self.lib.NewCartPole.restype = ctypes.c_void_p
        
        # Reset
        self.lib.Reset.argtypes = [ctypes.c_void_p]
        
        # GetState
        self.lib.GetState.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double)]
        
        # Step
        # func Step(ptr *C.double, action int, outState *C.double, outReward *C.double, outDone *C.int)
        self.lib.Step.argtypes = [
            ctypes.c_void_p,
            ctypes.c_longlong, # Go int is usually 64bit on 64bit sys, c_longlong is safer? Or c_int?
                               # On standard 64-bit architecture, Go int is 64-bit.
                               # Python ctypes c_int is usually 32-bit.
                               # We should use c_longlong to match Go int.
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_int) 
        ]
        
        self.obj = self.lib.NewCartPole()
        
        # Buffers
        self.out_state = (ctypes.c_double * 4)()
        self.out_reward = ctypes.c_double()
        self.out_done = ctypes.c_int()
        
    def reset(self) -> np.ndarray:
        self.lib.Reset(self.obj)
        self.lib.GetState(self.obj, self.out_state)
        return np.frombuffer(self.out_state, dtype=np.float64).astype(np.float32)

    def step(self, action: int):
        # Go expects int (64 bit), passing python int works
        self.lib.Step(
            self.obj, 
            int(action), # Force cast to python int
            self.out_state, 
            ctypes.byref(self.out_reward), 
            ctypes.byref(self.out_done)
        )
        
        s = np.frombuffer(self.out_state, dtype=np.float64).astype(np.float32)
        r = float(self.out_reward.value)
        d = bool(self.out_done.value)
        
        return s, r, d, False, {}
