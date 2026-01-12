import time
from dataclasses import dataclass, field
from typing import Dict, List
import numpy as np


@dataclass
class TimingStats:
    total_time: float = 0.0
    env_step_time: float = 0.0
    env_reset_time: float = 0.0
    agent_time: float = 0.0
    total_steps: int = 0
    episodes: int = 0

    @property
    def fps(self) -> float:
        if self.total_time == 0:
            return 0.0
        return self.total_steps / self.total_time

    @property
    def env_fps(self) -> float:
        env_total = self.env_step_time + self.env_reset_time
        if env_total == 0:
            return 0.0
        return self.total_steps / env_total


class BenchmarkTimer:
    def __init__(self):
        self.stats = TimingStats()
        self._start_time = 0.0

    def start_total(self):
        self.stats.total_time = 0
        self._global_start = time.perf_counter()

    def stop_total(self):
        self.stats.total_time = time.perf_counter() - self._global_start

    def time_block(self, block_type: str):
        """Context manager for timing specific blocks"""

        class TimerContext:
            def __init__(self, timer, b_type):
                self.timer = timer
                self.b_type = b_type
                self.start = 0

            def __enter__(self):
                self.start = time.perf_counter()

            def __exit__(self, exc_type, exc_val, exc_tb):
                elapsed = time.perf_counter() - self.start
                if self.b_type == "step":
                    self.timer.stats.env_step_time += elapsed
                elif self.b_type == "reset":
                    self.timer.stats.env_reset_time += elapsed
                elif self.b_type == "agent":
                    self.timer.stats.agent_time += elapsed

        return TimerContext(self, block_type)


class BaseEnv:
    """Standard Interface for all implementations"""

    def reset(self) -> np.ndarray:
        raise NotImplementedError

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        # returns state, reward, terminated, truncated, info
        raise NotImplementedError
