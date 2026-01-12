# RL Environment Performance Benchmark

这是一个用于对比不同编程语言（Python, Cython, C, Go, JAX）在 Reinforcement Learning 环境模拟及训练中性能差异的基准测试项目。

包含从 **"Hello World" (CartPole)** 到 **复杂多智能体 (Boids)** 的场景，以及从 **纯环境模拟** 到 **端到端 PPO 训练** 的全栈测试。

## 项目结构

- `src/envs/`: 环境实现 (CartPole, Boids)。
- `run_benchmark.py`: **纯环境模拟**基准测试（测试物理引擎 FPS）。
- `run_training_sb3.py`: **混合架构训练**基准测试 (CPU Env + GPU PyTorch)。
- `run_training_jax.py`: **端到端训练**基准测试 (GPU JAX Env + GPU JAX Policy)。

## 环境要求

### Linux (NVIDIA GPU) - 推荐
- Python 3.8+
- CUDA 12+
- GCC / Clang

### macOS (Apple Silicon)
- Python 3.10+
- `jax-metal` (可选，用于 GPU 加速)

## 安装

1. **安装依赖**
   ```bash
   pip install -r requirements.txt
   ```
   *注意：Linux 下 `jax[cuda12]` 会自动安装，macOS 下若需 GPU 加速请手动安装 `jax-metal`。*

2. **编译扩展 (C/Cython/Go)**
   ```bash
   make all
   ```

## 运行基准测试

### 1. 纯环境模拟 (Physics FPS)
测试物理引擎的极限吞吐量，不包含神经网络。

```bash
# 测试 CartPole (所有后端)
python run_benchmark.py --env cartpole --backend all --episodes 1000

# 测试 Boids (N=100)
python run_benchmark.py --env boids --backend cython --num_boids 100

# 测试 Boids 大规模 (N=2000, 推荐在 GPU 上跑 JAX)
python run_benchmark.py --env boids --backend jax --num_boids 2000
```

### 2. PPO 训练 (Training SPS)
测试包含神经网络前向/反向传播的完整训练吞吐量。

**场景 A: 传统架构 (SB3 + Cython)**
CPU 跑环境，GPU 跑网络。
```bash
python run_training_sb3.py --env boids --backend cython --num_envs 8 --num_boids 100
```

**场景 B: 端到端架构 (Pure JAX)**
GPU 跑环境，GPU 跑网络（零数据传输）。
```bash
python run_training_jax.py --env boids --num_envs 256 --num_boids 100
```

## 贡献
欢迎提交 PR 增加更多环境（如 MuJoCo 绑定）或更多语言后端（如 Rust）。
