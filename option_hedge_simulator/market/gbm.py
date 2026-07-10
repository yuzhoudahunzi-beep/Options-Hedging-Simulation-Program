"""
几何布朗运动 (Geometric Brownian Motion) 模拟器

dS = μ·S·dt + σ·S·dW
S(t+dt) = S(t) · exp((μ - σ²/2)·dt + σ·√dt·Z)
"""

import numpy as np
from typing import Optional


class GBMSimulator:
    """几何布朗运动市场模拟器"""

    def __init__(
        self,
        S0: float,
        mu: float,
        sigma: float,
        dt: float,
        seed: Optional[int] = None,
    ):
        """
        参数:
            S0: 初始价格
            mu: 漂移率（年化期望收益率）
            sigma: 波动率（年化）
            dt: 时间步长（年化，如 1/252 表示日步长）
            seed: 随机种子
        """
        self.S0 = S0
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.rng = np.random.RandomState(seed)

    def simulate(self, n_steps: int, n_paths: int = 1) -> np.ndarray:
        """
        模拟价格路径

        参数:
            n_steps: 时间步数
            n_paths: 路径数量

        返回:
            shape=(n_paths, n_steps+1) 的价格矩阵（含初始价格）
        """
        dt = self.dt
        drift = self.mu - 0.5 * self.sigma ** 2
        diffusion = self.sigma * np.sqrt(dt)

        # 生成随机增量
        Z = self.rng.standard_normal((n_paths, n_steps))

        # 对数增量
        log_returns = drift * dt + diffusion * Z

        # 构建价格路径
        log_prices = np.zeros((n_paths, n_steps + 1))
        log_prices[:, 0] = np.log(self.S0)
        log_prices[:, 1:] = np.log(self.S0) + np.cumsum(log_returns, axis=1)

        return np.exp(log_prices)

    def simulate_with_times(
        self, n_steps: int, n_paths: int = 1
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        返回 (时间数组, 价格矩阵)
        """
        times = np.arange(n_steps + 1) * self.dt
        prices = self.simulate(n_steps, n_paths)
        return times, prices

    # ============ 单步生成（实时模拟用） ============

    def step(self, S_current: float) -> float:
        """
        单步推进，返回新价格

        参数:
            S_current: 当前价格

        返回:
            下一步价格
        """
        dt = self.dt
        drift = self.mu - 0.5 * self.sigma ** 2
        diffusion = self.sigma * np.sqrt(dt)
        Z = self.rng.standard_normal()
        log_return = drift * dt + diffusion * Z
        return S_current * np.exp(log_return)

    def reset(self, seed: Optional[int] = None):
        """重置随机数生成器"""
        if seed is not None:
            self.rng = np.random.RandomState(seed)
        else:
            self.rng = np.random.RandomState()
