"""
Merton 跳跃扩散模型 (Jump Diffusion Model)

dS = μ·S·dt + σ·S·dW + S·(J-1)·dN

其中:
  N: 泊松过程，强度 λ
  J: 跳跃幅度，J ~ LogNormal(μ_J, σ_J²)
"""

import numpy as np
from typing import Optional


class MertonSimulator:
    """Merton 跳跃扩散模型模拟器"""

    def __init__(
        self,
        S0: float,
        mu: float,
        sigma: float,
        dt: float,
        lam: float = 1.0,
        mu_J: float = 0.0,
        sigma_J: float = 0.1,
        seed: Optional[int] = None,
    ):
        """
        参数:
            S0: 初始价格
            mu: 漂移率
            sigma: 扩散波动率
            dt: 时间步长
            lam: 跳跃强度 λ（年均跳跃次数）
            mu_J: 跳跃幅度的对数均值
            sigma_J: 跳跃幅度的对数标准差
            seed: 随机种子
        """
        self.S0 = S0
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.lam = lam
        self.mu_J = mu_J
        self.sigma_J = sigma_J
        self.rng = np.random.RandomState(seed)

    def simulate(self, n_steps: int, n_paths: int = 1) -> np.ndarray:
        """
        模拟含跳跃的价格路径

        返回:
            shape=(n_paths, n_steps+1) 的价格矩阵
        """
        dt = self.dt
        prices = np.zeros((n_paths, n_steps + 1))
        prices[:, 0] = self.S0

        for t in range(n_steps):
            # 布朗部分
            Z = self.rng.standard_normal(n_paths)
            brownian = (self.mu - 0.5 * self.sigma ** 2) * dt + self.sigma * np.sqrt(dt) * Z

            # 跳跃部分
            # 每步发生跳跃的概率 = λ·dt (泊松过程在小时间步的近似)
            jump_prob = self.lam * dt
            has_jump = self.rng.random(n_paths) < jump_prob

            # 跳跃幅度: J ~ LogNormal(μ_J, σ_J²)
            jump_size = np.zeros(n_paths)
            n_jumps = has_jump.sum()
            if n_jumps > 0:
                jump_log = self.rng.normal(
                    self.mu_J, self.sigma_J, size=n_jumps
                )
                jump_size[has_jump] = np.exp(jump_log) - 1.0  # (J-1)

            # 组合: S(t+1) = S(t) · exp(brownian) · (1 + jump)
            prices[:, t + 1] = prices[:, t] * np.exp(brownian) * (1.0 + jump_size)

        return prices

    def simulate_with_times(
        self, n_steps: int, n_paths: int = 1
    ) -> tuple[np.ndarray, np.ndarray]:
        """返回 (时间数组, 价格矩阵)"""
        times = np.arange(n_steps + 1) * self.dt
        prices = self.simulate(n_steps, n_paths)
        return times, prices

    # ============ 单步生成（实时模拟用） ============

    def step(self, S_current: float) -> tuple[float, bool]:
        """
        单步推进，返回 (新价格, 是否有跳跃)

        参数:
            S_current: 当前价格

        返回:
            (S_new, has_jump)
        """
        dt = self.dt
        Z = self.rng.standard_normal()
        brownian = (self.mu - 0.5 * self.sigma ** 2) * dt + self.sigma * np.sqrt(dt) * Z

        jump_prob = self.lam * dt
        has_jump = self.rng.random() < jump_prob

        jump_size = 0.0
        if has_jump:
            jump_log = self.rng.normal(self.mu_J, self.sigma_J)
            jump_size = np.exp(jump_log) - 1.0

        S_new = S_current * np.exp(brownian) * (1.0 + jump_size)
        return float(S_new), has_jump

    def reset(self, seed: Optional[int] = None):
        """重置随机数生成器"""
        if seed is not None:
            self.rng = np.random.RandomState(seed)
        else:
            self.rng = np.random.RandomState()
