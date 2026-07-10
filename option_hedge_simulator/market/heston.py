"""
Heston 随机波动率模型

dS = μ·S·dt + √v·S·dW₁
dv = κ·(θ-v)·dt + ξ·√v·dW₂
corr(dW₁, dW₂) = ρ·dt

使用 Cholesky 分解生成相关的布朗运动
"""

import numpy as np
from typing import Optional


class HestonSimulator:
    """Heston 随机波动率模型模拟器"""

    def __init__(
        self,
        S0: float,
        mu: float,
        v0: float,
        kappa: float,
        theta: float,
        xi: float,
        rho: float,
        dt: float,
        seed: Optional[int] = None,
    ):
        """
        参数:
            S0: 初始价格
            mu: 漂移率
            v0: 初始方差 (v = σ²)
            kappa: 均值回归速度
            theta: 长期方差
            xi: 方差过程的波动率 (vol of vol)
            rho: 价格与波动率的相关性 (通常 < 0)
            dt: 时间步长
            seed: 随机种子
        """
        self.S0 = S0
        self.mu = mu
        self.v0 = v0
        self.kappa = kappa
        self.theta = theta
        self.xi = xi
        self.rho = rho
        self.dt = dt
        self.rng = np.random.RandomState(seed)

    def simulate(
        self, n_steps: int, n_paths: int = 1, full_state: bool = False
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """
        模拟价格路径（及可选的方差路径）

        参数:
            n_steps: 时间步数
            n_paths: 路径数量
            full_state: 是否同时返回方差路径

        返回:
            prices: shape=(n_paths, n_steps+1)
            variance: shape=(n_paths, n_steps+1)（仅当 full_state=True）
        """
        dt = self.dt
        rho = self.rho

        # Cholesky 分解: [Z1, Z2] 使得 corr = rho
        # Z1 = W1
        # Z2 = rho * W1 + sqrt(1-rho^2) * W2
        sqrt_1m_rho2 = np.sqrt(1.0 - rho ** 2)

        prices = np.zeros((n_paths, n_steps + 1))
        variances = np.zeros((n_paths, n_steps + 1))
        prices[:, 0] = self.S0
        variances[:, 0] = self.v0

        for t in range(n_steps):
            v = variances[:, t]
            S = prices[:, t]

            # 确保方差非负（使用 full truncation scheme）
            v_pos = np.maximum(v, 0.0)
            sqrt_v = np.sqrt(v_pos)

            # 生成独立的布朗增量
            W1 = self.rng.standard_normal(n_paths)
            W2 = self.rng.standard_normal(n_paths)

            # 相关的布朗增量
            Z1 = W1
            Z2 = rho * W1 + sqrt_1m_rho2 * W2

            # 价格更新 (Euler-Maruyama)
            prices[:, t + 1] = S + self.mu * S * dt + sqrt_v * S * np.sqrt(dt) * Z1

            # 方差更新 (Euler-Maruyama with full truncation)
            variances[:, t + 1] = (
                v
                + self.kappa * (self.theta - v_pos) * dt
                + self.xi * sqrt_v * np.sqrt(dt) * Z2
            )

        if full_state:
            return prices, variances
        return prices

    def simulate_with_times(
        self, n_steps: int, n_paths: int = 1
    ) -> tuple[np.ndarray, np.ndarray]:
        """返回 (时间数组, 价格矩阵)"""
        times = np.arange(n_steps + 1) * self.dt
        prices = self.simulate(n_steps, n_paths)
        return times, prices

    # ============ 单步生成（实时模拟用） ============

    def step(self, S_current: float, v_current: float) -> tuple[float, float]:
        """
        单步推进，返回 (新价格, 新方差)

        参数:
            S_current: 当前价格
            v_current: 当前方差

        返回:
            (S_new, v_new)
        """
        dt = self.dt
        rho = self.rho
        sqrt_1m_rho2 = np.sqrt(1.0 - rho ** 2)

        v_pos = max(v_current, 0.0)
        sqrt_v = np.sqrt(v_pos)

        W1 = self.rng.standard_normal()
        W2 = self.rng.standard_normal()
        Z1 = W1
        Z2 = rho * W1 + sqrt_1m_rho2 * W2

        S_new = S_current + self.mu * S_current * dt + sqrt_v * S_current * np.sqrt(dt) * Z1
        v_new = v_current + self.kappa * (self.theta - v_pos) * dt + self.xi * sqrt_v * np.sqrt(dt) * Z2

        return float(S_new), float(v_new)

    def reset(self, seed: Optional[int] = None):
        """重置随机数生成器"""
        if seed is not None:
            self.rng = np.random.RandomState(seed)
        else:
            self.rng = np.random.RandomState()
