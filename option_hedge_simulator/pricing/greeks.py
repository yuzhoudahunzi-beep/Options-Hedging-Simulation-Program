"""
Greeks 计算器

解析 Greeks:
  Delta = ∂V/∂S
  Gamma = ∂²V/∂S²
  Theta = ∂V/∂t
  Vega  = ∂V/∂σ
  Rho   = ∂V/∂r
"""

import numpy as np
from scipy.stats import norm
from typing import Optional
from pricing.black_scholes import BlackScholes


class GreeksCalculator:
    """期权 Greeks 计算器（解析 + 数值）"""

    # ============ 解析 Greeks ============

    @staticmethod
    def delta(
        S: float, K: float, r: float, sigma: float, T: float,
        option_type: str = "call"
    ) -> float:
        """Delta = ∂V/∂S"""
        if T <= 0:
            if option_type == "call":
                return 1.0 if S > K else 0.0
            else:
                return -1.0 if S < K else 0.0

        d1 = BlackScholes._d1(S, K, r, sigma, T)

        if option_type == "call":
            return norm.cdf(d1)
        else:
            return norm.cdf(d1) - 1.0

    @staticmethod
    def gamma(
        S: float, K: float, r: float, sigma: float, T: float
    ) -> float:
        """Gamma = ∂²V/∂S²（Call 和 Put 相同）"""
        if T <= 0:
            return 0.0

        d1 = BlackScholes._d1(S, K, r, sigma, T)
        return norm.pdf(d1) / (S * sigma * np.sqrt(T))

    @staticmethod
    def theta(
        S: float, K: float, r: float, sigma: float, T: float,
        option_type: str = "call"
    ) -> float:
        """Theta = ∂V/∂t（注意：通常 Θ < 0 表示时间衰减）"""
        if T <= 0:
            return 0.0

        d1 = BlackScholes._d1(S, K, r, sigma, T)
        d2 = d1 - sigma * np.sqrt(T)

        common = -(S * norm.pdf(d1) * sigma) / (2.0 * np.sqrt(T))

        if option_type == "call":
            return common - r * K * np.exp(-r * T) * norm.cdf(d2)
        else:
            return common + r * K * np.exp(-r * T) * norm.cdf(-d2)

    @staticmethod
    def vega(
        S: float, K: float, r: float, sigma: float, T: float
    ) -> float:
        """Vega = ∂V/∂σ（Call 和 Put 相同）"""
        if T <= 0:
            return 0.0

        d1 = BlackScholes._d1(S, K, r, sigma, T)
        return S * norm.pdf(d1) * np.sqrt(T)

    @staticmethod
    def rho(
        S: float, K: float, r: float, sigma: float, T: float,
        option_type: str = "call"
    ) -> float:
        """Rho = ∂V/∂r"""
        if T <= 0:
            return 0.0

        d1 = BlackScholes._d1(S, K, r, sigma, T)
        d2 = d1 - sigma * np.sqrt(T)

        if option_type == "call":
            return K * T * np.exp(-r * T) * norm.cdf(d2)
        else:
            return -K * T * np.exp(-r * T) * norm.cdf(-d2)

    @staticmethod
    def all_greeks(
        S: float, K: float, r: float, sigma: float, T: float,
        option_type: str = "call"
    ) -> dict:
        """计算所有 Greeks 并返回字典"""
        return {
            "price": BlackScholes.price(S, K, r, sigma, T, option_type),
            "delta": GreeksCalculator.delta(S, K, r, sigma, T, option_type),
            "gamma": GreeksCalculator.gamma(S, K, r, sigma, T),
            "theta": GreeksCalculator.theta(S, K, r, sigma, T, option_type),
            "vega": GreeksCalculator.vega(S, K, r, sigma, T),
            "rho": GreeksCalculator.rho(S, K, r, sigma, T, option_type),
        }

    # ============ 向量化 Greeks（用于批量计算） ============

    @staticmethod
    def delta_vec(
        S: np.ndarray, K: float, r: float, sigma: float, T: float,
        option_type: str = "call"
    ) -> np.ndarray:
        """向量化 Delta"""
        if T <= 0:
            if option_type == "call":
                return np.where(S > K, 1.0, 0.0)
            else:
                return np.where(S < K, -1.0, 0.0)

        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

        if option_type == "call":
            return norm.cdf(d1)
        else:
            return norm.cdf(d1) - 1.0

    @staticmethod
    def gamma_vec(
        S: np.ndarray, K: float, r: float, sigma: float, T: float
    ) -> np.ndarray:
        """向量化 Gamma"""
        if T <= 0:
            return np.zeros_like(S)

        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        return norm.pdf(d1) / (S * sigma * np.sqrt(T))

    @staticmethod
    def theta_vec(
        S: np.ndarray, K: float, r: float, sigma: float, T: float,
        option_type: str = "call"
    ) -> np.ndarray:
        """向量化 Theta"""
        if T <= 0:
            return np.zeros_like(S)

        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        common = -(S * norm.pdf(d1) * sigma) / (2.0 * np.sqrt(T))

        if option_type == "call":
            return common - r * K * np.exp(-r * T) * norm.cdf(d2)
        else:
            return common + r * K * np.exp(-r * T) * norm.cdf(-d2)

    # ============ 数值 Greeks（有限差分法，用于校验） ============

    @staticmethod
    def numerical_delta(
        S: float, K: float, r: float, sigma: float, T: float,
        option_type: str = "call", dS: float = 0.01
    ) -> float:
        """数值 Delta（中心差分）"""
        up = BlackScholes.price(S + dS, K, r, sigma, T, option_type)
        dn = BlackScholes.price(S - dS, K, r, sigma, T, option_type)
        return (up - dn) / (2 * dS)

    @staticmethod
    def numerical_gamma(
        S: float, K: float, r: float, sigma: float, T: float,
        option_type: str = "call", dS: float = 0.01
    ) -> float:
        """数值 Gamma（中心差分）"""
        up = BlackScholes.price(S + dS, K, r, sigma, T, option_type)
        mid = BlackScholes.price(S, K, r, sigma, T, option_type)
        dn = BlackScholes.price(S - dS, K, r, sigma, T, option_type)
        return (up - 2 * mid + dn) / (dS ** 2)
