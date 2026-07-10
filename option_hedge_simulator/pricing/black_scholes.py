"""
Black-Scholes 期权定价模型

欧式看涨/看跌期权解析解:
  Call = S·N(d1) - K·e^(-rT)·N(d2)
  Put  = K·e^(-rT)·N(-d2) - S·N(-d1)
"""

import numpy as np
from scipy.stats import norm
from typing import Optional


class BlackScholes:
    """Black-Scholes 期权定价"""

    @staticmethod
    def _d1(S: float, K: float, r: float, sigma: float, T: float) -> float:
        """计算 d1"""
        return (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

    @staticmethod
    def _d2(S: float, K: float, r: float, sigma: float, T: float) -> float:
        """计算 d2 = d1 - σ√T"""
        d1 = BlackScholes._d1(S, K, r, sigma, T)
        return d1 - sigma * np.sqrt(T)

    @staticmethod
    def call_price(
        S: float, K: float, r: float, sigma: float, T: float
    ) -> float:
        """
        欧式看涨期权价格

        参数:
            S: 标的价格
            K: 行权价
            r: 无风险利率（年化）
            sigma: 波动率（年化）
            T: 到期时间（年化）
        """
        if T <= 0:
            return max(S - K, 0.0)

        d1 = BlackScholes._d1(S, K, r, sigma, T)
        d2 = BlackScholes._d2(S, K, r, sigma, T)

        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

    @staticmethod
    def put_price(
        S: float, K: float, r: float, sigma: float, T: float
    ) -> float:
        """欧式看跌期权价格"""
        if T <= 0:
            return max(K - S, 0.0)

        d1 = BlackScholes._d1(S, K, r, sigma, T)
        d2 = BlackScholes._d2(S, K, r, sigma, T)

        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    @staticmethod
    def price(
        S: float,
        K: float,
        r: float,
        sigma: float,
        T: float,
        option_type: str = "call",
    ) -> float:
        """
        通用定价接口

        参数:
            option_type: "call" 或 "put"
        """
        if option_type == "call":
            return BlackScholes.call_price(S, K, r, sigma, T)
        elif option_type == "put":
            return BlackScholes.put_price(S, K, r, sigma, T)
        else:
            raise ValueError(f"不支持的期权类型: {option_type}")

    # ============ 向量化版本（用于批量计算） ============

    @staticmethod
    def call_price_vec(
        S: np.ndarray, K: float, r: float, sigma: float, T: float
    ) -> np.ndarray:
        """向量化看涨期权定价（S 为数组）"""
        if T <= 0:
            return np.maximum(S - K, 0.0)

        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

    @staticmethod
    def put_price_vec(
        S: np.ndarray, K: float, r: float, sigma: float, T: float
    ) -> np.ndarray:
        """向量化看跌期权定价"""
        if T <= 0:
            return np.maximum(K - S, 0.0)

        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    @staticmethod
    def price_vec(
        S: np.ndarray,
        K: float,
        r: float,
        sigma: float,
        T: float,
        option_type: str = "call",
    ) -> np.ndarray:
        """向量化通用定价"""
        if option_type == "call":
            return BlackScholes.call_price_vec(S, K, r, sigma, T)
        elif option_type == "put":
            return BlackScholes.put_price_vec(S, K, r, sigma, T)
        else:
            raise ValueError(f"不支持的期权类型: {option_type}")
