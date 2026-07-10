"""
通用工具函数
"""

import numpy as np
from typing import Optional


def compute_statistics(pnl_array: np.ndarray) -> dict:
    """
    计算 PnL 数组的统计指标

    参数:
        pnl_array: 每条路径的最终 PnL

    返回:
        包含均值、标准差、胜率等指标的字典
    """
    stats = {
        "mean_pnl": np.mean(pnl_array),
        "std_pnl": np.std(pnl_array),
        "median_pnl": np.median(pnl_array),
        "win_rate": np.mean(pnl_array > 0),
        "min_pnl": np.min(pnl_array),
        "max_pnl": np.max(pnl_array),
        "p5": np.percentile(pnl_array, 5),
        "p25": np.percentile(pnl_array, 25),
        "p75": np.percentile(pnl_array, 75),
        "p95": np.percentile(pnl_array, 95),
        "skewness": float(_skewness(pnl_array)),
    }

    # Sharpe Ratio（假设无风险利率为 0，简化计算）
    if stats["std_pnl"] > 0:
        stats["sharpe_ratio"] = stats["mean_pnl"] / stats["std_pnl"]
    else:
        stats["sharpe_ratio"] = 0.0

    # 最大回撤（对 PnL 路径不适用，这里仅作为参考）
    # 实际回撤需在路径上逐点计算

    return stats


def _skewness(x: np.ndarray) -> float:
    """计算偏度"""
    n = len(x)
    if n < 3:
        return 0.0
    mean = np.mean(x)
    std = np.std(x)
    if std == 0:
        return 0.0
    return (n / ((n - 1) * (n - 2))) * np.sum(((x - mean) / std) ** 3)


def max_drawdown(pnl_path: np.ndarray) -> float:
    """
    计算单条 PnL 路径的最大回撤

    参数:
        pnl_path: 时间序列上的累计 PnL

    返回:
        最大回撤（负值）
    """
    cummax = np.maximum.accumulate(pnl_path)
    drawdown = pnl_path - cummax
    return np.min(drawdown)


def realized_volatility(
    prices: np.ndarray, dt: float, annualize: bool = True
) -> float:
    """
    计算已实现波动率

    参数:
        prices: 价格时间序列
        dt: 时间步长
        annualize: 是否年化

    返回:
        已实现波动率
    """
    log_returns = np.diff(np.log(prices))
    vol = np.std(log_returns) / np.sqrt(dt)
    if annualize:
        vol *= np.sqrt(1.0 / dt)  # 已经是年化
    return vol


def format_currency(value: float, decimals: int = 2) -> str:
    """格式化货币显示"""
    return f"¥{value:,.{decimals}f}"


def format_percentage(value: float, decimals: int = 1) -> str:
    """格式化百分比显示"""
    return f"{value * 100:.{decimals}f}%"
