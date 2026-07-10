"""
历史数据加载器 — 从 AKShare 爬取中国 ETF 历史行情

支持标的:
  - 510050: 50ETF (上证50ETF)
  - 510300: 300ETF (沪深300ETF)

数据缓存到 data/ 目录，避免重复爬取。
网络不可用时自动回退到模拟数据。
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Optional
from datetime import datetime, timedelta

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


# 常用标的信息
UNDERLYINGS = {
    "50ETF": {
        "code": "510050",
        "name": "50ETF (上证50)",
        "default_S0": 3.200,
        "default_iv": 0.18,
    },
    "300ETF": {
        "code": "510300",
        "name": "300ETF (沪深300)",
        "default_S0": 4.000,
        "default_iv": 0.20,
    },
}


def fetch_etf_history(
    symbol: str = "510050",
    days: int = 250,
    force_refresh: bool = False,
) -> Optional[pd.DataFrame]:
    """
    从 AKShare 获取 ETF 历史日线数据

    参数:
        symbol: 股票代码
        days: 获取多少天的数据
        force_refresh: 是否强制刷新缓存

    返回:
        DataFrame with columns: [date, open, high, low, close, volume]
        失败时返回 None
    """
    cache_file = os.path.join(DATA_DIR, f"{symbol}_daily.csv")

    # 检查缓存
    if not force_refresh and os.path.exists(cache_file):
        try:
            df = pd.read_csv(cache_file, parse_dates=["date"])
            if len(df) > 50:
                print(f"[DataLoader] 使用缓存数据: {cache_file} ({len(df)} 条)")
                return df.tail(days)
        except Exception:
            pass

    # 尝试从 AKShare 获取
    try:
        import akshare as ak

        end_date = datetime.now().strftime("%Y%m%d")
        start_date = (datetime.now() - timedelta(days=int(days * 1.5))).strftime("%Y%m%d")

        print(f"[DataLoader] 正在从 AKShare 获取 {symbol} 历史数据...")

        # akshare 的 ETF 日线接口
        df = ak.fund_etf_hist_em(
            symbol=symbol,
            period="daily",
            start_date=start_date,
            end_date=end_date,
            adjust="qfq",  # 前复权
        )

        if df is None or df.empty:
            raise ValueError("AKShare 返回空数据")

        # 标准化列名
        df = df.rename(columns={
            "日期": "date",
            "开盘": "open",
            "最高": "high",
            "最低": "low",
            "收盘": "close",
            "成交量": "volume",
        })

        # 只保留需要的列
        cols = ["date", "open", "high", "low", "close", "volume"]
        df = df[[c for c in cols if c in df.columns]].copy()

        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])

        df = df.sort_values("date").reset_index(drop=True)

        # 缓存
        os.makedirs(DATA_DIR, exist_ok=True)
        df.to_csv(cache_file, index=False)
        print(f"[DataLoader] 数据已缓存: {cache_file} ({len(df)} 条)")

        return df.tail(days)

    except ImportError:
        print("[DataLoader] AKShare 未安装，使用模拟数据")
        return None
    except Exception as e:
        print(f"[DataLoader] 数据获取失败: {e}，使用模拟数据")
        return None


def generate_synthetic_history(
    S0: float = 100.0,
    sigma: float = 0.20,
    mu: float = 0.05,
    n_days: int = 250,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    生成模拟的历史日线数据（用于无法获取真实数据时）

    返回:
        DataFrame with columns: [date, open, high, low, close, volume]
    """
    rng = np.random.RandomState(seed)
    dt = 1.0 / 252

    prices = [S0]
    for _ in range(n_days - 1):
        Z = rng.standard_normal()
        ret = (mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z
        prices.append(prices[-1] * np.exp(ret))

    prices = np.array(prices)

    # 生成 OHLC
    daily_vol = sigma * np.sqrt(dt)
    opens = prices * (1 + rng.standard_normal(n_days) * daily_vol * 0.3)
    highs = np.maximum(prices, opens) * (1 + np.abs(rng.standard_normal(n_days)) * daily_vol * 0.5)
    lows = np.minimum(prices, opens) * (1 - np.abs(rng.standard_normal(n_days)) * daily_vol * 0.5)

    dates = pd.date_range(end=datetime.now(), periods=n_days, freq="B")  # 工作日

    df = pd.DataFrame({
        "date": dates,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": prices,
        "volume": (rng.lognormal(15, 1, n_days)).astype(int),
    })

    return df


def load_history(
    underlying_key: str = "50ETF",
    days: int = 250,
    force_simulate: bool = False,
) -> pd.DataFrame:
    """
    加载历史数据（优先真实数据，回退模拟数据）

    参数:
        underlying_key: 标的键名（"50ETF" / "300ETF"）
        days: 需要的天数
        force_simulate: 强制使用模拟数据

    返回:
        DataFrame with OHLCV data
    """
    info = UNDERLYINGS.get(underlying_key, UNDERLYINGS["50ETF"])
    symbol = info["code"]

    if not force_simulate:
        df = fetch_etf_history(symbol=symbol, days=days)
        if df is not None and len(df) > 0:
            return df

    # 回退到模拟数据
    S0 = info["default_S0"]
    print(f"[DataLoader] 生成模拟历史数据 (S0={S0})")
    return generate_synthetic_history(S0=S0, n_days=days)


def get_realized_vol_from_history(df: pd.DataFrame, window: int = 20) -> float:
    """从历史数据计算已实现波动率"""
    if len(df) < window + 1:
        return 0.20  # 默认值
    log_returns = np.diff(np.log(df["close"].values[-window - 1:]))
    return float(np.std(log_returns) * np.sqrt(252))


if __name__ == "__main__":
    # 测试
    df = load_history("50ETF", days=60)
    print(f"\n数据形状: {df.shape}")
    print(df.head())
    print(f"\n已实现波动率: {get_realized_vol_from_history(df):.2%}")
