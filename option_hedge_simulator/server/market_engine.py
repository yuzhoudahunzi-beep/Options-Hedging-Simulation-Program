"""
实时市场引擎 — 逐步生成价格 + 隐含波动率

两种模式:
  1. 纯模拟: Heston 随机波动率模型逐步生成 (S, IV)
  2. 历史回放: 加载历史日线数据 + Heston 噪声叠加

每 tick 输出: {S, IV, dS_pct, dIV_pct, timestamp, step}
"""

import numpy as np
import pandas as pd
from typing import Optional
from market.heston import HestonSimulator
from market.gbm import GBMSimulator
from server.data_loader import load_history, UNDERLYINGS


class MarketEngine:
    """实时市场引擎"""

    def __init__(
        self,
        underlying: str = "50ETF",
        mode: str = "simulate",       # "simulate" | "replay"
        model: str = "heston",        # "gbm" | "heston"
        S0: Optional[float] = None,
        iv0: Optional[float] = None,
        tick_dt: float = 1.0 / 252,  # 每 tick 对应的时间（年）
        difficulty: str = "medium",   # "easy" | "medium" | "hard"
        history_days: int = 120,
        seed: Optional[int] = None,
    ):
        """
        参数:
            underlying: 标的代码 ("50ETF" / "300ETF")
            mode: "simulate" 纯模拟 | "replay" 历史回放
            model: 市场模型 ("gbm" | "heston")
            S0: 初始价格（None 则用标的默认值）
            iv0: 初始隐含波动率（None 则用标的默认值）
            tick_dt: 每 tick 对应的时间
            difficulty: 难度（影响波动率和跳跃概率）
            history_days: 历史数据天数
            seed: 随机种子
        """
        self.underlying = underlying
        self.mode = mode
        self.model_type = model
        self.tick_dt = tick_dt
        self.difficulty = difficulty
        self.seed = seed

        info = UNDERLYINGS.get(underlying, UNDERLYINGS["50ETF"])
        self.underlying_name = info["name"]

        # 难度参数（提高波动性和跳跃概率，让游戏更刺激）
        diff_params = {
            "easy":   {"vol_scale": 1.0, "jump_prob": 0.005, "iv_vol_of_vol": 0.25},
            "medium": {"vol_scale": 1.5, "jump_prob": 0.012, "iv_vol_of_vol": 0.45},
            "hard":   {"vol_scale": 2.2, "jump_prob": 0.025, "iv_vol_of_vol": 0.70},
        }
        self.dp = diff_params.get(difficulty, diff_params["medium"])

        # 初始参数
        self.S0 = S0 or info["default_S0"]
        self.iv0 = iv0 or info["default_iv"]

        # 当前状态
        self.S = self.S0
        self.prev_S = self.S0
        self.iv = self.iv0
        self.prev_iv = self.iv0
        self.step_count = 0

        # Heston 状态（方差过程驱动 IV）
        self.v = self.iv0 ** 2  # 方差 = IV^2

        # 加载历史数据（replay 模式）
        self.history = None
        self.history_idx = 0
        if mode == "replay":
            self.history = load_history(underlying, days=history_days)
            if self.history is not None and len(self.history) > 0:
                # 从历史数据的中间开始，留一些预热空间
                self.history_idx = max(20, len(self.history) // 4)
                self.S = float(self.history.iloc[self.history_idx]["close"])
                self.S0 = self.S
                self.prev_S = self.S
            else:
                self.mode = "simulate"  # 回退

        # 创建底层模拟器
        self._init_simulator()

    def _init_simulator(self):
        """初始化底层模拟器"""
        if self.model_type == "heston":
            self.simulator = HestonSimulator(
                S0=self.S,
                mu=0.02,  # 低漂移
                v0=self.v,
                kappa=3.0,           # 均值回归速度
                theta=self.iv0 ** 2,  # 长期方差
                xi=self.dp["iv_vol_of_vol"],  # vol of vol
                rho=-0.7,            # 价格-波动率负相关
                dt=self.tick_dt,
                seed=self.seed,
            )
        else:
            self.simulator = GBMSimulator(
                S0=self.S,
                mu=0.02,
                sigma=self.iv0,
                dt=self.tick_dt,
                seed=self.seed,
            )

    def step(self) -> dict:
        """
        推进一步，返回新市场状态

        返回:
            {
                "S": 新价格,
                "IV": 新隐含波动率,
                "dS_pct": 价格变动百分比,
                "dIV_pct": IV变动（绝对值）,
                "step": 当前步数,
                "timestamp": 时间戳字符串,
                "event": 特殊事件 ("none" | "jump" | "vol_spike" | "vol_crash"),
            }
        """
        self.prev_S = self.S
        self.prev_IV = self.iv

        event = "none"

        if self.mode == "replay" and self.history is not None:
            # 历史回放模式
            self.history_idx += 1
            if self.history_idx >= len(self.history):
                self.history_idx = len(self.history) - 1

            row = self.history.iloc[self.history_idx]
            hist_close = float(row["close"])

            # 历史价格做骨架 + Heston 噪声
            if self.model_type == "heston":
                S_sim, v_new = self.simulator.step(self.S, self.v)
                self.v = max(v_new, 0.0001)
                # 混合：70% 历史趋势 + 30% 模拟噪声
                hist_return = hist_close / float(self.history.iloc[self.history_idx - 1]["close"]) if self.history_idx > 0 else 1.0
                noise = S_sim / self.S if self.S > 0 else 1.0
                combined_return = (hist_return ** 0.7) * (noise ** 0.3)
                self.S = self.prev_S * combined_return
            else:
                S_new = self.simulator.step(self.S)
                hist_return = hist_close / float(self.history.iloc[self.history_idx - 1]["close"]) if self.history_idx > 0 else 1.0
                noise = S_new / self.S if self.S > 0 else 1.0
                combined_return = (hist_return ** 0.7) * (noise ** 0.3)
                self.S = self.prev_S * combined_return

            # IV 由 Heston 方差驱动
            self.iv = np.sqrt(max(self.v, 0.01))

        else:
            # 纯模拟模式
            if self.model_type == "heston":
                S_new, v_new = self.simulator.step(self.S, self.v)
                self.S = max(S_new, 0.01)
                self.v = max(v_new, 0.0001)
                self.iv = np.sqrt(self.v)
            else:
                S_new, _ = self.simulator.step(self.S)
                self.S = max(S_new, 0.01)
                # GBM 模式下 IV 独立随机游走
                iv_noise = np.random.standard_normal() * self.dp["iv_vol_of_vol"] * np.sqrt(self.tick_dt)
                self.iv = max(self.iv + iv_noise, 0.05)

        # 随机波动率冲击事件
        rng = np.random.random()
        if rng < self.dp["jump_prob"] * 0.5:
            # 波动率飙升
            spike = 0.03 + np.random.exponential(0.05)
            self.iv = min(self.iv + spike, 1.0)
            self.v = self.iv ** 2
            event = "vol_spike"
        elif rng < self.dp["jump_prob"]:
            # 波动率坍塌
            crash = 0.02 + np.random.exponential(0.03)
            self.iv = max(self.iv - crash, 0.05)
            self.v = self.iv ** 2
            event = "vol_crash"

        # 价格跳跃（叠加在正常波动上）— 更大幅度的涨跌
        if np.random.random() < self.dp["jump_prob"]:
            jump_size = np.random.choice([-1, 1]) * np.random.exponential(0.035) * self.dp["vol_scale"]
            self.S = self.S * (1 + jump_size)
            if event == "none":
                event = "jump"

        self.step_count += 1

        dS_pct = (self.S - self.prev_S) / self.prev_S if self.prev_S > 0 else 0.0
        dIV = self.iv - self.prev_IV

        return {
            "S": round(self.S, 4),
            "IV": round(self.iv, 4),
            "dS_pct": round(dS_pct, 6),
            "dIV": round(dIV, 6),
            "step": self.step_count,
            "event": event,
        }

    def get_state(self) -> dict:
        """获取当前市场状态快照"""
        return {
            "S": round(self.S, 4),
            "IV": round(self.iv, 4),
            "S0": round(self.S0, 4),
            "iv0": round(self.iv0, 4),
            "step": self.step_count,
            "underlying": self.underlying,
            "underlying_name": self.underlying_name,
            "mode": self.mode,
            "difficulty": self.difficulty,
        }

    def reset(self, seed: Optional[int] = None):
        """重置引擎"""
        self.S = self.S0
        self.prev_S = self.S0
        self.iv = self.iv0
        self.prev_IV = self.iv0
        self.v = self.iv0 ** 2
        self.step_count = 0
        self.history_idx = max(20, len(self.history) // 4) if self.history is not None else 0
        if seed is not None:
            self.seed = seed
        self._init_simulator()
