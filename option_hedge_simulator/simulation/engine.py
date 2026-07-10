"""
蒙特卡洛模拟引擎

核心功能:
  - 支持多种市场模型 (GBM / Merton / Heston)
  - 支持多种对冲策略
  - 沿价格路径执行策略
  - 统计 PnL 分布和关键指标
"""

import numpy as np
from typing import Optional
from market.gbm import GBMSimulator
from market.merton import MertonSimulator
from market.heston import HestonSimulator
from strategies.delta_hedge import DeltaHedgeStrategy
from strategies.delta_gamma_hedge import DeltaGammaHedgeStrategy
from strategies.gamma_scalping import GammaScalpingStrategy
from utils.helpers import compute_statistics, max_drawdown


class MonteCarloEngine:
    """蒙特卡洛模拟引擎"""

    def __init__(
        self,
        S0: float = 100.0,
        K: float = 100.0,
        r: float = 0.05,
        sigma: float = 0.2,
        T: float = 1.0,
        n_steps: int = 252,
        n_paths: int = 1000,
        seed: Optional[int] = None,
    ):
        """
        参数:
            S0: 初始标的价格
            K: 行权价
            r: 无风险利率
            sigma: 波动率
            T: 到期时间（年）
            n_steps: 时间步数（如 252 表示日步长）
            n_paths: 蒙特卡洛路径数
            seed: 随机种子
        """
        self.S0 = S0
        self.K = K
        self.r = r
        self.sigma = sigma
        self.T = T
        self.n_steps = n_steps
        self.n_paths = n_paths
        self.seed = seed
        self.dt = T / n_steps

    # ============ 市场模型工厂 ============

    def _create_market_model(
        self,
        model_type: str,
        model_params: Optional[dict] = None,
    ):
        """创建市场模拟器"""
        params = model_params or {}

        if model_type == "gbm":
            return GBMSimulator(
                S0=self.S0,
                mu=params.get("mu", self.r),
                sigma=params.get("sigma", self.sigma),
                dt=self.dt,
                seed=self.seed,
            )
        elif model_type == "merton":
            return MertonSimulator(
                S0=self.S0,
                mu=params.get("mu", self.r),
                sigma=params.get("sigma", self.sigma),
                dt=self.dt,
                lam=params.get("lam", 1.0),
                mu_J=params.get("mu_J", 0.0),
                sigma_J=params.get("sigma_J", 0.1),
                seed=self.seed,
            )
        elif model_type == "heston":
            return HestonSimulator(
                S0=self.S0,
                mu=params.get("mu", self.r),
                v0=params.get("v0", self.sigma ** 2),
                kappa=params.get("kappa", 2.0),
                theta=params.get("theta", self.sigma ** 2),
                xi=params.get("xi", 0.3),
                rho=params.get("rho", -0.7),
                dt=self.dt,
                seed=self.seed,
            )
        else:
            raise ValueError(f"不支持的市场模型: {model_type}")

    # ============ 策略工厂 ============

    def _create_strategy(
        self,
        strategy_type: str,
        strategy_params: Optional[dict] = None,
    ):
        """创建对冲策略"""
        params = strategy_params or {}

        if strategy_type == "delta":
            return DeltaHedgeStrategy(
                K=self.K,
                r=self.r,
                sigma=self.sigma,
                T=self.T,
                option_type=params.get("option_type", "call"),
                rebalance_freq=params.get("rebalance_freq", 1),
                transaction_cost=params.get("transaction_cost", 0.0),
            )
        elif strategy_type == "delta_gamma":
            return DeltaGammaHedgeStrategy(
                K=self.K,
                r=self.r,
                sigma=self.sigma,
                T=self.T,
                option_type=params.get("option_type", "call"),
                K_hedge=params.get("K_hedge", None),
                T_hedge_offset=params.get("T_hedge_offset", 0.0),
                hedge_option_type=params.get("hedge_option_type", "call"),
                rebalance_freq=params.get("rebalance_freq", 1),
                transaction_cost=params.get("transaction_cost", 0.0),
            )
        elif strategy_type == "gamma_scalping":
            return GammaScalpingStrategy(
                K=self.K,
                r=self.r,
                implied_vol=params.get("implied_vol", self.sigma),
                T=self.T,
                option_type=params.get("option_type", "call"),
                rebalance_freq=params.get("rebalance_freq", 1),
                transaction_cost=params.get("transaction_cost", 0.0),
            )
        else:
            raise ValueError(f"不支持的策略类型: {strategy_type}")

    # ============ 核心模拟 ============

    def run_simulation(
        self,
        model_type: str = "gbm",
        model_params: Optional[dict] = None,
        strategy_type: str = "delta",
        strategy_params: Optional[dict] = None,
        is_short: bool = True,
        show_paths: int = 50,
        progress_callback=None,
    ) -> dict:
        """
        运行蒙特卡洛模拟

        参数:
            model_type: 市场模型 ("gbm", "merton", "heston")
            model_params: 市场模型参数
            strategy_type: 策略类型 ("delta", "delta_gamma", "gamma_scalping")
            strategy_params: 策略参数
            is_short: 是否为期权卖方（Gamma Scalping 不适用此参数）
            show_paths: 展示多少条路径的详细信息
            progress_callback: 进度回调函数

        返回:
            dict 包含:
              - prices: 所有价格路径
              - pnl_array: 每条路径的最终 PnL
              - pnl_paths: 前 show_paths 条路径的 PnL 演化
              - stats: 统计指标
              - sample_result: 第一条路径的详细结果
              - times: 时间数组
              - model_type, strategy_type: 记录使用的模型和策略
        """
        # 1. 生成价格路径
        market_model = self._create_market_model(model_type, model_params)
        all_prices = market_model.simulate(self.n_steps, self.n_paths)
        times = np.arange(self.n_steps + 1) * self.dt

        # 2. 对每条路径执行策略
        strategy = self._create_strategy(strategy_type, strategy_params)

        pnl_array = np.zeros(self.n_paths)
        pnl_paths = []
        sample_result = None

        for i in range(self.n_paths):
            path_prices = all_prices[i]

            if strategy_type == "gamma_scalping":
                result = strategy.run(path_prices, times, self.S0)
            else:
                result = strategy.run(path_prices, times, self.S0, is_short=is_short)

            pnl_array[i] = result["pnl"]

            if i < show_paths:
                pnl_paths.append(result["pnl_path"])

            if i == 0:
                sample_result = result

            if progress_callback and i % max(1, self.n_paths // 20) == 0:
                progress = (i + 1) / self.n_paths
                progress_callback(progress)

        pnl_paths = np.array(pnl_paths)

        # 3. 计算统计指标
        stats = compute_statistics(pnl_array)

        # 最大回撤统计
        drawdowns = []
        for i in range(min(self.n_paths, show_paths)):
            if i < len(pnl_paths):
                dd = max_drawdown(pnl_paths[i])
                drawdowns.append(dd)
        if drawdowns:
            stats["avg_max_drawdown"] = np.mean(drawdowns)
            stats["worst_max_drawdown"] = np.min(drawdowns)

        return {
            "prices": all_prices,
            "pnl_array": pnl_array,
            "pnl_paths": pnl_paths,
            "stats": stats,
            "sample_result": sample_result,
            "times": times,
            "model_type": model_type,
            "strategy_type": strategy_type,
            "n_paths": self.n_paths,
            "n_steps": self.n_steps,
        }

    # ============ 便捷方法 ============

    def compare_strategies(
        self,
        model_type: str = "gbm",
        model_params: Optional[dict] = None,
        is_short: bool = True,
        progress_callback=None,
    ) -> dict:
        """
        对比三种策略的表现

        返回:
            dict: {strategy_name: simulation_result}
        """
        strategies = ["delta", "delta_gamma", "gamma_scalping"]
        results = {}

        for i, strat in enumerate(strategies):
            if progress_callback:
                progress_callback((i) / len(strategies))

            sp = {}
            result = self.run_simulation(
                model_type=model_type,
                model_params=model_params,
                strategy_type=strat,
                strategy_params=sp,
                is_short=is_short,
                show_paths=50,
            )
            results[strat] = result

        if progress_callback:
            progress_callback(1.0)

        return results
