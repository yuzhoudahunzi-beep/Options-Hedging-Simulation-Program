"""
Delta-Gamma 对冲策略

核心思路:
  同时消除 Delta 和 Gamma 风险
  需要两个对冲工具: 标的资产 + 另一个期权（对冲期权）

  解方程组:
    w₁·Δ₁ + w₂·Δ₂ + Δ_option = 0
    w₁·Γ₁ + w₂·Γ₂ + Γ_option = 0

  标的: Δ₁ = 1, Γ₁ = 0
  对冲期权: Δ₂, Γ₂ (由 BS 公式计算)

  => w₁ = -Δ_option - w₂·Δ₂
  => w₂ = -Γ_option / Γ₂
  => w₁ = -Δ_option + (Γ_option / Γ₂) · Δ₂
"""

import numpy as np
from typing import Optional
from pricing.greeks import GreeksCalculator
from pricing.black_scholes import BlackScholes


class DeltaGammaHedgeStrategy:
    """Delta-Gamma 双因子对冲策略"""

    def __init__(
        self,
        K: float,
        r: float,
        sigma: float,
        T: float,
        option_type: str = "call",
        # 对冲期权的参数（不同行权价或不同到期日）
        K_hedge: Optional[float] = None,
        T_hedge_offset: float = 0.0,  # 对冲期权额外的到期时间
        hedge_option_type: str = "call",
        rebalance_freq: int = 1,
        transaction_cost: float = 0.0,
    ):
        """
        参数:
            K: 被对冲期权的行权价
            r: 无风险利率
            sigma: 隐含波动率
            T: 到期时间
            option_type: 被对冲期权类型
            K_hedge: 对冲期权的行权价（默认为平值，即 S0）
            T_hedge_offset: 对冲期权额外的到期时间（默认与被对冲期权相同）
            hedge_option_type: 对冲期权类型
            rebalance_freq: 调仓频率
            transaction_cost: 交易成本率
        """
        self.K = K
        self.r = r
        self.sigma = sigma
        self.T = T
        self.option_type = option_type
        self.K_hedge = K_hedge  # None 表示稍后根据 S0 设定
        self.T_hedge_offset = T_hedge_offset
        self.hedge_option_type = hedge_option_type
        self.rebalance_freq = rebalance_freq
        self.transaction_cost = transaction_cost

    def run(
        self,
        prices: np.ndarray,
        times: np.ndarray,
        S0: float,
        is_short: bool = True,
    ) -> dict:
        """
        在单条价格路径上执行 Delta-Gamma 对冲

        参数:
            prices: 标的价格路径
            times: 时间数组
            S0: 初始标的价格
            is_short: 是否为期权空头

        返回:
            dict 包含详细对冲结果
        """
        n_steps = len(prices) - 1
        K_hedge = self.K_hedge if self.K_hedge is not None else S0

        # 跟踪变量
        pnl_path = np.zeros(n_steps + 1)
        hedge_stock = np.zeros(n_steps + 1)   # 标的持仓
        hedge_option_qty = np.zeros(n_steps + 1)  # 对冲期权持仓数量
        option_values = np.zeros(n_steps + 1)
        hedge_option_values = np.zeros(n_steps + 1)
        deltas_main = np.zeros(n_steps + 1)
        gammas_main = np.zeros(n_steps + 1)
        rebalance_costs_cum = np.zeros(n_steps + 1)

        # 期权费
        opt_val_0 = BlackScholes.price(S0, self.K, self.r, self.sigma, self.T, self.option_type)
        T_hedge_0 = self.T + self.T_hedge_offset
        hedge_opt_val_0 = BlackScholes.price(S0, K_hedge, self.r, self.sigma, T_hedge_0, self.hedge_option_type)

        option_sign = -1 if is_short else 1
        premium_received = -option_sign * opt_val_0

        # 初始 Greeks
        delta_main = GreeksCalculator.delta(S0, self.K, self.r, self.sigma, self.T, self.option_type)
        gamma_main = GreeksCalculator.gamma(S0, self.K, self.r, self.sigma, self.T)
        deltas_main[0] = delta_main
        gammas_main[0] = gamma_main

        # 对冲期权的 Greeks
        delta_hedge_opt = GreeksCalculator.delta(S0, K_hedge, self.r, self.sigma, T_hedge_0, self.hedge_option_type)
        gamma_hedge_opt = GreeksCalculator.gamma(S0, K_hedge, self.r, self.sigma, T_hedge_0)

        # 计算对冲比例: 我们需要 w₂ 份对冲期权使得 Gamma neutral
        # option_sign * gamma_main + w₂ * gamma_hedge_opt = 0
        # w₂ = -option_sign * gamma_main / gamma_hedge_opt
        if abs(gamma_hedge_opt) > 1e-12:
            w2 = -option_sign * gamma_main / gamma_hedge_opt
        else:
            w2 = 0.0

        # w₁ (标的持仓): delta neutral
        # option_sign * delta_main + w₁ * 1 + w₂ * delta_hedge_opt = 0
        # w₁ = -option_sign * delta_main - w₂ * delta_hedge_opt
        w1 = -option_sign * delta_main - w2 * delta_hedge_opt

        hedge_stock[0] = w1
        hedge_option_qty[0] = w2

        cumulative_cost = 0.0
        # 初始建仓成本
        stock_cost = abs(w1) * S0 * self.transaction_cost
        option_cost = abs(w2) * hedge_opt_val_0 * self.transaction_cost
        cumulative_cost += stock_cost + option_cost

        # 逐步模拟
        for t in range(1, n_steps + 1):
            S_t = prices[t]
            T_remaining = max(self.T - times[t], 0.0)
            T_hedge_remaining = max(self.T + self.T_hedge_offset - times[t], 0.0)

            # 期权价值
            opt_val = BlackScholes.price(S_t, self.K, self.r, self.sigma, T_remaining, self.option_type)
            hedge_opt_val = BlackScholes.price(S_t, K_hedge, self.r, self.sigma, T_hedge_remaining, self.hedge_option_type)
            option_values[t] = opt_val
            hedge_option_values[t] = hedge_opt_val

            # 持仓 PnL
            prev_stock = hedge_stock[t - 1]
            prev_hedge_opt = hedge_option_qty[t - 1]

            stock_pnl = prev_stock * (S_t - prices[t - 1])
            main_opt_pnl = option_sign * (opt_val - option_values[t - 1])
            hedge_opt_pnl = w2 * (hedge_opt_val - hedge_option_values[t - 1]) if t == 1 else prev_hedge_opt * (hedge_opt_val - hedge_option_values[t - 1])

            # 调仓
            if t % self.rebalance_freq == 0 and T_remaining > 0:
                new_delta = GreeksCalculator.delta(S_t, self.K, self.r, self.sigma, T_remaining, self.option_type)
                new_gamma = GreeksCalculator.gamma(S_t, self.K, self.r, self.sigma, T_remaining)
                deltas_main[t] = new_delta
                gammas_main[t] = new_gamma

                new_delta_h = GreeksCalculator.delta(S_t, K_hedge, self.r, self.sigma, T_hedge_remaining, self.hedge_option_type)
                new_gamma_h = GreeksCalculator.gamma(S_t, K_hedge, self.r, self.sigma, T_hedge_remaining)

                if abs(new_gamma_h) > 1e-12:
                    new_w2 = -option_sign * new_gamma / new_gamma_h
                else:
                    new_w2 = 0.0

                new_w1 = -option_sign * new_delta - new_w2 * new_delta_h
            else:
                new_w1 = prev_stock
                new_w2 = prev_hedge_opt

            # 调仓成本
            stock_trade = abs(new_w1 - prev_stock)
            option_trade = abs(new_w2 - prev_hedge_opt)
            trade_cost = stock_trade * S_t * self.transaction_cost + option_trade * hedge_opt_val * self.transaction_cost
            cumulative_cost += trade_cost
            rebalance_costs_cum[t] = cumulative_cost

            hedge_stock[t] = new_w1
            hedge_option_qty[t] = new_w2

            # 累计 PnL
            pnl_path[t] = pnl_path[t - 1] + stock_pnl + main_opt_pnl + hedge_opt_pnl - trade_cost

        # 到期结算
        final_main = BlackScholes.price(prices[-1], self.K, self.r, self.sigma, 0.0, self.option_type)
        final_hedge = BlackScholes.price(prices[-1], K_hedge, self.r, self.sigma, max(self.T_hedge_offset, 0.0), self.hedge_option_type)

        # 期权赔付
        pnl_path[-1] += option_sign * final_main - option_sign * option_values[-1]
        pnl_path[-1] += hedge_option_qty[-1] * (final_hedge - hedge_option_values[-1])

        # 清算标的
        pnl_path[-1] += hedge_stock[-1] * prices[-1]

        # 总 PnL
        total_pnl = pnl_path[-1] + premium_received - cumulative_cost

        return {
            "pnl": total_pnl,
            "pnl_path": pnl_path,
            "hedge_stock": hedge_stock,
            "hedge_option_qty": hedge_option_qty,
            "option_values": option_values,
            "hedge_option_values": hedge_option_values,
            "deltas": deltas_main,
            "gammas": gammas_main,
            "rebalance_costs": rebalance_costs_cum,
            "premium_received": premium_received,
        }
