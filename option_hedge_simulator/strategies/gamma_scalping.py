"""
Gamma Scalping 策略

核心思路:
  做市商策略 — 买入期权（Long Gamma），Delta 对冲维持 Delta Neutral
  每次对冲调整本质上是"低买高卖"（因为 Long Gamma 的凸性收益）

  盈利条件: 已实现波动率 > 买入时的隐含波动率
  关键等式: Gamma PnL ≈ 0.5·Γ·(dS)² - |Θ|·dt

  如果 realized_var > implied_var²:
    Gamma 收益 > Theta 衰减 → 盈利
  否则:
    亏损
"""

import numpy as np
from typing import Optional
from pricing.greeks import GreeksCalculator
from pricing.black_scholes import BlackScholes


class GammaScalpingStrategy:
    """Gamma Scalping 策略（Long Gamma + Delta 对冲）"""

    def __init__(
        self,
        K: float,
        r: float,
        implied_vol: float,
        T: float,
        option_type: str = "call",
        rebalance_freq: int = 1,
        transaction_cost: float = 0.0,
    ):
        """
        参数:
            K: 行权价
            r: 无风险利率
            implied_vol: 买入期权时的隐含波动率
            T: 到期时间
            option_type: 期权类型
            rebalance_freq: 调仓频率
            transaction_cost: 交易成本率
        """
        self.K = K
        self.r = r
        self.implied_vol = implied_vol
        self.T = T
        self.option_type = option_type
        self.rebalance_freq = rebalance_freq
        self.transaction_cost = transaction_cost

    def run(
        self,
        prices: np.ndarray,
        times: np.ndarray,
        S0: float,
    ) -> dict:
        """
        在单条价格路径上执行 Gamma Scalping

        参数:
            prices: 标的价格路径
            times: 时间数组
            S0: 初始标的价格

        返回:
            dict 包含详细结果，含 Gamma/Theta 收益分解
        """
        n_steps = len(prices) - 1

        # 跟踪变量
        pnl_path = np.zeros(n_steps + 1)
        hedge_positions = np.zeros(n_steps + 1)
        option_values = np.zeros(n_steps + 1)
        deltas = np.zeros(n_steps + 1)
        gammas = np.zeros(n_steps + 1)
        thetas = np.zeros(n_steps + 1)
        gamma_pnl_components = np.zeros(n_steps + 1)  # 0.5 * Gamma * (dS)^2
        theta_pnl_components = np.zeros(n_steps + 1)  # Theta * dt
        rebalance_costs_cum = np.zeros(n_steps + 1)

        # 初始买入期权（Long position）
        opt_val_0 = BlackScholes.price(S0, self.K, self.r, self.implied_vol, self.T, self.option_type)
        premium_paid = opt_val_0  # 支付期权费

        # Long 期权 = +1 份
        option_sign = 1  # 买入

        # 初始 Greeks
        delta_0 = GreeksCalculator.delta(S0, self.K, self.r, self.implied_vol, self.T, self.option_type)
        gamma_0 = GreeksCalculator.gamma(S0, self.K, self.r, self.implied_vol, self.T)
        theta_0 = GreeksCalculator.theta(S0, self.K, self.r, self.implied_vol, self.T, self.option_type)

        deltas[0] = delta_0
        gammas[0] = gamma_0
        thetas[0] = theta_0
        option_values[0] = opt_val_0

        # 初始 Delta 对冲
        # 组合 = 期权 + 标的
        # Delta neutral: option_delta + hedge * 1 = 0
        # hedge = -option_delta
        initial_hedge = -delta_0
        hedge_positions[0] = initial_hedge

        cumulative_cost = abs(initial_hedge) * S0 * self.transaction_cost
        cumulative_gamma_pnl = 0.0
        cumulative_theta_pnl = 0.0

        # 逐步模拟
        for t in range(1, n_steps + 1):
            S_t = prices[t]
            dS = S_t - prices[t - 1]
            T_remaining = max(self.T - times[t], 0.0)
            dt = times[t] - times[t - 1] if t > 0 else 0

            # 期权价值变化
            opt_val = BlackScholes.price(S_t, self.K, self.r, self.implied_vol, T_remaining, self.option_type)
            option_values[t] = opt_val

            # 标的持仓 PnL
            prev_hedge = hedge_positions[t - 1]
            stock_pnl = prev_hedge * dS

            # 期权 PnL
            opt_pnl = option_sign * (opt_val - option_values[t - 1])

            # Gamma 和 Theta 收益分解
            gamma_component = 0.5 * gammas[t - 1] * dS ** 2
            theta_component = thetas[t - 1] * dt  # Theta 通常为负

            cumulative_gamma_pnl += gamma_component
            cumulative_theta_pnl += theta_component
            gamma_pnl_components[t] = cumulative_gamma_pnl
            theta_pnl_components[t] = cumulative_theta_pnl

            # 调仓
            if t % self.rebalance_freq == 0 and T_remaining > 0:
                new_delta = GreeksCalculator.delta(S_t, self.K, self.r, self.implied_vol, T_remaining, self.option_type)
                new_gamma = GreeksCalculator.gamma(S_t, self.K, self.r, self.implied_vol, T_remaining)
                new_theta = GreeksCalculator.theta(S_t, self.K, self.r, self.implied_vol, T_remaining, self.option_type)

                deltas[t] = new_delta
                gammas[t] = new_gamma
                thetas[t] = new_theta

                new_hedge = -new_delta  # Delta neutral
            else:
                new_hedge = prev_hedge

            # 调仓成本
            trade_size = abs(new_hedge - prev_hedge)
            trade_cost = trade_size * S_t * self.transaction_cost
            cumulative_cost += trade_cost
            rebalance_costs_cum[t] = cumulative_cost

            hedge_positions[t] = new_hedge

            # 累计 PnL = 标的PnL + 期权PnL - 调仓成本
            pnl_path[t] = pnl_path[t - 1] + stock_pnl + opt_pnl - trade_cost

        # 到期结算
        final_val = BlackScholes.price(prices[-1], self.K, self.r, self.implied_vol, 0.0, self.option_type)
        pnl_path[-1] += option_sign * final_val - option_sign * option_values[-1]
        pnl_path[-1] += hedge_positions[-1] * prices[-1]

        # 总 PnL = 对冲收益 - 支付的期权费 - 交易成本
        total_pnl = pnl_path[-1] - premium_paid - cumulative_cost

        # 计算已实现波动率
        log_returns = np.diff(np.log(prices))
        if len(log_returns) > 0 and n_steps > 0:
            dt_step = times[1] - times[0]
            realized_vol = np.std(log_returns) / np.sqrt(dt_step) * np.sqrt(1.0)  # 年化
        else:
            realized_vol = 0.0

        return {
            "pnl": total_pnl,
            "pnl_path": pnl_path,
            "hedge_positions": hedge_positions,
            "option_values": option_values,
            "deltas": deltas,
            "gammas": gammas,
            "thetas": thetas,
            "gamma_pnl_components": gamma_pnl_components,
            "theta_pnl_components": theta_pnl_components,
            "rebalance_costs": rebalance_costs_cum,
            "premium_paid": premium_paid,
            "realized_vol": realized_vol,
            "implied_vol": self.implied_vol,
            "vol_diff": realized_vol - self.implied_vol,
        }
