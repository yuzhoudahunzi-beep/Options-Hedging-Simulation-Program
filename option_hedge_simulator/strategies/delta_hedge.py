"""
Delta Neutral 对冲策略

核心思路:
  卖出期权后，持有 Δ 份标的资产使组合 Delta = 0
  每个调仓步重新计算 Delta，调整标的头寸

PnL 构成:
  + 期权费收入（卖出期权时）
  - 调仓成本（交易标的的摩擦成本）
  + 对冲损益（标的持仓的价格变动）
  - 到期赔付
"""

import numpy as np
from typing import Optional
from pricing.greeks import GreeksCalculator
from pricing.black_scholes import BlackScholes


class DeltaHedgeStrategy:
    """Delta Neutral 对冲策略"""

    def __init__(
        self,
        K: float,
        r: float,
        sigma: float,
        T: float,
        option_type: str = "call",
        rebalance_freq: int = 1,
        transaction_cost: float = 0.0,
    ):
        """
        参数:
            K: 行权价
            r: 无风险利率
            sigma: 隐含波动率（用于计算 Delta）
            T: 到期时间（年化）
            option_type: "call" 或 "put"
            rebalance_freq: 每 N 步调仓一次
            transaction_cost: 交易成本率（如 0.001 = 0.1%）
        """
        self.K = K
        self.r = r
        self.sigma = sigma
        self.T = T
        self.option_type = option_type
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
        在单条价格路径上执行 Delta 对冲

        参数:
            prices: 标的价格路径 shape=(n_steps+1,)
            times: 时间数组 shape=(n_steps+1,)
            S0: 初始标的价格
            is_short: True=卖出期权（Short），False=买入期权（Long）

        返回:
            dict 包含:
              - pnl: 总 PnL
              - pnl_path: 每步累计 PnL
              - hedge_positions: 每步的标的持仓
              - option_values: 每步的期权价值
              - deltas: 每步的 Delta
              - rebalance_costs: 调仓成本累计
              - gamma_pnl: Gamma 收益分解
        """
        n_steps = len(prices) - 1
        dt = times[1] - times[0] if n_steps > 0 else 0

        # 跟踪变量
        pnl_path = np.zeros(n_steps + 1)
        hedge_positions = np.zeros(n_steps + 1)  # 标的持仓数量
        option_values = np.zeros(n_steps + 1)
        deltas = np.zeros(n_steps + 1)
        rebalance_costs_cum = np.zeros(n_steps + 1)

        # 初始状态
        T_remaining = self.T
        option_value = BlackScholes.price(S0, self.K, self.r, self.sigma, T_remaining, self.option_type)
        option_values[0] = option_value

        # 初始 Delta
        current_delta = GreeksCalculator.delta(S0, self.K, self.r, self.sigma, T_remaining, self.option_type)
        deltas[0] = current_delta

        # 期权费收入/支出
        if is_short:
            premium_received = option_value  # 卖出期权收到期权费
            option_sign = -1  # 空头期权
        else:
            premium_received = -option_value  # 买入期权支付期权费
            option_sign = 1

        # 初始对冲: 需要持有 delta 份标的来对冲
        # Short call: delta < 0 (对卖方来说)，需要买入 delta 份标的
        # 对冲者视角: 需要 -option_delta 份标的
        initial_hedge = -current_delta if is_short else -current_delta
        # 实际上，如果卖出期权（is_short），期权的 delta 是正的(call)，
        # 为了 delta neutral，需要持有 -delta 份标的（做空）
        # 但这里我们用另一种方式理解:
        # 组合 = 期权头寸 + 标的头寸
        # 要使组合 delta = 0: hedge_position * 1 + option_sign * option_delta = 0
        # hedge_position = -option_sign * option_delta
        hedge = -option_sign * current_delta
        hedge_positions[0] = hedge

        cumulative_cost = 0.0
        # 建立初始对冲的成本
        if abs(hedge) > 0:
            cost = abs(hedge) * S0 * self.transaction_cost
            cumulative_cost += cost

        # 逐步模拟
        for t in range(1, n_steps + 1):
            S_t = prices[t]
            T_remaining = max(self.T - times[t], 0.0)

            # 期权在 t 时刻的价值（用初始隐含波动率 sigma 定价）
            opt_val = BlackScholes.price(S_t, self.K, self.r, self.sigma, T_remaining, self.option_type)
            option_values[t] = opt_val

            # 标的持仓的 PnL 变动
            prev_hedge = hedge_positions[t - 1]
            price_change = S_t - prices[t - 1]
            hedge_pnl = prev_hedge * price_change

            # 期权头寸的 PnL 变动
            option_pnl = option_sign * (opt_val - option_values[t - 1])

            # 是否需要调仓
            if t % self.rebalance_freq == 0 and T_remaining > 0:
                new_delta = GreeksCalculator.delta(S_t, self.K, self.r, self.sigma, T_remaining, self.option_type)
                deltas[t] = new_delta
                new_hedge = -option_sign * new_delta
            else:
                deltas[t] = deltas[t - 1]
                new_hedge = prev_hedge

            # 调仓成本
            trade_size = abs(new_hedge - prev_hedge)
            trade_cost = trade_size * S_t * self.transaction_cost
            cumulative_cost += trade_cost
            rebalance_costs_cum[t] = cumulative_cost

            # 更新持仓
            hedge_positions[t] = new_hedge
            hedge = new_hedge

            # 累计 PnL
            pnl_path[t] = pnl_path[t - 1] + hedge_pnl + option_pnl - trade_cost

        # 到期结算
        T_remaining_final = 0.0
        final_option_value = BlackScholes.price(prices[-1], self.K, self.r, self.sigma, T_remaining_final, self.option_type)

        # 到期时期权赔付
        payoff = option_sign * final_option_value
        pnl_path[-1] += payoff - (option_sign * option_values[-1])

        # 清算剩余标的持仓
        final_hedge = hedge_positions[-1]
        liquidation_pnl = final_hedge * prices[-1]
        pnl_path[-1] += liquidation_pnl

        # 加上期权费
        total_pnl = pnl_path[-1] + premium_received - cumulative_cost

        # Gamma PnL 近似分解: 0.5 * Gamma * (dS)^2 - Theta * dt
        # 这是理论上的对冲 PnL 来源

        return {
            "pnl": total_pnl,
            "pnl_path": pnl_path,
            "hedge_positions": hedge_positions,
            "option_values": option_values,
            "deltas": deltas,
            "rebalance_costs": rebalance_costs_cum,
            "premium_received": premium_received,
        }
