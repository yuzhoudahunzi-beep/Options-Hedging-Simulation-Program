"""
投资组合管理器 — 持仓跟踪、实时 Greeks、Mark-to-Market P&L

管理:
  - 期权头寸（空头，因为用户是做市商/卖方）
  - 标的持仓（用于对冲）
  - 现金账户
  - 实时计算总 Greeks 和 P&L
"""

import numpy as np
from typing import Optional
from dataclasses import dataclass, field
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pricing.black_scholes import BlackScholes
from pricing.greeks import GreeksCalculator


@dataclass
class OptionPosition:
    """单个期权头寸"""
    id: int
    client_name: str        # 客户名
    option_type: str        # "call" | "put"
    K: float               # 行权价
    T_total: float         # 总到期时间（年）
    T_remaining: float     # 剩余到期时间（年）
    qty: int               # 数量（张）
    premium_received: float # 收到的权利金（总额）
    iv_at_trade: float     # 交易时的 IV
    S_at_trade: float      # 交易时的标的价

    @property
    def sign(self) -> int:
        """头寸方向: -1 = 空头（卖方）"""
        return -1


@dataclass
class TradeRecord:
    """交易记录"""
    step: int
    action: str           # "buy_stock" | "sell_stock" | "accept_option" | "close_option"
    detail: str
    amount: float = 0.0  # 金额


class Portfolio:
    """投资组合管理器"""

    CONTRACT_MULTIPLIER = 100  # 每张期权对应100份标的（训练简化版，真实50ETF为10000）

    def __init__(self, initial_cash: float = 100000.0):
        """
        参数:
            initial_cash: 初始现金
        """
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.stock_position = 0     # 标的持仓股数（正=多头，负=空头）
        self.stock_avg_cost = 0.0   # 标的持仓均价
        self.option_positions: list[OptionPosition] = []
        self.trade_log: list[TradeRecord] = []
        self.next_option_id = 1

        # P&L 跟踪
        self.pnl_history: list[dict] = []  # [{step, total_pnl, option_pnl, stock_pnl, cash}]
        self.peak_pnl = 0.0
        self.max_drawdown = 0.0

    # ============ 交易操作 ============

    def accept_option(
        self,
        client_name: str,
        option_type: str,
        K: float,
        T: float,
        qty: int,
        premium_per_unit: float,
        iv: float,
        S: float,
        step: int,
    ) -> OptionPosition:
        """
        接受客户订单（卖出期权，收取权利金）

        参数:
            client_name: 客户名
            option_type: "call" | "put"
            K: 行权价
            T: 到期时间（年）
            qty: 数量（张）
            premium_per_unit: 每张的权利金
            iv: 当前 IV
            S: 当前标的价
            step: 当前步数

        返回:
            创建的 OptionPosition
        """
        total_premium = premium_per_unit * qty
        self.cash += total_premium

        pos = OptionPosition(
            id=self.next_option_id,
            client_name=client_name,
            option_type=option_type,
            K=K,
            T_total=T,
            T_remaining=T,
            qty=qty,
            premium_received=total_premium,
            iv_at_trade=iv,
            S_at_trade=S,
        )
        self.next_option_id += 1
        self.option_positions.append(pos)

        self.trade_log.append(TradeRecord(
            step=step,
            action="accept_option",
            detail=f"卖出{qty}张 {option_type.upper()} K={K:.3f} T={T:.3f} 给{client_name}",
            amount=total_premium,
        ))

        return pos

    def buy_stock(self, qty: int, price: float, step: int, cost_rate: float = 0.0003):
        """
        买入标的

        参数:
            qty: 买入数量（股）
            price: 买入价格
            step: 当前步数
            cost_rate: 交易成本率
        """
        cost = qty * price * (1 + cost_rate)
        if cost > self.cash:
            return False  # 资金不足

        # 更新均价
        total_shares = self.stock_position + qty
        if total_shares > 0:
            self.stock_avg_cost = (
                (self.stock_position * self.stock_avg_cost + qty * price) / total_shares
            )
        self.stock_position = total_shares
        self.cash -= cost

        self.trade_log.append(TradeRecord(
            step=step,
            action="buy_stock",
            detail=f"买入{qty}股 @ {price:.4f}",
            amount=-cost,
        ))
        return True

    def sell_stock(self, qty: int, price: float, step: int, cost_rate: float = 0.0003):
        """卖出标的"""
        if qty > self.stock_position:
            qty = self.stock_position  # 最多卖到0
        if qty <= 0:
            return False

        proceeds = qty * price * (1 - cost_rate)
        self.stock_position -= qty
        self.cash += proceeds

        self.trade_log.append(TradeRecord(
            step=step,
            action="sell_stock",
            detail=f"卖出{qty}股 @ {price:.4f}",
            amount=proceeds,
        ))
        return True

    def close_option(self, option_id: int, current_price: float, step: int) -> bool:
        """
        平仓某个期权头寸（买回期权）

        参数:
            option_id: 期权头寸 ID
            current_price: 当前期权单价
            step: 当前步数
        """
        pos = None
        for p in self.option_positions:
            if p.id == option_id:
                pos = p
                break
        if pos is None:
            return False

        # 买回期权的成本
        close_cost = current_price * pos.qty
        self.cash -= close_cost

        self.trade_log.append(TradeRecord(
            step=step,
            action="close_option",
            detail=f"平仓期权#{option_id} {pos.client_name} @ {current_price:.4f}",
            amount=-close_cost,
        ))

        self.option_positions.remove(pos)
        return True

    # ============ 实时计算 ============

    def get_option_mtm(self, pos: OptionPosition, S: float, iv: float) -> dict:
        """计算单个期权头寸的 MTM"""
        T = max(pos.T_remaining, 0.0001)
        mtm_price = BlackScholes.price(S, pos.K, 0.03, iv, T, pos.option_type)
        total_mtm = mtm_price * pos.qty

        # 卖方 P&L = 收到的权利金 - 当前市值
        pnl = pos.premium_received - total_mtm

        # Greeks（空头，所以取反）
        greeks = GreeksCalculator.all_greeks(S, pos.K, 0.03, iv, T, pos.option_type)

        return {
            "id": pos.id,
            "client_name": pos.client_name,
            "option_type": pos.option_type,
            "K": pos.K,
            "T_remaining": round(pos.T_remaining, 4),
            "T_days": round(pos.T_remaining * 252),
            "qty": pos.qty,
            "premium_received": round(pos.premium_received, 2),
            "mtm_price": round(mtm_price, 4),
            "mtm_total": round(total_mtm, 2),
            "pnl": round(pnl, 2),
            "delta": round(-greeks["delta"] * pos.qty * self.CONTRACT_MULTIPLIER, 2),
            "gamma": round(-greeks["gamma"] * pos.qty * self.CONTRACT_MULTIPLIER, 4),
            "vega": round(-greeks["vega"] * pos.qty * self.CONTRACT_MULTIPLIER, 4),
            "theta": round(-greeks["theta"] * pos.qty * self.CONTRACT_MULTIPLIER / 252, 4),
        }

    def compute_greeks(self, S: float, iv: float) -> dict:
        """计算组合总 Greeks"""
        total_delta = 0.0
        total_gamma = 0.0
        total_vega = 0.0
        total_theta = 0.0

        for pos in self.option_positions:
            T = max(pos.T_remaining, 0.0001)
            g = GreeksCalculator.all_greeks(S, pos.K, 0.03, iv, T, pos.option_type)
            # 空头取反，乘以合约乘数
            total_delta += -g["delta"] * pos.qty * self.CONTRACT_MULTIPLIER
            total_gamma += -g["gamma"] * pos.qty * self.CONTRACT_MULTIPLIER
            total_vega += -g["vega"] * pos.qty * self.CONTRACT_MULTIPLIER
            total_theta += -g["theta"] * pos.qty * self.CONTRACT_MULTIPLIER / 252

        # 加上标的的 Delta
        stock_delta = self.stock_position

        return {
            "total_delta": round(total_delta + stock_delta, 2),
            "option_delta": round(total_delta, 2),
            "stock_delta": stock_delta,
            "total_gamma": round(total_gamma, 4),
            "total_vega": round(total_vega, 4),
            "total_theta": round(total_theta, 4),  # 每日 theta
            "net_exposure": round(abs(total_delta + stock_delta), 2),
        }

    def compute_pnl(self, S: float, iv: float) -> dict:
        """计算总 P&L（Mark-to-Market）"""
        # 期权部分 P&L
        option_pnl = 0.0
        option_details = []
        for pos in self.option_positions:
            mtm = self.get_option_mtm(pos, S, iv)
            option_pnl += mtm["pnl"]
            option_details.append(mtm)

        # 标的部分 P&L
        stock_pnl = 0.0
        if self.stock_position != 0 and self.stock_avg_cost > 0:
            stock_pnl = self.stock_position * (S - self.stock_avg_cost)

        # 总 P&L = 期权P&L + 标的P&L
        total_pnl = option_pnl + stock_pnl

        # 更新峰值和回撤
        self.peak_pnl = max(self.peak_pnl, total_pnl)
        self.max_drawdown = min(self.max_drawdown, total_pnl - self.peak_pnl)

        return {
            "total_pnl": round(total_pnl, 2),
            "option_pnl": round(option_pnl, 2),
            "stock_pnl": round(stock_pnl, 2),
            "cash": round(self.cash, 2),
            "peak_pnl": round(self.peak_pnl, 2),
            "max_drawdown": round(self.max_drawdown, 2),
            "option_details": option_details,
        }

    def update_time(self, dt: float):
        """更新时间（每 tick 调用）"""
        for pos in self.option_positions:
            pos.T_remaining = max(pos.T_remaining - dt, 0.0)

    def settle(self, S: float) -> dict:
        """
        到期结算

        参数:
            S: 到期时标的价格

        返回:
            结算结果
        """
        total_payoff = 0.0

        for pos in self.option_positions:
            if pos.option_type == "call":
                intrinsic = max(S - pos.K, 0.0)
            else:
                intrinsic = max(pos.K - S, 0.0)

            # 卖方需要支付 intrinsic
            payoff = -intrinsic * pos.qty * self.CONTRACT_MULTIPLIER
            total_payoff += payoff

        self.cash += total_payoff

        # 清算标的持仓
        stock_liquidation = self.stock_position * S
        self.cash += stock_liquidation
        self.stock_position = 0

        final_pnl = self.cash - self.initial_cash

        self.option_positions = []

        return {
            "final_pnl": round(final_pnl, 2),
            "settlement_payoff": round(total_payoff, 2),
            "stock_liquidation": round(stock_liquidation, 2),
            "final_cash": round(self.cash, 2),
            "initial_cash": self.initial_cash,
        }

    def get_full_state(self, S: float, iv: float, step: int) -> dict:
        """获取完整的组合状态快照"""
        pnl = self.compute_pnl(S, iv)
        greeks = self.compute_greeks(S, iv)

        # 记录 P&L 历史
        self.pnl_history.append({
            "step": step,
            "total_pnl": pnl["total_pnl"],
            "option_pnl": pnl["option_pnl"],
            "stock_pnl": pnl["stock_pnl"],
        })

        return {
            **pnl,
            **greeks,
            "stock_position": self.stock_position,
            "stock_avg_cost": round(self.stock_avg_cost, 4),
            "n_options": len(self.option_positions),
            "trade_count": len(self.trade_log),
        }

    def get_delta_target(self, S: float, iv: float) -> int:
        """计算 Delta Neutral 需要的标的持仓"""
        greeks = self.compute_greeks(S, iv)
        # 需要 stock_delta = -option_delta 来对冲
        target = int(round(-greeks["option_delta"]))
        return target
