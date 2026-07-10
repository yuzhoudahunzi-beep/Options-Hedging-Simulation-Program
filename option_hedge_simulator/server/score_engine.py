"""
评分引擎 — 评估交易员对冲表现

对比用户操作与理论最优（Delta Neutral），给出评级和反馈。
"""

import numpy as np
from typing import Optional


class ScoreEngine:
    """评分引擎"""

    def __init__(self):
        # 每 tick 记录
        self.delta_history: list[float] = []   # 净 Delta 暴露
        self.pnl_history: list[float] = []     # P&L 路径
        self.action_count = 0                  # 用户操作次数
        self.hedge_actions = 0                 # 对冲操作次数

    def record_tick(self, net_delta: float, total_pnl: float):
        """记录每个 tick 的状态"""
        self.delta_history.append(abs(net_delta))
        self.pnl_history.append(total_pnl)

    def record_action(self, is_hedge: bool = False):
        """记录用户操作"""
        self.action_count += 1
        if is_hedge:
            self.hedge_actions += 1

    def compute_score(
        self,
        final_pnl: float,
        initial_cash: float,
        n_option_trades: int,
        total_premium_earned: float,
    ) -> dict:
        """
        计算最终评分

        参数:
            final_pnl: 最终 P&L
            initial_cash: 初始资金
            n_option_trades: 期权交易笔数
            total_premium_earned: 总权利金收入

        返回:
            评分结果 dict
        """
        if not self.pnl_history:
            return self._empty_score()

        # === 1. 收益评分 (40分) ===
        # 基于 P&L / 总权利金 的比例
        if total_premium_earned > 0:
            pnl_ratio = final_pnl / total_premium_earned
        else:
            pnl_ratio = 0.0

        # 收益占比映射: >80% → 满分，<0% → 0分
        profit_score = np.clip(pnl_ratio, 0, 1.0) * 40

        # === 2. Delta 控制评分 (30分) ===
        # 平均 |net delta| 越小越好
        if self.delta_history:
            avg_delta = np.mean(self.delta_history)
            median_delta = np.median(self.delta_history)
        else:
            avg_delta = 0
            median_delta = 0

        # 平均 delta 暴露映射: 0 → 满分，很大 → 0分
        # 使用相对尺度：相对于交易过的期权数量
        scale = max(n_option_trades * 5000, 10000)
        delta_score = np.clip(1 - avg_delta / scale, 0, 1) * 30

        # === 3. 回撤控制 (15分) ===
        if self.pnl_history:
            pnl_arr = np.array(self.pnl_history)
            peak = np.maximum.accumulate(pnl_arr)
            drawdown = pnl_arr - peak
            max_dd = np.min(drawdown) if len(drawdown) > 0 else 0
            max_dd_abs = abs(max_dd)
        else:
            max_dd_abs = 0

        # 最大回撤映射: 回撤小于权利金10% → 满分
        dd_scale = max(total_premium_earned * 0.3, 5000)
        drawdown_score = np.clip(1 - max_dd_abs / dd_scale, 0, 1) * 15

        # === 4. 操作活跃度 (15分) ===
        # 对冲操作次数适中最好
        if n_option_trades > 0:
            hedge_ratio = self.hedge_actions / max(n_option_trades, 1)
        else:
            hedge_ratio = 0

        # 每笔期权交易平均对冲1-3次最佳
        if 0.5 <= hedge_ratio <= 5:
            activity_score = 15
        elif hedge_ratio < 0.5:
            activity_score = hedge_ratio * 2 * 15
        else:
            activity_score = max(15 - (hedge_ratio - 5) * 2, 5)

        # === 总分 ===
        total_score = profit_score + delta_score + drawdown_score + activity_score
        total_score = round(total_score, 1)

        # 评级
        if total_score >= 90:
            grade = "S"
            comment = "完美对冲！你对 Delta、Gamma、Vega 的理解已经非常深入。"
        elif total_score >= 80:
            grade = "A"
            comment = "优秀！你的对冲策略很稳健，继续保持。"
        elif total_score >= 65:
            grade = "B"
            comment = "不错！基本掌握了对冲要领，注意控制 Delta 暴露。"
        elif total_score >= 45:
            grade = "C"
            comment = "及格。建议更积极地调仓，减少 Delta 裸露。"
        elif total_score >= 25:
            grade = "D"
            comment = "需要加强练习。注意：卖出期权后必须及时 Delta 对冲。"
        else:
            grade = "F"
            comment = "对冲严重不足。建议先理解 Delta Neutral 的原理再实操。"

        # 具体建议
        suggestions = []
        if avg_delta > scale * 0.5:
            suggestions.append("Delta 暴露过大 — 每次标的价格变动后应尽快调仓")
        if max_dd_abs > total_premium_earned * 0.5:
            suggestions.append("回撤过大 — 考虑更频繁对冲或使用 Delta-Gamma 对冲")
        if self.hedge_actions < n_option_trades * 0.5:
            suggestions.append("对冲操作太少 — 收到期权订单后应立即建立对冲头寸")
        if final_pnl < 0:
            suggestions.append("最终亏损 — 检查是否波动率大幅变化时没有及时调整")
        if not suggestions:
            suggestions.append("表现均衡，继续保持!")

        return {
            "total_score": total_score,
            "grade": grade,
            "comment": comment,
            "suggestions": suggestions,
            "breakdown": {
                "profit_score": round(profit_score, 1),
                "delta_score": round(delta_score, 1),
                "drawdown_score": round(drawdown_score, 1),
                "activity_score": round(activity_score, 1),
            },
            "stats": {
                "final_pnl": round(final_pnl, 2),
                "total_premium": round(total_premium_earned, 2),
                "pnl_ratio": round(pnl_ratio, 4),
                "avg_abs_delta": round(avg_delta, 2),
                "max_drawdown": round(max_dd_abs, 2),
                "total_actions": self.action_count,
                "hedge_actions": self.hedge_actions,
                "n_option_trades": n_option_trades,
            },
            "pnl_path": [round(p, 2) for p in self.pnl_history],
        }

    def _empty_score(self) -> dict:
        """无交易时的评分"""
        return {
            "total_score": 0,
            "grade": "F",
            "comment": "没有任何交易活动。",
            "suggestions": ["尝试接受客户订单并进行对冲操作"],
            "breakdown": {"profit_score": 0, "delta_score": 0, "drawdown_score": 0, "activity_score": 0},
            "stats": {},
            "pnl_path": [],
        }

    def reset(self):
        """重置评分引擎"""
        self.delta_history = []
        self.pnl_history = []
        self.action_count = 0
        self.hedge_actions = 0
