"""
游戏引擎 — 核心 asyncio 事件循环

状态机: SETUP → TRADING → SETTLED → REVIEW
每 tick: 推进市场 → 检查客户 → 更新 P&L → 广播事件
"""

import asyncio
import json
import time
import numpy as np
from typing import Optional
from enum import Enum

from server.market_engine import MarketEngine
from server.portfolio import Portfolio
from server.client_generator import ClientGenerator
from server.score_engine import ScoreEngine


class GameState(str, Enum):
    SETUP = "setup"
    TRADING = "trading"
    PAUSED = "paused"
    SETTLED = "settled"
    REVIEW = "review"


class GameEngine:
    """游戏引擎"""

    def __init__(self):
        self.state = GameState.SETUP
        self.market: Optional[MarketEngine] = None
        self.portfolio: Optional[Portfolio] = None
        self.client_gen: Optional[ClientGenerator] = None
        self.score_engine: Optional[ScoreEngine] = None

        # 游戏参数
        self.total_days = 60          # 总交易日
        self.tick_dt = 1.0 / 252      # 每 tick = 1 交易日
        self.speed = 1.0              # 速度倍率
        self.difficulty = "medium"

        # 运行时
        self.tick_count = 0
        self.game_day = 0
        self.pending_order: Optional[dict] = None  # 待处理的客户订单
        self.total_premium_earned = 0.0
        self.n_option_trades = 0

        # WebSocket 连接池
        self.connections: list = []

        # 游戏循环控制
        self._running = False
        self._task: Optional[asyncio.Task] = None

        # 价格历史（给前端画图）
        self.price_history: list[dict] = []
        self.pnl_history: list[dict] = []

    # ============ 游戏配置 ============

    def setup_game(
        self,
        underlying: str = "50ETF",
        difficulty: str = "medium",
        mode: str = "simulate",
        model: str = "heston",
        total_days: int = 60,
        initial_cash: float = 100000.0,
        speed: float = 1.0,
        seed: Optional[int] = None,
    ):
        """初始化游戏"""
        self.difficulty = difficulty
        self.total_days = total_days
        self.tick_dt = 1.0 / 252
        self.speed = speed

        # 难度参数
        diff_params = {
            "easy":   {"arrival": 0.10, "S0": None},
            "medium": {"arrival": 0.18, "S0": None},
            "hard":   {"arrival": 0.25, "S0": None},
        }
        dp = diff_params.get(difficulty, diff_params["medium"])

        self.market = MarketEngine(
            underlying=underlying,
            mode=mode,
            model=model,
            tick_dt=self.tick_dt,
            difficulty=difficulty,
            seed=seed,
        )

        self.portfolio = Portfolio(initial_cash=initial_cash)
        self.client_gen = ClientGenerator(
            arrival_rate=dp["arrival"],
            difficulty=difficulty,
            S0=self.market.S0,
        )
        self.score_engine = ScoreEngine()

        self.tick_count = 0
        self.game_day = 0
        self.pending_order = None
        self.total_premium_earned = 0.0
        self.n_option_trades = 0
        self.price_history = []
        self.pnl_history = []

        self.state = GameState.SETUP

    async def start_game(self):
        """开始游戏"""
        if self.state != GameState.SETUP:
            print(f"[GameEngine] ⚠️ start_game called but state is {self.state}, not SETUP")
            return
        self.state = GameState.TRADING
        self._running = True
        self._task = asyncio.create_task(self._game_loop())
        print(f"[GameEngine] 🚀 游戏开始! total_days={self.total_days}, speed={self.speed}")

    async def pause_game(self):
        """暂停"""
        if self.state == GameState.TRADING:
            self.state = GameState.PAUSED
            self._running = False

    async def resume_game(self):
        """继续"""
        if self.state == GameState.PAUSED:
            self.state = GameState.TRADING
            self._running = True
            self._task = asyncio.create_task(self._game_loop())

    async def stop_game(self):
        """停止"""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    # ============ 核心循环 ============

    async def _game_loop(self):
        """主游戏循环"""
        try:
            print(f"[GameEngine] 🔄 游戏循环启动, state={self.state.value}, running={self._running}")
            tick_num = 0
            while self._running and self.state == GameState.TRADING:
                await self._tick()
                tick_num += 1
                if tick_num <= 3 or tick_num % 10 == 0:
                    print(f"[GameEngine] tick #{tick_num}, day={self.game_day}, "
                          f"connections={len(self.connections)}, S={self.market.S:.4f}")

                # 检查是否到期
                if self.game_day >= self.total_days:
                    await self._settle()
                    break

                # 等待（考虑速度）
                wait_time = 1.0 / self.speed
                await asyncio.sleep(wait_time)
            print(f"[GameEngine] 游戏循环退出, state={self.state.value}, running={self._running}")
        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"[GameEngine] 循环错误: {e}")
            import traceback
            traceback.print_exc()

    async def _tick(self):
        """单 tick 处理"""
        if not self.market or not self.portfolio:
            return

        # 1. 推进市场
        market_update = self.market.step()
        S = market_update["S"]
        IV = market_update["IV"]

        # 2. 更新时间
        self.portfolio.update_time(self.tick_dt)
        self.tick_count += 1
        self.game_day = self.tick_count  # 1 tick = 1 交易日

        # 3. 检查客户到达
        new_order = None
        if self.pending_order is None:
            T_market = max((self.total_days - self.game_day) / 252, 1 / 252)
            new_order = self.client_gen.check_arrival(S, IV, T_market)
            if new_order is not None:
                self.pending_order = new_order
                print(f"[GameEngine] 🟢 客户到来: {new_order['client_name']} "
                      f"{new_order['qty']}张 {new_order['option_type']} K={new_order['K']:.3f}")

        # 4. 更新 P&L
        portfolio_state = self.portfolio.get_full_state(S, IV, self.tick_count)

        # 5. 评分引擎记录
        if self.score_engine:
            self.score_engine.record_tick(
                net_delta=portfolio_state["total_delta"],
                total_pnl=portfolio_state["total_pnl"],
            )

        # 6. 记录历史
        self.price_history.append({
            "step": self.tick_count,
            "S": S,
            "IV": IV,
        })
        self.pnl_history.append({
            "step": self.tick_count,
            "total_pnl": portfolio_state["total_pnl"],
        })

        # 7. 构建广播消息
        broadcast = {
            "type": "tick",
            "market": market_update,
            "portfolio": portfolio_state,
            "game": {
                "state": self.state.value,
                "day": self.game_day,
                "total_days": self.total_days,
                "tick_count": self.tick_count,
            },
            "total_premium_earned": round(self.total_premium_earned, 2),
        }

        # 如果有新客户
        if new_order is not None:
            broadcast["new_order"] = new_order

        # 如果有市场事件
        if market_update["event"] != "none":
            broadcast["market_event"] = market_update["event"]

        # 8. 广播给所有 WebSocket 连接
        await self._broadcast(broadcast)

    async def _settle(self):
        """到期结算"""
        if not self.market or not self.portfolio:
            return

        S = self.market.S
        settlement = self.portfolio.settle(S)

        # 评分
        score = self.score_engine.compute_score(
            final_pnl=settlement["final_pnl"],
            initial_cash=self.portfolio.initial_cash,
            n_option_trades=self.n_option_trades,
            total_premium_earned=self.total_premium_earned,
        )

        self.state = GameState.SETTLED

        await self._broadcast({
            "type": "settled",
            "settlement": settlement,
            "score": score,
        })

    # ============ 用户操作 ============

    async def handle_action(self, action: dict) -> dict:
        """处理用户操作"""
        act_type = action.get("type")

        if act_type == "start":
            self.setup_game(
                underlying=action.get("underlying", "50ETF"),
                difficulty=action.get("difficulty", "medium"),
                mode=action.get("mode", "simulate"),
                model=action.get("model", "heston"),
                total_days=action.get("total_days", 60),
                initial_cash=action.get("initial_cash", 100000),
                speed=action.get("speed", 1.0),
                seed=action.get("seed"),
            )
            await self.start_game()
            return {"status": "ok", "message": "游戏开始！"}

        elif act_type == "pause":
            await self.pause_game()
            return {"status": "ok", "message": "已暂停"}

        elif act_type == "resume":
            await self.resume_game()
            return {"status": "ok", "message": "继续"}

        elif act_type == "accept_order":
            return await self._accept_order()

        elif act_type == "reject_order":
            self.pending_order = None
            await self._broadcast({"type": "info", "message": "已拒绝客户订单"})
            # 处理完订单，自动恢复游戏
            if self.state == GameState.PAUSED:
                self.state = GameState.TRADING
                self._running = True
                self._task = asyncio.create_task(self._game_loop())
            return {"status": "ok", "message": "已拒绝"}

        elif act_type == "buy_stock":
            qty = int(action.get("qty", 0))
            if qty <= 0 or not self.market:
                return {"status": "error", "message": "数量无效"}
            success = self.portfolio.buy_stock(qty, self.market.S, self.tick_count)
            if success:
                self.score_engine.record_action(is_hedge=True)
                await self._broadcast({"type": "info", "message": f"买入 {qty} 股 @ {self.market.S:.4f}"})
                return {"status": "ok"}
            else:
                return {"status": "error", "message": "资金不足"}

        elif act_type == "sell_stock":
            qty = int(action.get("qty", 0))
            if qty <= 0 or not self.market:
                return {"status": "error", "message": "数量无效"}
            success = self.portfolio.sell_stock(qty, self.market.S, self.tick_count)
            if success:
                self.score_engine.record_action(is_hedge=True)
                await self._broadcast({"type": "info", "message": f"卖出 {qty} 股 @ {self.market.S:.4f}"})
                return {"status": "ok"}
            else:
                return {"status": "error", "message": "无可卖标的"}

        elif act_type == "delta_hedge":
            return await self._delta_hedge()

        elif act_type == "close_option":
            option_id = int(action.get("option_id", 0))
            if not self.market:
                return {"status": "error", "message": "市场未启动"}
            # 计算当前期权市场价
            from pricing.black_scholes import BlackScholes
            pos = None
            for p in self.portfolio.option_positions:
                if p.id == option_id:
                    pos = p
                    break
            if pos is None:
                return {"status": "error", "message": "找不到该期权头寸"}
            T = max(pos.T_remaining, 0.0001)
            opt_price = BlackScholes.price(
                self.market.S, pos.K, 0.03, self.market.iv, T, pos.option_type
            )
            success = self.portfolio.close_option(option_id, opt_price, self.tick_count)
            if success:
                self.score_engine.record_action()
                await self._broadcast({"type": "info", "message": f"已平仓期权#{option_id}"})
                return {"status": "ok"}
            else:
                return {"status": "error", "message": "平仓失败"}

        elif act_type == "restart":
            await self.stop_game()
            self.state = GameState.SETUP
            return {"status": "ok", "message": "可以重新开始"}

        elif act_type == "get_hint":
            hint = self.get_hint()
            return {"status": "ok", **hint}

        else:
            return {"status": "error", "message": f"未知操作: {act_type}"}

    async def _accept_order(self) -> dict:
        """接受当前待处理的客户订单"""
        if self.pending_order is None:
            return {"status": "error", "message": "没有待处理订单"}

        order = self.pending_order
        pos = self.portfolio.accept_option(
            client_name=order["client_name"],
            option_type=order["option_type"],
            K=order["K"],
            T=order["T"],
            qty=order["qty"],
            premium_per_unit=order["bid_premium"],
            iv=order["IV_at_request"],
            S=order["S_at_request"],
            step=self.tick_count,
        )

        self.total_premium_earned += order["total_premium"]
        self.n_option_trades += 1
        self.pending_order = None
        self.score_engine.record_action()

        await self._broadcast({
            "type": "info",
            "message": f"✅ 接受 {order['client_name']} 订单: "
                       f"卖出{order['qty']}张 {order['option_type'].upper()} "
                       f"K={order['K']:.3f} 收权利金 ¥{order['total_premium']:.0f}",
        })

        # 处理完订单，自动恢复游戏
        if self.state == GameState.PAUSED:
            self.state = GameState.TRADING
            self._running = True
            self._task = asyncio.create_task(self._game_loop())

        return {"status": "ok", "premium": order["total_premium"]}

    async def _delta_hedge(self) -> dict:
        """一键 Delta 对冲"""
        if not self.market:
            return {"status": "error", "message": "市场未启动"}

        target = self.portfolio.get_delta_target(self.market.S, self.market.iv)
        current = self.portfolio.stock_position
        diff = target - current

        if abs(diff) < 10:
            return {"status": "ok", "message": "Delta 已接近中性，无需调仓"}

        if diff > 0:
            success = self.portfolio.buy_stock(diff, self.market.S, self.tick_count)
            action_str = f"买入 {diff} 股"
        else:
            success = self.portfolio.sell_stock(abs(diff), self.market.S, self.tick_count)
            action_str = f"卖出 {abs(diff)} 股"

        if success:
            self.score_engine.record_action(is_hedge=True)
            await self._broadcast({
                "type": "info",
                "message": f"🔄 Delta对冲: {action_str} → 目标持仓 {target} 股",
            })
            return {"status": "ok", "target": target, "traded": diff}
        else:
            return {"status": "error", "message": "对冲失败（资金不足或无可卖）"}

    # ============ 提示系统 ============

    def get_hint(self) -> dict:
        """
        根据当前状态生成操作提示 + 原理讲解

        返回:
            {
                "suggestion": "建议操作（一句话）",
                "reason": "为什么（原理解释）",
                "actions": ["具体操作列表"],
                "risk_level": "low" | "medium" | "high",
            }
        """
        if not self.market or not self.portfolio:
            return {
                "suggestion": "游戏还未开始",
                "reason": '点击「开始交易」按钮后，提示系统会实时分析你的持仓和市场状态。',
                "actions": [],
                "risk_level": "low",
            }

        S = self.market.S
        IV = self.market.iv
        pf_state = self.portfolio.get_full_state(S, IV, self.tick_count)

        total_delta = pf_state["total_delta"]
        option_delta = pf_state["option_delta"]
        stock_delta = pf_state["stock_delta"]
        net_exposure = pf_state["net_exposure"]
        total_pnl = pf_state["total_pnl"]
        n_options = pf_state["n_options"]

        suggestions = []
        reasons = []
        actions = []
        risk_level = "low"

        # === 1. 分析 Delta 暴露 ===
        abs_delta = abs(total_delta)

        # 相对于合约规模判断 Delta 风险
        scale = max(n_options * 100 * self.portfolio.CONTRACT_MULTIPLIER, 1000)
        delta_ratio = abs_delta / scale

        if delta_ratio > 0.5:
            risk_level = "high"
            direction = "正" if total_delta > 0 else "负"
            suggestions.append(f"⚠️ Delta 暴露过大！当前净 Delta = {total_delta:.0f}（{direction}方向）")
            reasons.append(
                "Delta 衡量的是你的组合对标的价格变动的敏感度。"
                f"Delta = {total_delta:.0f} 意味着标的每涨 1 元，你的组合{'赚' if total_delta > 0 else '亏'}约 {total_delta:.0f} 元。"
                "作为期权卖方，你的目标是 Delta Neutral（Delta ≈ 0），这样无论标涨跌，你都能稳稳赚取权利金。"
            )
            target = self.portfolio.get_delta_target(S, IV)
            diff = target - stock_delta
            if diff > 0:
                actions.append(f"买入约 {diff} 股标的来对冲（减少正 Delta）")
            elif diff < 0:
                actions.append(f"卖出约 {abs(diff)} 股标的来对冲（减少负 Delta）")
            actions.append('或者点击「🔄 一键 Delta 对冲」按钮自动调仓')

        elif delta_ratio > 0.2:
            risk_level = "medium"
            suggestions.append(f"📊 Delta 暴露中等（净 Delta = {total_delta:.0f}），建议适当调仓")
            reasons.append(
                "当前 Delta 偏离中性不算太大，但标的一个中等幅度的波动就可能造成明显损益。"
                "做市商的核心原则是保持 Delta 接近零，让时间价值衰减（Theta）成为你的利润来源。"
            )
            target = self.portfolio.get_delta_target(S, IV)
            diff = target - stock_delta
            if abs(diff) > 50:
                if diff > 0:
                    actions.append(f"买入约 {diff} 股标的")
                else:
                    actions.append(f"卖出约 {abs(diff)} 股标的")

        else:
            suggestions.append("✅ Delta 控制良好，接近中性！")
            reasons.append(
                "Delta 接近零意味着你的组合对标的价格变动不敏感。"
                "此时你主要靠 Theta（时间价值衰减）赚钱——每过一天，期权价值减少，作为卖方你就赚到了。"
            )

        # === 2. 分析待处理订单 ===
        if self.pending_order:
            order = self.pending_order
            suggestions.append(f"🟢 有客户订单待处理：{order['client_name']} 想买 {order['qty']}张 {order['option_type'].upper()}")
            bs_vs_bid = order['bid_premium'] / order['bs_price'] if order['bs_price'] > 0 else 1
            if bs_vs_bid >= 1.05:
                suggestions.append(f"💰 客户出价 {order['bid_premium']:.2f} 高于 BS 理论价 {order['bs_price']:.2f}，溢价 {((bs_vs_bid-1)*100):.0f}%，建议接受！")
                reasons.append(
                    "当客户愿意支付高于理论价的权利金时，你有了额外的安全垫。"
                    "即使后续对冲成本略高，多收的权利金也能覆盖。这是做市商最喜欢的交易。"
                )
                actions.append('点击「✅ 接受订单」')
            elif bs_vs_bid >= 0.95:
                suggestions.append(f"📊 客户出价 {order['bid_premium']:.2f} 接近 BS 理论价 {order['bs_price']:.2f}，可以考虑接受")
                reasons.append(
                    "出价在理论价附近是公平交易。接受后记得立刻做 Delta 对冲。"
                )
                actions.append("接受订单后立即做 Delta 对冲")
            else:
                suggestions.append(f"⚠️ 客户出价 {order['bid_premium']:.2f} 低于 BS 理论价 {order['bs_price']:.2f}，不太划算")
                reasons.append(
                    "出价低于理论价意味着你卖便宜了。除非你急需交易活跃度，否则可以拒绝。"
                )
                actions.append("考虑拒绝这个订单")

            # IV 分析
            if IV > 0.35:
                reasons.append(
                    f"当前 IV = {IV*100:.1f}% 较高。高 IV 时卖出期权能收到更多权利金（好消息），"
                    "但也意味着市场预期未来波动很大（坏消息）。接受后要更积极地做 Delta 对冲。"
                )

        # === 3. 分析市场状态 ===
        if IV > 0.40:
            suggestions.append(f"⚡ 当前 IV = {IV*100:.1f}%，波动率较高，期权定价偏高")
            reasons.append(
                "高 IV 对卖方来说既是机会（权利金多）也是风险（标的可能大幅波动）。"
                "此时对冲要更频繁，因为标的一个大跳动就可能突破你的对冲范围。"
            )
        elif IV < 0.12:
            suggestions.append(f"📉 当前 IV = {IV*100:.1f}%，波动率较低")
            reasons.append(
                "低 IV 意味着期权便宜，卖期权收不到太多权利金。"
                "但好处是市场平静，对冲压力小。适合新手练习 Delta 对冲。"
            )

        # === 4. 分析持仓 ===
        if n_options == 0 and self.game_day > 5:
            suggestions.append("🤔 还没有接受任何期权订单。主动接受客户订单才能赚到权利金！")
            reasons.append(
                "作为做市商，你的利润来源是：收权利金 → Delta 对冲 → 等待时间衰减。"
                "没有期权头寸就没有 Theta 收入。当客户来时，合理定价的订单建议接受。"
            )
            actions.append("等待下一个客户到来，考虑接受订单")

        # 检查即将到期的期权
        for pos in self.portfolio.option_positions:
            if pos.T_remaining < 3 / 252 and pos.T_remaining > 0:
                suggestions.append(f"⏰ 期权 #{pos.id} 即将到期（剩 {max(round(pos.T_remaining * 252), 1)} 天）")
                reasons.append(
                    "临近到期的期权 Gamma 风险很大（Gamma 风险 = 标的价格稍微变动，Delta 就剧烈变化）。"
                    "如果不想到期时被行权，可以考虑提前平仓。"
                    "如果 Delta 已经对冲好了，也可以持有到期赚取最后的 Theta。"
                )
                actions.append(f"考虑平仓期权 #{pos.id}，或确保 Delta 对冲到位")

        # 如果没有特别建议
        if not suggestions:
            suggestions.append("📊 当前状态正常，继续监控市场和持仓")
            reasons.append("做市商的核心就是持续监控 Delta、监控市场，及时调整对冲。")

        # 如果没有操作建议
        if not actions:
            actions.append('点击「🔄 一键 Delta 对冲」保持 Delta 中性')
            actions.append("等待新客户到来")

        # 最终建议（取第一条）
        main_suggestion = suggestions[0]
        main_reason = reasons[0] if reasons else ""

        # 如果有多条，合并
        extra = ""
        if len(suggestions) > 1:
            extra = "\n\n此外：\n" + "\n".join(f"• {s}" for s in suggestions[1:])

        return {
            "suggestion": main_suggestion + extra,
            "reason": main_reason,
            "actions": actions,
            "risk_level": risk_level,
        }

    # ============ WebSocket 管理 ============

    async def connect(self, websocket):
        """新的 WebSocket 连接"""
        await websocket.accept()
        self.connections.append(websocket)

        # 发送当前状态
        await self._send_state(websocket)

    def disconnect(self, websocket):
        """断开连接"""
        if websocket in self.connections:
            self.connections.remove(websocket)

    async def _broadcast(self, message: dict):
        """广播消息给所有连接"""
        dead = []
        for ws in self.connections:
            try:
                await ws.send_json(message)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)

    async def _send_state(self, websocket):
        """发送当前完整状态给新连接"""
        state = {
            "type": "state_sync",
            "game_state": self.state.value,
            "game": {
                "day": self.game_day,
                "total_days": self.total_days,
                "difficulty": self.difficulty,
                "tick_count": self.tick_count,
            },
            "price_history": self.price_history[-200:],  # 最近200步
            "pnl_history": self.pnl_history[-200:],
        }

        if self.market:
            state["market"] = self.market.get_state()

        if self.portfolio and self.market:
            state["portfolio"] = self.portfolio.get_full_state(
                self.market.S, self.market.iv, self.tick_count
            )

        if self.pending_order:
            state["pending_order"] = self.pending_order

        if self.state == GameState.SETTLED and self.score_engine:
            state["score"] = self.score_engine.compute_score(
                final_pnl=self.portfolio.cash - self.portfolio.initial_cash if self.portfolio else 0,
                initial_cash=self.portfolio.initial_cash if self.portfolio else 100000,
                n_option_trades=self.n_option_trades,
                total_premium_earned=self.total_premium_earned,
            )

        try:
            await websocket.send_json(state)
        except Exception:
            pass
