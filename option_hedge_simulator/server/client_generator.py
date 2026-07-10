"""
客户订单生成器 — 模拟客户随机到来买期权

泊松过程控制到达频率，每个客户的参数随机生成。
"""

import numpy as np
from typing import Optional


# 客户名池
CLIENT_NAMES = [
    "张总", "李经理", "王老板", "赵姐", "刘总",
    "陈总", "杨总", "黄老板", "周总", "吴经理",
    "徐总", "孙老板", "马经理", "朱总", "胡老板",
    "郭总", "林经理", "何总", "罗老板", "梁总",
    "机构A-套利部", "机构B-对冲基金", "私募C-量化部",
    "保险D-固收部", "券商E-自营部",
]


class ClientGenerator:
    """客户订单生成器"""

    def __init__(
        self,
        arrival_rate: float = 0.05,     # 每 tick 到达概率
        difficulty: str = "medium",
        S0: float = 3.2,
    ):
        """
        参数:
            arrival_rate: 每 tick 客户到达的概率
            difficulty: 难度（影响客户订单复杂度）
            S0: 初始标的价格
        """
        self.arrival_rate = arrival_rate
        self.difficulty = difficulty
        self.S0 = S0
        self.client_count = 0

        # 难度影响客户到达频率（提高：让游戏更有互动感）
        diff_rates = {
            "easy": 0.10,     # 每 tick 10% → 平均 10 天来 1 个
            "medium": 0.18,   # 每 tick 18% → 平均 5-6 天来 1 个
            "hard": 0.25,     # 每 tick 25% → 平均 4 天来 1 个
        }
        self.arrival_rate = diff_rates.get(difficulty, 0.18)

    def check_arrival(self, S: float, iv: float, T_market: float) -> Optional[dict]:
        """
        检查本 tick 是否有客户到来

        参数:
            S: 当前标的价
            iv: 当前 IV
            T_market: 市场剩余时间（年）

        返回:
            客户订单信息 dict 或 None
        """
        if np.random.random() > self.arrival_rate:
            return None

        # 有客户到来，生成订单
        self.client_count += 1
        return self._generate_order(S, iv, T_market)

    def _generate_order(self, S: float, iv: float, T_market: float) -> dict:
        """生成一个随机客户订单"""
        from pricing.black_scholes import BlackScholes

        # 客户名
        name = np.random.choice(CLIENT_NAMES)

        # 期权类型
        option_type = np.random.choice(["call", "put"], p=[0.55, 0.45])

        # 行权价选择（相对当前价格）— 偏向 ATM 以确保合理权利金
        moneyness = np.random.choice(
            ["ITM", "ATM", "OTM3", "OTM5"],
            p=[0.15, 0.45, 0.25, 0.15],
        )

        if option_type == "call":
            if moneyness == "ITM":
                K = S * np.random.uniform(0.96, 0.99)
            elif moneyness == "ATM":
                K = S * np.random.uniform(0.99, 1.01)
            elif moneyness == "OTM3":
                K = S * np.random.uniform(1.01, 1.04)
            else:  # OTM5
                K = S * np.random.uniform(1.04, 1.07)
        else:  # put
            if moneyness == "ITM":
                K = S * np.random.uniform(1.01, 1.04)
            elif moneyness == "ATM":
                K = S * np.random.uniform(0.99, 1.01)
            elif moneyness == "OTM3":
                K = S * np.random.uniform(0.96, 0.99)
            else:  # OTM5
                K = S * np.random.uniform(0.93, 0.96)

        # 对齐到合理精度
        K = round(K, 3)

        # 到期日选择
        T_options = [7, 14, 30, 60, 90]
        T_days = int(np.random.choice(T_options, p=[0.10, 0.20, 0.35, 0.25, 0.10]))
        T = min(T_days / 252, T_market)  # 不能超过市场剩余时间
        T = max(T, 3 / 252)  # 最少3天

        # 数量
        qty_options = [10, 20, 50, 100, 200]
        qty = int(np.random.choice(qty_options, p=[0.25, 0.30, 0.25, 0.15, 0.05]))

        # 计算理论价格（每份标的的期权价格）
        r = 0.03
        bs_price_per_share = BlackScholes.price(S, K, r, iv, T, option_type)

        # 转换为每张合约的价格（合约乘数 = 100）
        CONTRACT_MULTIPLIER = 100
        bs_price_per_contract = bs_price_per_share * CONTRACT_MULTIPLIER

        # 客户愿意支付的价格（每张合约，理论价 ± 偏差）
        bid_premium = bs_price_per_contract * np.random.uniform(0.95, 1.15)  # 偏向多付
        # 确保最低权利金
        min_premium = S * CONTRACT_MULTIPLIER * 0.005  # 至少是合约价值的0.5%
        bid_premium = round(max(bid_premium, min_premium), 4)

        total_premium = bid_premium * qty

        return {
            "client_name": name,
            "option_type": option_type,
            "K": K,
            "T": round(T, 4),
            "T_days": round(T * 252),
            "qty": qty,
            "bs_price": round(bs_price_per_contract, 4),
            "bid_premium": bid_premium,
            "total_premium": round(total_premium, 2),
            "moneyness": moneyness,
            "S_at_request": round(S, 4),
            "IV_at_request": round(iv, 4),
        }

    def reset(self):
        """重置生成器"""
        self.client_count = 0
