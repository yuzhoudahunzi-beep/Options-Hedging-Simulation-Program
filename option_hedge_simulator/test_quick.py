"""
快速验证脚本 — 测试所有核心模块
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

print("=" * 60)
print("期权对冲模拟器 — 快速验证")
print("=" * 60)

# 1. 测试 Black-Scholes 定价
print("\n[1] Black-Scholes 定价测试")
from pricing import BlackScholes, GreeksCalculator

S, K, r, sigma, T = 100, 100, 0.05, 0.2, 1.0
call_price = BlackScholes.call_price(S, K, r, sigma, T)
put_price = BlackScholes.put_price(S, K, r, sigma, T)

# Put-Call Parity: C - P = S - K*e^(-rT)
parity_check = call_price - put_price - (S - K * np.exp(-r * T))
print(f"  Call = {call_price:.4f}, Put = {put_price:.4f}")
print(f"  Put-Call Parity 误差: {abs(parity_check):.8f} (应接近 0)")
assert abs(parity_check) < 1e-8, "Put-Call Parity 失败!"

# 2. 测试 Greeks
print("\n[2] Greeks 测试")
greeks = GreeksCalculator.all_greeks(S, K, r, sigma, T, "call")
print(f"  Delta = {greeks['delta']:.4f}")
print(f"  Gamma = {greeks['gamma']:.6f}")
print(f"  Theta = {greeks['theta']:.4f}")
print(f"  Vega  = {greeks['vega']:.4f}")

# 解析 vs 数值 Delta
num_delta = GreeksCalculator.numerical_delta(S, K, r, sigma, T, "call")
print(f"  数值 Delta = {num_delta:.4f}, 解析 Delta = {greeks['delta']:.4f}")
assert abs(num_delta - greeks['delta']) < 0.001, "Delta 解析/数值不一致!"

# 3. 测试市场模拟器
print("\n[3] 市场模拟器测试")
from market import GBMSimulator, MertonSimulator, HestonSimulator

gbm = GBMSimulator(S0=100, mu=0.05, sigma=0.2, dt=1/252, seed=42)
gbm_paths = gbm.simulate(252, 100)
print(f"  GBM: shape={gbm_paths.shape}, 终值均值={gbm_paths[:, -1].mean():.2f}")
assert gbm_paths.shape == (100, 253)

merton = MertonSimulator(S0=100, mu=0.05, sigma=0.2, dt=1/252, lam=2.0, seed=42)
merton_paths = merton.simulate(252, 100)
print(f"  Merton: shape={merton_paths.shape}, 终值均值={merton_paths[:, -1].mean():.2f}")

heston = HestonSimulator(S0=100, mu=0.05, v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7, dt=1/252, seed=42)
heston_paths = heston.simulate(252, 100)
print(f"  Heston: shape={heston_paths.shape}, 终值均值={heston_paths[:, -1].mean():.2f}")

# 4. 测试对冲策略
print("\n[4] 对冲策略测试")
from strategies import DeltaHedgeStrategy, DeltaGammaHedgeStrategy, GammaScalpingStrategy

# 单条路径测试
test_path = gbm_paths[0]
test_times = np.arange(253) / 252.0

# Delta Hedge
delta_strat = DeltaHedgeStrategy(K=100, r=0.05, sigma=0.2, T=1.0, option_type="call")
delta_result = delta_strat.run(test_path, test_times, S0=100, is_short=True)
print(f"  Delta Hedge PnL: ¥{delta_result['pnl']:.2f}")

# Delta-Gamma Hedge
dg_strat = DeltaGammaHedgeStrategy(K=100, r=0.05, sigma=0.2, T=1.0, option_type="call")
dg_result = dg_strat.run(test_path, test_times, S0=100, is_short=True)
print(f"  Delta-Gamma Hedge PnL: ¥{dg_result['pnl']:.2f}")

# Gamma Scalping
gs_strat = GammaScalpingStrategy(K=100, r=0.05, implied_vol=0.2, T=1.0, option_type="call")
gs_result = gs_strat.run(test_path, test_times, S0=100)
print(f"  Gamma Scalping PnL: ¥{gs_result['pnl']:.2f}")
print(f"    已实现波动率: {gs_result['realized_vol']:.2%}")
print(f"    隐含波动率: {gs_result['implied_vol']:.2%}")

# 5. 测试蒙特卡洛引擎
print("\n[5] 蒙特卡洛引擎测试")
from simulation.engine import MonteCarloEngine

engine = MonteCarloEngine(
    S0=100, K=100, r=0.05, sigma=0.2, T=1.0,
    n_steps=252, n_paths=200, seed=42,
)

result = engine.run_simulation(
    model_type="gbm",
    strategy_type="delta",
    is_short=True,
    show_paths=20,
)
print(f"  路径数: {result['n_paths']}, 步数: {result['n_steps']}")
print(f"  平均 PnL: ¥{result['stats']['mean_pnl']:.2f}")
print(f"  胜率: {result['stats']['win_rate']:.1%}")
print(f"  Sharpe: {result['stats']['sharpe_ratio']:.3f}")

# 6. 测试策略对比
print("\n[6] 策略对比")
comparison = engine.compare_strategies(model_type="gbm", is_short=True)
for name, res in comparison.items():
    print(f"  {name:15s}: 平均PnL=¥{res['stats']['mean_pnl']:8.2f}  "
          f"胜率={res['stats']['win_rate']:.1%}  "
          f"Sharpe={res['stats']['sharpe_ratio']:.3f}")

print("\n" + "=" * 60)
print("✅ 所有测试通过！")
print("=" * 60)
print("\n启动 Web 界面: cd option_hedge_simulator && streamlit run app.py")
