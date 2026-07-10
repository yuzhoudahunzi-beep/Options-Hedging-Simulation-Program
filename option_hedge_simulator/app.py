"""
期权对冲模拟器 — Streamlit 交互式 Web 界面

运行方式: streamlit run app.py
"""

import sys
import os

# 确保项目根目录在 path 中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from market import GBMSimulator, MertonSimulator, HestonSimulator
from pricing import BlackScholes, GreeksCalculator
from strategies import DeltaHedgeStrategy, DeltaGammaHedgeStrategy, GammaScalpingStrategy
from simulation.engine import MonteCarloEngine
from utils.helpers import compute_statistics, format_currency, format_percentage

# ============ 页面配置 ============
st.set_page_config(
    page_title="期权对冲模拟器",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 自定义样式
st.markdown("""
<style>
    .metric-card {
        background: #f0f2f6;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
    }
    .profit { color: #e74c3c; font-weight: bold; }
    .loss { color: #27ae60; font-weight: bold; }
    .stMetric { background: #1e1e1e; padding: 10px; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)


# ============ 侧边栏 ============
st.sidebar.title("📊 期权对冲模拟器")
st.sidebar.markdown("---")

# 市场参数
st.sidebar.header("🏷️ 市场参数")
S0 = st.sidebar.number_input("初始标的价格 (S₀)", value=100.0, min_value=1.0, step=1.0)
K = st.sidebar.number_input("行权价 (K)", value=100.0, min_value=1.0, step=1.0)
r = st.sidebar.number_input("无风险利率 (r)", value=0.05, min_value=-0.1, max_value=1.0, step=0.01, format="%.3f")
T = st.sidebar.number_input("到期时间 T (年)", value=1.0, min_value=0.01, max_value=5.0, step=0.1, format="%.1f")
sigma = st.sidebar.slider("波动率 (σ)", min_value=0.05, max_value=1.0, value=0.20, step=0.01)

st.sidebar.markdown("---")

# 市场模型选择
st.sidebar.header("📈 市场模型")
model_type = st.sidebar.selectbox(
    "选择市场模型",
    ["gbm", "merton", "heston"],
    format_func=lambda x: {
        "gbm": "几何布朗运动 (GBM)",
        "merton": "跳跃扩散模型 (Merton)",
        "heston": "随机波动率 (Heston)",
    }[x],
)

model_params = {}
if model_type == "gbm":
    model_params["mu"] = st.sidebar.number_input("漂移率 μ", value=r, step=0.01, format="%.3f")
    model_params["sigma"] = sigma

elif model_type == "merton":
    model_params["mu"] = st.sidebar.number_input("漂移率 μ", value=r, step=0.01, format="%.3f")
    model_params["sigma"] = sigma
    st.sidebar.subheader("跳跃参数")
    model_params["lam"] = st.sidebar.slider("跳跃强度 λ (次/年)", 0.0, 10.0, 2.0, 0.5)
    model_params["mu_J"] = st.sidebar.number_input("跳跃均值 μ_J", value=0.0, step=0.01, format="%.2f")
    model_params["sigma_J"] = st.sidebar.number_input("跳跃标准差 σ_J", value=0.1, min_value=0.01, step=0.01, format="%.2f")

elif model_type == "heston":
    model_params["mu"] = st.sidebar.number_input("漂移率 μ", value=r, step=0.01, format="%.3f")
    st.sidebar.subheader("Heston 参数")
    model_params["v0"] = st.sidebar.number_input("初始方差 v₀", value=sigma**2, step=0.01, format="%.4f")
    model_params["kappa"] = st.sidebar.number_input("均值回归速度 κ", value=2.0, min_value=0.1, step=0.1, format="%.1f")
    model_params["theta"] = st.sidebar.number_input("长期方差 θ", value=sigma**2, step=0.01, format="%.4f")
    model_params["xi"] = st.sidebar.number_input("波动率的波动率 ξ", value=0.3, min_value=0.01, step=0.01, format="%.2f")
    model_params["rho"] = st.sidebar.slider("相关系数 ρ", -1.0, 1.0, -0.7, 0.05)

st.sidebar.markdown("---")

# 策略选择
st.sidebar.header("🎯 对冲策略")
strategy_type = st.sidebar.selectbox(
    "选择对冲策略",
    ["delta", "delta_gamma", "gamma_scalping"],
    format_func=lambda x: {
        "delta": "Delta Neutral 对冲",
        "delta_gamma": "Delta-Gamma 双因子对冲",
        "gamma_scalping": "Gamma Scalping",
    }[x],
)

strategy_params = {}
option_type = st.sidebar.selectbox("期权类型", ["call", "put"], format_func=lambda x: "看涨 (Call)" if x == "call" else "看跌 (Put)")
is_short = st.sidebar.checkbox("卖出期权 (Short)", value=True, help="卖方收取期权费，承担对冲义务")
rebalance_freq = st.sidebar.number_input("调仓频率 (步)", value=1, min_value=1, step=1, help="每 N 步调仓一次")
transaction_cost = st.sidebar.number_input("交易成本率", value=0.001, min_value=0.0, max_value=0.1, step=0.001, format="%.4f")

strategy_params["option_type"] = option_type
strategy_params["rebalance_freq"] = rebalance_freq
strategy_params["transaction_cost"] = transaction_cost

if strategy_type == "delta_gamma":
    st.sidebar.subheader("对冲期权参数")
    K_hedge_ratio = st.sidebar.slider("对冲期权行权价/S₀", 0.8, 1.2, 1.0, 0.05)
    strategy_params["K_hedge"] = S0 * K_hedge_ratio
    T_hedge_offset = st.sidebar.number_input("对冲期权额外到期时间", value=0.0, min_value=0.0, step=0.1, format="%.1f")
    strategy_params["T_hedge_offset"] = T_hedge_offset

if strategy_type == "gamma_scalping":
    implied_vol = st.sidebar.slider("隐含波动率", 0.05, 1.0, sigma, 0.01, help="买入期权时的隐含波动率")
    strategy_params["implied_vol"] = implied_vol

st.sidebar.markdown("---")

# 蒙特卡洛参数
st.sidebar.header("🔬 蒙特卡洛参数")
n_paths = st.sidebar.slider("路径数量", 100, 10000, 1000, 100)
n_steps = st.sidebar.slider("时间步数", 50, 504, 252, 10)
seed = st.sidebar.number_input("随机种子", value=42, step=1)
show_paths = st.sidebar.slider("展示路径数", 5, 200, 50, 5)

st.sidebar.markdown("---")
run_button = st.sidebar.button("🚀 运行模拟", type="primary", use_container_width=True)


# ============ 主区域 ============

# 顶部信息
st.title("📊 期权对冲模拟器")
st.markdown("""
模拟如何通过对冲策略把期权费赚出来。支持多种市场模型（GBM / 跳跃扩散 / 随机波动率）
和多种对冲策略（Delta Neutral / Delta-Gamma / Gamma Scalping），通过蒙特卡洛模拟验证策略有效性。
""")

# 初始定价信息
st.markdown("---")
col1, col2, col3, col4, col5 = st.columns(5)

bs_price = BlackScholes.price(S0, K, r, sigma, T, option_type)
bs_delta = GreeksCalculator.delta(S0, K, r, sigma, T, option_type)
bs_gamma = GreeksCalculator.gamma(S0, K, r, sigma, T)
bs_theta = GreeksCalculator.theta(S0, K, r, sigma, T, option_type)
bs_vega = GreeksCalculator.vega(S0, K, r, sigma, T)

with col1:
    st.metric("期权价格 (BS)", f"¥{bs_price:.2f}")
with col2:
    st.metric("Delta (Δ)", f"{bs_delta:.4f}")
with col3:
    st.metric("Gamma (Γ)", f"{bs_gamma:.6f}")
with col4:
    st.metric("Theta (Θ)", f"{bs_theta:.4f}")
with col5:
    st.metric("Vega (ν)", f"{bs_vega:.4f}")


if run_button:
    # 运行模拟
    progress_bar = st.progress(0, text="正在运行蒙特卡洛模拟...")

    engine = MonteCarloEngine(
        S0=S0, K=K, r=r, sigma=sigma, T=T,
        n_steps=n_steps, n_paths=n_paths, seed=seed,
    )

    def update_progress(p):
        progress_bar.progress(p, text=f"模拟中... {int(p*100)}%")

    result = engine.run_simulation(
        model_type=model_type,
        model_params=model_params,
        strategy_type=strategy_type,
        strategy_params=strategy_params,
        is_short=is_short,
        show_paths=show_paths,
        progress_callback=update_progress,
    )

    progress_bar.progress(1.0, text="模拟完成！")

    stats = result["stats"]
    sample = result["sample_result"]
    times = result["times"]
    prices = result["prices"]
    pnl_array = result["pnl_array"]
    pnl_paths = result["pnl_paths"]

    # ============ 统计指标 ============
    st.markdown("---")
    st.subheader("📋 模拟结果统计")

    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.metric("平均 PnL", f"¥{stats['mean_pnl']:.2f}")
    with m2:
        st.metric("PnL 标准差", f"¥{stats['std_pnl']:.2f}")
    with m3:
        win_color = "normal" if stats['win_rate'] >= 0.5 else "inverse"
        st.metric("胜率", format_percentage(stats['win_rate']))
    with m4:
        st.metric("Sharpe Ratio", f"{stats['sharpe_ratio']:.3f}")
    with m5:
        st.metric("偏度", f"{stats['skewness']:.3f}")

    # 详细统计表
    with st.expander("查看详细统计指标"):
        detail_cols = st.columns(4)
        with detail_cols[0]:
            st.write(f"**中位数 PnL**: ¥{stats['median_pnl']:.2f}")
            st.write(f"**最小 PnL**: ¥{stats['min_pnl']:.2f}")
            st.write(f"**最大 PnL**: ¥{stats['max_pnl']:.2f}")
        with detail_cols[1]:
            st.write(f"**5% 分位**: ¥{stats['p5']:.2f}")
            st.write(f"**25% 分位**: ¥{stats['p25']:.2f}")
            st.write(f"**75% 分位**: ¥{stats['p75']:.2f}")
        with detail_cols[2]:
            st.write(f"**95% 分位**: ¥{stats['p95']:.2f}")
            if "avg_max_drawdown" in stats:
                st.write(f"**平均最大回撤**: ¥{stats['avg_max_drawdown']:.2f}")
            if "worst_max_drawdown" in stats:
                st.write(f"**最差最大回撤**: ¥{stats['worst_max_drawdown']:.2f}")
        with detail_cols[3]:
            if strategy_type == "gamma_scalping":
                st.write(f"**已实现波动率**: {sample['realized_vol']:.2%}")
                st.write(f"**隐含波动率**: {sample['implied_vol']:.2%}")
                st.write(f"**Vol 差异**: {sample['vol_diff']:.2%}")

    # ============ 图表区域 ============
    st.markdown("---")

    # 1. 价格路径图
    st.subheader("📈 模拟价格路径")
    fig_prices = go.Figure()

    n_display = min(show_paths, len(prices))
    for i in range(n_display):
        fig_prices.add_trace(go.Scatter(
            x=times, y=prices[i],
            mode='lines',
            line=dict(color='rgba(100, 149, 237, 0.15)', width=0.5),
            showlegend=False,
            hoverinfo='skip',
        ))
    # 添加均值路径
    mean_path = np.mean(prices[:n_display], axis=0)
    fig_prices.add_trace(go.Scatter(
        x=times, y=mean_path,
        mode='lines',
        line=dict(color='red', width=2),
        name='均值路径',
    ))
    # 标注行权价
    fig_prices.add_hline(y=K, line_dash="dash", line_color="orange",
                          annotation_text=f"行权价 K={K}")

    fig_prices.update_layout(
        xaxis_title="时间 (年)",
        yaxis_title="标的价格",
        height=400,
        template="plotly_white",
    )
    st.plotly_chart(fig_prices, use_container_width=True)

    # 2. PnL 路径图
    st.subheader("💰 累计 PnL 路径")
    fig_pnl = go.Figure()

    n_pnl_display = min(show_paths, len(pnl_paths))
    for i in range(n_pnl_display):
        color = 'rgba(231, 76, 60, 0.2)' if pnl_paths[i][-1] > 0 else 'rgba(39, 174, 96, 0.2)'
        fig_pnl.add_trace(go.Scatter(
            x=times, y=pnl_paths[i],
            mode='lines',
            line=dict(color=color, width=0.5),
            showlegend=False,
            hoverinfo='skip',
        ))

    # 中位数路径
    median_pnl = np.median(pnl_paths, axis=0)
    fig_pnl.add_trace(go.Scatter(
        x=times, y=median_pnl,
        mode='lines',
        line=dict(color='gold', width=2.5),
        name='中位数 PnL',
    ))
    fig_pnl.add_hline(y=0, line_dash="solid", line_color="gray")

    fig_pnl.update_layout(
        xaxis_title="时间 (年)",
        yaxis_title="累计 PnL",
        height=400,
        template="plotly_white",
    )
    st.plotly_chart(fig_pnl, use_container_width=True)

    # 3. 两个并排图: PnL 分布 + Greeks 演化
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("📊 PnL 分布")
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=pnl_array,
            nbinsx=60,
            marker_color=np.where(
                np.histogram_bin_edges(pnl_array, bins=60)[:-1] > 0,
                'rgba(231, 76, 60, 0.7)',
                'rgba(39, 174, 96, 0.7)',
            ),
            name='PnL',
        ))
        fig_hist.add_vline(x=0, line_dash="solid", line_color="black")
        fig_hist.add_vline(x=stats['mean_pnl'], line_dash="dash", line_color="red",
                            annotation_text=f"均值 ¥{stats['mean_pnl']:.1f}")
        fig_hist.update_layout(
            xaxis_title="PnL (¥)",
            yaxis_title="频率",
            height=400,
            template="plotly_white",
            showlegend=False,
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    with col_right:
        st.subheader("📐 Greeks 演化 (样本路径)")
        fig_greeks = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                    subplot_titles=("Delta", "Gamma"))

        if sample:
            fig_greeks.add_trace(go.Scatter(
                x=times, y=sample["deltas"],
                mode='lines', line=dict(color='blue', width=1.5),
                name='Delta',
            ), row=1, col=1)
            fig_greeks.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=1)

            if "gammas" in sample:
                gamma_data = sample["gammas"]
            else:
                gamma_data = np.zeros_like(times)
            fig_greeks.add_trace(go.Scatter(
                x=times, y=gamma_data,
                mode='lines', line=dict(color='purple', width=1.5),
                name='Gamma',
            ), row=2, col=1)

        fig_greeks.update_layout(
            height=400,
            template="plotly_white",
            showlegend=True,
        )
        fig_greeks.update_xaxes(title_text="时间 (年)", row=2, col=1)
        st.plotly_chart(fig_greeks, use_container_width=True)

    # 4. 对冲头寸 + 期权价值
    col_left2, col_right2 = st.columns(2)

    with col_left2:
        st.subheader("🔄 对冲头寸变化")
        fig_hedge = go.Figure()

        if sample:
            if strategy_type == "delta_gamma":
                fig_hedge.add_trace(go.Scatter(
                    x=times, y=sample["hedge_stock"],
                    mode='lines', line=dict(color='blue', width=1.5),
                    name='标的持仓',
                ))
                fig_hedge.add_trace(go.Scatter(
                    x=times, y=sample["hedge_option_qty"],
                    mode='lines', line=dict(color='orange', width=1.5),
                    name='对冲期权持仓',
                ))
            else:
                fig_hedge.add_trace(go.Scatter(
                    x=times, y=sample["hedge_positions"],
                    mode='lines', line=dict(color='blue', width=1.5),
                    name='标的持仓',
                ))

        fig_hedge.update_layout(
            xaxis_title="时间 (年)",
            yaxis_title="持仓数量",
            height=350,
            template="plotly_white",
        )
        st.plotly_chart(fig_hedge, use_container_width=True)

    with col_right2:
        st.subheader("💎 期权价值变化 (样本路径)")
        fig_optval = go.Figure()

        if sample:
            fig_optval.add_trace(go.Scatter(
                x=times, y=sample["option_values"],
                mode='lines', line=dict(color='green', width=1.5),
                name='期权价值',
            ))
            fig_optval.add_trace(go.Scatter(
                x=times, y=prices[0],
                mode='lines', line=dict(color='gray', width=1, dash='dash'),
                name='标的价格',
            ))

        fig_optval.update_layout(
            xaxis_title="时间 (年)",
            yaxis_title="价值 (¥)",
            height=350,
            template="plotly_white",
        )
        st.plotly_chart(fig_optval, use_container_width=True)

    # 5. Gamma Scalping 专属: Gamma vs Theta
    if strategy_type == "gamma_scalping" and sample:
        st.markdown("---")
        st.subheader("⚔️ Gamma 收益 vs Theta 衰减")

        fig_gt = go.Figure()
        fig_gt.add_trace(go.Scatter(
            x=times, y=sample["gamma_pnl_components"],
            mode='lines', line=dict(color='red', width=2),
            name='Gamma 收益 (0.5·Γ·dS²)',
        ))
        fig_gt.add_trace(go.Scatter(
            x=times, y=sample["theta_pnl_components"],
            mode='lines', line=dict(color='blue', width=2),
            name='Theta 衰减 (Θ·dt)',
        ))
        fig_gt.add_trace(go.Scatter(
            x=times, y=sample["gamma_pnl_components"] + sample["theta_pnl_components"],
            mode='lines', line=dict(color='gold', width=2.5),
            name='净收益 (Gamma + Theta)',
        ))

        fig_gt.update_layout(
            xaxis_title="时间 (年)",
            yaxis_title="累计 PnL (¥)",
            height=400,
            template="plotly_white",
            title=f"已实现波动率: {sample['realized_vol']:.2%} | "
                  f"隐含波动率: {sample['implied_vol']:.2%} | "
                  f"差异: {sample['vol_diff']:.2%}",
        )
        st.plotly_chart(fig_gt, use_container_width=True)

    # 6. 数据下载
    st.markdown("---")
    st.subheader("📥 数据导出")

    # 导出 PnL 数据
    pnl_df = pd.DataFrame({
        "路径编号": range(1, len(pnl_array) + 1),
        "最终 PnL (¥)": pnl_array,
        "是否盈利": pnl_array > 0,
    })

    csv = pnl_df.to_csv(index=False).encode('utf-8-sig')
    st.download_button(
        label="📥 下载 PnL 结果 (CSV)",
        data=csv,
        file_name="hedge_simulation_pnl.csv",
        mime="text/csv",
    )

else:
    # 未运行时的提示
    st.info("👈 请在左侧面板设置参数，然后点击 **运行模拟** 按钮开始蒙特卡洛模拟。")

    # 显示一些示例说明
    st.markdown("---")
    st.subheader("💡 使用说明")

    tab1, tab2, tab3 = st.tabs(["Delta Neutral", "Delta-Gamma", "Gamma Scalping"])

    with tab1:
        st.markdown("""
        ### Delta Neutral 对冲

        **核心思路**: 卖出期权后，持有 Δ 份标的资产使组合 Delta = 0

        - 初始: 卖出 1 份期权，买入 Δ 份标的
        - 每个调仓步: 重新计算 Δ，调整标的头寸至 Delta Neutral
        - PnL = 期权费收入 - 调仓成本 + 对冲损益

        **赚钱条件**: 已实现波动率 ≈ 隐含波动率时，时间价值衰减 (Theta) 是利润来源
        """)

    with tab2:
        st.markdown("""
        ### Delta-Gamma 双因子对冲

        **核心思路**: 同时消除 Delta 和 Gamma 风险

        - 需要两个对冲工具: 标的资产 + 另一个期权
        - 解方程组同时令 Delta=0 和 Gamma=0
        - 对冲更精确，但需要交易两种工具

        **优势**: 减少因标的价格大幅变动带来的 Gamma 风险暴露
        """)

    with tab3:
        st.markdown("""
        ### Gamma Scalping

        **核心思路**: 做市商策略 — 买入期权（Long Gamma），Delta 对冲

        - 买入期权（支付期权费）
        - 持续 Delta 对冲维持 Delta Neutral
        - 每次对冲调整都在"低买高卖"（因为 Long Gamma 的凸性收益）

        **赚钱条件**: 已实现波动率 > 隐含波动率 → Gamma 收益 > Theta 衰减 → 盈利
        """)

# ============ 页脚 ============
st.markdown("---")
st.markdown(
    '<div style="text-align: center; color: gray; font-size: 0.8em;">'
    '期权对冲模拟器 v1.0 | 蒙特卡洛模拟 | '
    '支持 GBM / Merton Jump Diffusion / Heston 模型'
    '</div>',
    unsafe_allow_html=True,
)
