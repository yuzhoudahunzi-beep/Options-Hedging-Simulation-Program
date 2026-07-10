/**
 * 期权对冲训练系统 — 前端核心逻辑
 *
 * 功能:
 *  - WebSocket 连接后端
 *  - 实时接收市场更新、P&L 变化
 *  - ECharts 图表实时更新
 *  - 用户操作发送
 */

// ============ 全局状态 ============
let ws = null;
let priceChart = null;
let pnlChart = null;
let priceData = [];    // [{step, S, IV}]
let pnlData = [];      // [{step, total_pnl}]
let gameState = 'setup';
let prevPnl = 0;
let prevDay = -1;

// ============ 初始化 ============
document.addEventListener('DOMContentLoaded', () => {
    initCharts();
    bindEvents();
});

function initCharts() {
    const priceEl = document.getElementById('chart-price');
    const pnlEl = document.getElementById('chart-pnl');

    if (priceEl) {
        priceChart = echarts.init(priceEl, 'dark');
        priceChart.setOption(getPriceChartOption());
    }
    if (pnlEl) {
        pnlChart = echarts.init(pnlEl, 'dark');
        pnlChart.setOption(getPnlChartOption());
    }

    // 响应式
    window.addEventListener('resize', () => {
        priceChart && priceChart.resize();
        pnlChart && pnlChart.resize();
    });
}

function getPriceChartOption() {
    return {
        backgroundColor: 'transparent',
        animation: false,
        title: {
            text: '📈 标的价格走势',
            left: 10, top: 5,
            textStyle: { color: '#8b949e', fontSize: 13 },
        },
        tooltip: {
            trigger: 'axis',
            backgroundColor: '#1c2333',
            borderColor: '#30363d',
            textStyle: { color: '#e6edf3' },
        },
        grid: { left: 60, right: 30, top: 40, bottom: 30 },
        xAxis: {
            type: 'category',
            data: [],
            axisLine: { lineStyle: { color: '#30363d' } },
            axisLabel: { color: '#8b949e' },
            name: '交易日',
            nameTextStyle: { color: '#6e7681' },
        },
        yAxis: [
            {
                type: 'value',
                name: '价格',
                nameTextStyle: { color: '#6e7681' },
                axisLine: { lineStyle: { color: '#30363d' } },
                axisLabel: { color: '#8b949e' },
                splitLine: { lineStyle: { color: '#21262d' } },
            },
            {
                type: 'value',
                name: 'IV%',
                nameTextStyle: { color: '#6e7681' },
                axisLine: { lineStyle: { color: '#30363d' } },
                axisLabel: { color: '#d29922' },
                splitLine: { show: false },
            },
        ],
        series: [
            {
                name: '标的价格',
                type: 'line',
                data: [],
                lineStyle: { color: '#58a6ff', width: 2 },
                itemStyle: { color: '#58a6ff' },
                showSymbol: false,
                yAxisIndex: 0,
            },
            {
                name: '隐含波动率',
                type: 'line',
                data: [],
                lineStyle: { color: '#d29922', width: 1.5, type: 'dashed' },
                itemStyle: { color: '#d29922' },
                showSymbol: false,
                yAxisIndex: 1,
            },
        ],
    };
}

function getPnlChartOption() {
    return {
        backgroundColor: 'transparent',
        animation: false,
        title: {
            text: '💰 累计损益 (P&L)',
            left: 10, top: 5,
            textStyle: { color: '#8b949e', fontSize: 13 },
        },
        tooltip: {
            trigger: 'axis',
            backgroundColor: '#1c2333',
            borderColor: '#30363d',
            textStyle: { color: '#e6edf3' },
            formatter: function(params) {
                const p = params[0];
                const val = p.value;
                const color = val >= 0 ? '#3fb950' : '#f85149';
                return `Day ${p.axisValue}<br/><span style="color:${color};font-weight:bold">¥${val.toLocaleString()}</span>`;
            },
        },
        grid: { left: 70, right: 30, top: 40, bottom: 30 },
        xAxis: {
            type: 'category',
            data: [],
            axisLine: { lineStyle: { color: '#30363d' } },
            axisLabel: { color: '#8b949e' },
            name: '交易日',
            nameTextStyle: { color: '#6e7681' },
        },
        yAxis: {
            type: 'value',
            name: 'P&L (¥)',
            nameTextStyle: { color: '#6e7681' },
            axisLine: { lineStyle: { color: '#30363d' } },
            axisLabel: { color: '#8b949e' },
            splitLine: { lineStyle: { color: '#21262d' } },
        },
        series: [
            {
                name: 'P&L',
                type: 'line',
                data: [],
                lineStyle: { width: 2.5 },
                itemStyle: { color: '#3fb950' },
                areaStyle: {
                    color: {
                        type: 'linear', x: 0, y: 0, x2: 0, y2: 1,
                        colorStops: [
                            { offset: 0, color: 'rgba(63, 185, 80, 0.3)' },
                            { offset: 1, color: 'rgba(63, 185, 80, 0.02)' },
                        ],
                    },
                },
                showSymbol: false,
                markLine: {
                    silent: true,
                    data: [{ yAxis: 0, lineStyle: { color: '#6e7681', type: 'solid' } }],
                    label: { show: false },
                },
            },
        ],
        visualMap: {
            show: false,
            pieces: [
                { gt: 0, color: '#3fb950' },
                { lte: 0, color: '#f85149' },
            ],
            seriesIndex: 0,
        },
    };
}

// ============ 事件绑定 ============
function bindEvents() {
    // Setup
    document.getElementById('btn-start').addEventListener('click', startGame);

    // Trading controls
    document.getElementById('btn-pause').addEventListener('click', () => sendAction({ type: 'pause' }));
    document.getElementById('btn-resume').addEventListener('click', () => sendAction({ type: 'resume' }));
    document.getElementById('btn-stop').addEventListener('click', () => {
        sendAction({ type: 'restart' });
        resetAllState();
        switchScreen('setup-screen');
    });

    // Stock trading
    document.getElementById('btn-buy-stock').addEventListener('click', () => {
        const qty = parseInt(document.getElementById('input-stock-qty').value) || 0;
        if (qty > 0) sendAction({ type: 'buy_stock', qty });
    });
    document.getElementById('btn-sell-stock').addEventListener('click', () => {
        const qty = parseInt(document.getElementById('input-stock-qty').value) || 0;
        if (qty > 0) sendAction({ type: 'sell_stock', qty });
    });

    // Delta hedge
    document.getElementById('btn-delta-hedge').addEventListener('click', () => {
        sendAction({ type: 'delta_hedge' });
    });

    // Close option
    document.getElementById('btn-close-option').addEventListener('click', () => {
        const sel = document.getElementById('select-close-option');
        const id = sel.value;
        if (id) sendAction({ type: 'close_option', option_id: parseInt(id) });
    });

    // Order popup
    document.getElementById('btn-accept-order').addEventListener('click', () => {
        sendAction({ type: 'accept_order' });
        document.getElementById('order-popup').classList.add('hidden');
    });
    document.getElementById('btn-reject-order').addEventListener('click', () => {
        sendAction({ type: 'reject_order' });
        document.getElementById('order-popup').classList.add('hidden');
    });

    // Restart
    document.getElementById('btn-restart').addEventListener('click', () => {
        sendAction({ type: 'restart' });
        resetAllState();
        switchScreen('setup-screen');
    });

    // Hint button
    document.getElementById('btn-hint').addEventListener('click', () => {
        sendAction({ type: 'get_hint' });
    });
}

// ============ 屏幕切换 ============
function switchScreen(screenId) {
    document.querySelectorAll('.screen').forEach(s => s.classList.remove('active'));
    document.getElementById(screenId).classList.add('active');

    if (screenId === 'trading-screen') {
        // 延迟更长时间让 DOM 完全渲染后再初始化图表
        setTimeout(() => {
            if (priceChart) {
                priceChart.resize();
                priceChart.setOption(getPriceChartOption());
            }
            if (pnlChart) {
                pnlChart.resize();
                pnlChart.setOption(getPnlChartOption());
            }
        }, 300);
    }
}

// ============ 重置全部状态 ============
function resetAllState() {
    // 关闭旧 WebSocket
    if (ws) {
        ws.onclose = null; // 防止触发断开日志
        ws.close();
        ws = null;
    }

    // 清空数据
    priceData = [];
    pnlData = [];
    prevPnl = 0;
    prevDay = -1;
    gameState = 'setup';

    // 重置图表
    if (priceChart) {
        priceChart.setOption({
            xAxis: { data: [] },
            series: [{ data: [] }, { data: [] }],
        });
    }
    if (pnlChart) {
        pnlChart.setOption({
            xAxis: { data: [] },
            series: [{ data: [] }],
        });
    }

    // 清空事件日志
    const logEl = document.getElementById('event-log');
    if (logEl) logEl.innerHTML = '';

    // 隐藏订单弹窗
    const popup = document.getElementById('order-popup');
    if (popup) popup.classList.add('hidden');

    // 重置 UI 显示
    const dayEl = document.getElementById('hdr-day');
    if (dayEl) dayEl.textContent = 'Day 0/60';
    const fillEl = document.getElementById('day-progress-fill');
    if (fillEl) fillEl.style.width = '0%';
    const pctEl = document.getElementById('hdr-day-pct');
    if (pctEl) pctEl.textContent = '0%';

    document.getElementById('hdr-price').textContent = '3.2000';
    document.getElementById('hdr-change').textContent = '+0.00%';
    document.getElementById('hdr-iv').textContent = '18.0%';

    // 重置持仓面板
    const posPanel = document.getElementById('positions-panel');
    if (posPanel) posPanel.innerHTML = '<div class="empty-hint">暂无持仓</div>';

    // 重置资金和损益显示
    document.getElementById('stat-cash').textContent = '¥100,000';
    document.getElementById('stat-stock').textContent = '0 股';
    document.getElementById('stat-avg-cost').textContent = '-';
    document.getElementById('pnl-total').textContent = '¥0';
    document.getElementById('pnl-total').className = 'pnl-value zero';
    document.getElementById('pnl-option').textContent = '¥0';
    document.getElementById('pnl-stock').textContent = '¥0';
    document.getElementById('pnl-drawdown').textContent = '¥0';

    // 重置 Greeks
    document.getElementById('greek-delta').textContent = '0';
    document.getElementById('greek-delta-sub').textContent = '';
    document.getElementById('greek-gamma').textContent = '0';
    document.getElementById('greek-vega').textContent = '0';
    document.getElementById('greek-theta').textContent = '0';

    // 重置风险面板
    document.getElementById('risk-delta-dev').textContent = '0';
    document.getElementById('risk-n-options').textContent = '0';
    const riskEl = document.getElementById('risk-level');
    riskEl.textContent = '低';
    riskEl.className = 'badge risk-low';

    // 重置权利金显示
    const premEl = document.getElementById('stat-premium');
    if (premEl) premEl.textContent = '¥0';

    // 暂停/继续按钮
    document.getElementById('btn-pause').classList.remove('hidden');
    document.getElementById('btn-resume').classList.add('hidden');
}

// ============ 游戏控制 ============
function startGame() {
    // 重置上一局的所有状态
    resetAllState();

    const cfg = {
        type: 'start',
        underlying: document.getElementById('cfg-underlying').value,
        mode: document.getElementById('cfg-mode').value,
        difficulty: document.getElementById('cfg-difficulty').value,
        total_days: parseInt(document.getElementById('cfg-days').value) || 60,
        initial_cash: parseFloat(document.getElementById('cfg-cash').value) || 100000,
        speed: parseFloat(document.getElementById('cfg-speed').value) || 1,
    };

    switchScreen('trading-screen');
    connectWebSocket(cfg);
}

// ============ WebSocket ============
function connectWebSocket(startAction) {
    // 确保旧连接已关闭
    if (ws) {
        ws.onclose = null;
        ws.close();
        ws = null;
    }

    const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
    ws = new WebSocket(`${protocol}//${location.host}/ws`);

    ws.onopen = () => {
        console.log('[WS] 已连接');
        addLog('系统连接成功', 'info');
        if (startAction) {
            sendAction(startAction);
        }
    };

    ws.onmessage = (event) => {
        const msg = JSON.parse(event.data);
        handleMessage(msg);
    };

    ws.onclose = () => {
        console.log('[WS] 断开');
        addLog('连接断开，请刷新页面', 'danger');
    };

    ws.onerror = (err) => {
        console.error('[WS] 错误:', err);
    };
}

function sendAction(action) {
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify(action));
    }
}

// ============ 消息处理 ============
function handleMessage(msg) {
    switch (msg.type) {
        case 'state_sync':
            handleStateSync(msg);
            break;
        case 'tick':
            handleTick(msg);
            break;
        case 'info':
            addLog(msg.message, 'info');
            break;
        case 'settled':
            handleSettlement(msg);
            break;
        case 'action_result':
            if (msg.status === 'error') {
                addLog('❌ ' + msg.message, 'danger');
            } else if (msg.suggestion) {
                // 这是提示结果
                showHintPopup(msg);
            }
            break;
    }
}

function handleStateSync(msg) {
    gameState = msg.game_state;

    if (msg.market) {
        updateHeader(msg.market);
    }

    if (msg.price_history) {
        priceData = msg.price_history;
        updatePriceChart();
    }

    if (msg.pnl_history) {
        pnlData = msg.pnl_history;
        updatePnlChart();
    }

    if (msg.portfolio) {
        updatePortfolio(msg.portfolio);
    }

    if (msg.pending_order) {
        showOrderPopup(msg.pending_order);
    }

    if (msg.game) {
        updateGameInfo(msg.game);
    }

    if (msg.game_state === 'trading') {
        switchScreen('trading-screen');
    } else if (msg.game_state === 'settled' && msg.score) {
        showResult(msg.score);
    }
}

function handleTick(msg) {
    // 市场更新
    if (msg.market) {
        updateHeader(msg.market);
        priceData.push({
            step: msg.market.step,
            S: msg.market.S,
            IV: msg.market.IV,
        });
        // 限制数据量
        if (priceData.length > 500) priceData.shift();
        updatePriceChart();
    }

    // 组合更新
    if (msg.portfolio) {
        updatePortfolio(msg.portfolio);
        pnlData.push({
            step: msg.game.tick_count,
            total_pnl: msg.portfolio.total_pnl,
        });
        if (pnlData.length > 500) pnlData.shift();
        updatePnlChart();

        // P&L 闪烁效果
        const pnlEl = document.getElementById('pnl-total');
        if (msg.portfolio.total_pnl > prevPnl) {
            pnlEl.classList.add('flash-up');
            setTimeout(() => pnlEl.classList.remove('flash-up'), 500);
        } else if (msg.portfolio.total_pnl < prevPnl) {
            pnlEl.classList.add('flash-down');
            setTimeout(() => pnlEl.classList.remove('flash-down'), 500);
        }
        prevPnl = msg.portfolio.total_pnl;
    }

    // 累计权利金
    if (msg.total_premium_earned !== undefined) {
        const premEl = document.getElementById('stat-premium');
        if (premEl) premEl.textContent = '¥' + formatNum(msg.total_premium_earned);
    }

    // 游戏信息
    if (msg.game) {
        updateGameInfo(msg.game);
    }

    // 新客户
    if (msg.new_order) {
        showOrderPopup(msg.new_order);
        addLog(`🟢 客户"${msg.new_order.client_name}"来了！想买${msg.new_order.qty}张 ` +
               `${msg.new_order.option_type.toUpperCase()} K=${msg.new_order.K.toFixed(3)} ` +
               `T=${msg.new_order.T_days}天 权利金¥${msg.new_order.total_premium.toFixed(0)}`, 'event');
    }

    // 市场事件
    if (msg.market_event) {
        const events = {
            'jump': '💥 标的价格跳跃！',
            'vol_spike': '⚠️ 波动率飙升！期权价格大涨！',
            'vol_crash': '📉 波动率坍塌！期权价格下跌！',
        };
        addLog(events[msg.market_event] || '市场事件', 'warning');
    }
}

// ============ UI 更新 ============
function updateHeader(market) {
    document.getElementById('hdr-price').textContent = market.S.toFixed(4);
    document.getElementById('hdr-iv').textContent = (market.IV * 100).toFixed(1) + '%';

    if (market.dS_pct !== undefined) {
        const chgEl = document.getElementById('hdr-change');
        const pct = (market.dS_pct * 100).toFixed(2);
        if (market.dS_pct > 0) {
            chgEl.textContent = `+${pct}%`;
            chgEl.className = 'change up';
        } else if (market.dS_pct < 0) {
            chgEl.textContent = `${pct}%`;
            chgEl.className = 'change down';
        } else {
            chgEl.textContent = '0.00%';
            chgEl.className = 'change flat';
        }
    }
}

function updateGameInfo(game) {
    const dayText = `Day ${game.day}/${game.total_days}`;
    const dayEl = document.getElementById('hdr-day');
    dayEl.textContent = dayText;

    // 进度条
    const pct = game.total_days > 0 ? Math.round((game.day / game.total_days) * 100) : 0;
    const fillEl = document.getElementById('day-progress-fill');
    const pctEl = document.getElementById('hdr-day-pct');
    if (fillEl) fillEl.style.width = pct + '%';
    if (pctEl) pctEl.textContent = pct + '%';

    // 数字变化时闪烁
    if (game.day !== prevDay && prevDay >= 0) {
        dayEl.classList.remove('flash');
        void dayEl.offsetWidth; // 强制 reflow 重启动画
        dayEl.classList.add('flash');
    }
    prevDay = game.day;

    // 暂停/继续按钮
    if (game.state === 'paused') {
        document.getElementById('btn-pause').classList.add('hidden');
        document.getElementById('btn-resume').classList.remove('hidden');
    } else if (game.state === 'trading') {
        document.getElementById('btn-pause').classList.remove('hidden');
        document.getElementById('btn-resume').classList.add('hidden');
    }
}

function updatePortfolio(pf) {
    // 资金
    document.getElementById('stat-cash').textContent = '¥' + formatNum(pf.cash);
    document.getElementById('stat-stock').textContent = pf.stock_position + ' 股';
    document.getElementById('stat-avg-cost').textContent =
        pf.stock_avg_cost > 0 ? pf.stock_avg_cost.toFixed(4) : '-';

    // P&L
    const pnlTotal = document.getElementById('pnl-total');
    pnlTotal.textContent = '¥' + formatNum(pf.total_pnl);
    pnlTotal.className = 'pnl-value ' + (pf.total_pnl > 0 ? 'profit' : pf.total_pnl < 0 ? 'loss' : 'zero');

    document.getElementById('pnl-option').textContent = '¥' + formatNum(pf.option_pnl);
    setPnlColor('pnl-option', pf.option_pnl);

    document.getElementById('pnl-stock').textContent = '¥' + formatNum(pf.stock_pnl);
    setPnlColor('pnl-stock', pf.stock_pnl);

    document.getElementById('pnl-drawdown').textContent = '¥' + formatNum(pf.max_drawdown);
    setPnlColor('pnl-drawdown', pf.max_drawdown);

    // Greeks
    document.getElementById('greek-delta').textContent = formatNum(pf.total_delta);
    document.getElementById('greek-delta-sub').textContent =
        `期权:${formatNum(pf.option_delta)} 标的:${pf.stock_delta}`;
    document.getElementById('greek-gamma').textContent = formatNum(pf.total_gamma, 2);
    document.getElementById('greek-vega').textContent = formatNum(pf.total_vega, 2);
    document.getElementById('greek-theta').textContent = formatNum(pf.total_theta, 2);

    // Delta 颜色
    const deltaEl = document.getElementById('greek-delta');
    const absDelta = Math.abs(pf.total_delta);
    if (absDelta < 1000) deltaEl.style.color = '#3fb950';
    else if (absDelta < 5000) deltaEl.style.color = '#d29922';
    else deltaEl.style.color = '#f85149';

    // 风险分析
    document.getElementById('risk-delta-dev').textContent = formatNum(pf.net_exposure);
    document.getElementById('risk-n-options').textContent = pf.n_options;

    const riskLevel = document.getElementById('risk-level');
    if (absDelta < 1000) {
        riskLevel.textContent = '低';
        riskLevel.className = 'badge risk-low';
    } else if (absDelta < 5000) {
        riskLevel.textContent = '中';
        riskLevel.className = 'badge risk-medium';
    } else {
        riskLevel.textContent = '高';
        riskLevel.className = 'badge risk-high';
    }

    // 持仓列表
    updatePositionsPanel(pf.option_details || []);

    // 平仓选择器
    updateCloseOptionSelect(pf.option_details || []);
}

function updatePositionsPanel(positions) {
    const panel = document.getElementById('positions-panel');

    if (positions.length === 0) {
        panel.innerHTML = '<div class="empty-hint">暂无期权持仓</div>';
        return;
    }

    let html = '';
    for (const pos of positions) {
        const pnlClass = pos.pnl >= 0 ? 'profit' : 'loss';
        const typeClass = pos.option_type;
        html += `
        <div class="position-card">
            <div class="pos-header">
                <span class="pos-type ${typeClass}">#${pos.id} ${pos.option_type.toUpperCase()} K=${pos.K.toFixed(3)}</span>
                <span class="pos-pnl" style="color: var(--${pos.pnl >= 0 ? 'profit' : 'loss'})">¥${formatNum(pos.pnl)}</span>
            </div>
            <div class="pos-detail">
                ${pos.client_name} | ${pos.qty}张 | 剩${pos.T_days}天 | 收¥${formatNum(pos.premium_received)}
            </div>
        </div>`;
    }
    panel.innerHTML = html;
}

function updateCloseOptionSelect(positions) {
    const sel = document.getElementById('select-close-option');
    let html = '<option value="">选择要平仓的期权</option>';
    for (const pos of positions) {
        html += `<option value="${pos.id}">#${pos.id} ${pos.option_type.toUpperCase()} K=${pos.K.toFixed(3)} ${pos.qty}张</option>`;
    }
    sel.innerHTML = html;
}

function showOrderPopup(order) {
    const popup = document.getElementById('order-popup');
    const details = document.getElementById('order-details');

    const typeLabel = order.option_type === 'call' ? '看涨 Call' : '看跌 Put';

    details.innerHTML = `
        <div class="order-line"><span class="order-label">客户</span><span class="order-val">${order.client_name}</span></div>
        <div class="order-line"><span class="order-label">类型</span><span class="order-val">${typeLabel}</span></div>
        <div class="order-line"><span class="order-label">行权价</span><span class="order-val">${order.K.toFixed(3)}</span></div>
        <div class="order-line"><span class="order-label">到期</span><span class="order-val">${order.T_days} 天</span></div>
        <div class="order-line"><span class="order-label">数量</span><span class="order-val">${order.qty} 张</span></div>
        <div class="order-line"><span class="order-label">BS理论价</span><span class="order-val">¥${order.bs_price.toFixed(4)}</span></div>
        <div class="order-line"><span class="order-label">出价</span><span class="order-val" style="color:var(--green)">¥${order.bid_premium.toFixed(4)}/张</span></div>
        <div class="order-line"><span class="order-label">总权利金</span><span class="order-val" style="color:var(--green);font-size:1.2em">¥${formatNum(order.total_premium)}</span></div>
    `;

    popup.classList.remove('hidden');
}

// ============ 提示弹窗 ============
function showHintPopup(hint) {
    document.getElementById('hint-suggestion').textContent = hint.suggestion;
    document.getElementById('hint-reason').textContent = hint.reason;

    const actionsList = document.getElementById('hint-actions');
    actionsList.innerHTML = (hint.actions || []).map(a => `<li>${a}</li>`).join('');

    const badge = document.getElementById('hint-risk-badge');
    const riskLabels = { low: '低风险', medium: '中风险', high: '高风险' };
    badge.textContent = riskLabels[hint.risk_level] || '未知';
    badge.className = `hint-risk-badge risk-${hint.risk_level || 'low'}`;

    document.getElementById('hint-popup').classList.remove('hidden');
}

// ============ 图表更新 ============
function updatePriceChart() {
    if (!priceChart || priceData.length === 0) return;

    const steps = priceData.map(d => d.step);
    const prices = priceData.map(d => d.S);
    const ivs = priceData.map(d => (d.IV * 100).toFixed(1));

    priceChart.setOption({
        xAxis: { data: steps },
        series: [
            { data: prices },
            { data: ivs },
        ],
    });
}

function updatePnlChart() {
    if (!pnlChart || pnlData.length === 0) return;

    const steps = pnlData.map(d => d.step);
    const pnls = pnlData.map(d => d.total_pnl);

    pnlChart.setOption({
        xAxis: { data: steps },
        series: [{ data: pnls }],
    });
}

// ============ 事件日志 ============
function addLog(message, type = 'info') {
    const log = document.getElementById('event-log');
    const now = new Date();
    const time = `${now.getHours().toString().padStart(2, '0')}:${now.getMinutes().toString().padStart(2, '0')}:${now.getSeconds().toString().padStart(2, '0')}`;

    const entry = document.createElement('div');
    entry.className = `log-entry log-${type}`;
    entry.innerHTML = `<span class="log-time">[${time}]</span> ${message}`;

    log.insertBefore(entry, log.firstChild);

    // 限制条数
    while (log.children.length > 100) {
        log.removeChild(log.lastChild);
    }
}

// ============ 结算页面 ============
function handleSettlement(msg) {
    gameState = 'settled';
    if (msg.score) {
        showResult(msg.score);
    }
}

function showResult(score) {
    switchScreen('result-screen');

    document.getElementById('result-grade').textContent = score.grade;
    document.getElementById('result-score').textContent = score.total_score + ' 分';

    const pnlVal = score.stats.final_pnl || 0;
    const pnlEl = document.getElementById('result-pnl-value');
    pnlEl.textContent = (pnlVal >= 0 ? '+' : '') + '¥' + formatNum(pnlVal);
    pnlEl.className = pnlVal >= 0 ? 'profit' : 'loss';

    document.getElementById('result-comment').textContent = score.comment;

    // 评分明细
    const bdGrid = document.querySelector('.breakdown-grid');
    bdGrid.innerHTML = `
        <div class="breakdown-item"><span>收益得分</span><span>${score.breakdown.profit_score}/40</span></div>
        <div class="breakdown-item"><span>Delta控制</span><span>${score.breakdown.delta_score}/30</span></div>
        <div class="breakdown-item"><span>回撤控制</span><span>${score.breakdown.drawdown_score}/15</span></div>
        <div class="breakdown-item"><span>操作活跃度</span><span>${score.breakdown.activity_score}/15</span></div>
    `;

    // 建议
    const sugList = document.querySelector('#result-suggestions ul');
    sugList.innerHTML = score.suggestions.map(s => `<li>• ${s}</li>`).join('');

    // 统计
    const statsGrid = document.querySelector('.stats-grid');
    const st = score.stats;
    statsGrid.innerHTML = `
        <div class="stat-row"><span>最终损益</span><span class="value" style="color:var(--${pnlVal >= 0 ? 'profit' : 'loss'})">¥${formatNum(st.final_pnl || 0)}</span></div>
        <div class="stat-row"><span>总权利金</span><span class="value">¥${formatNum(st.total_premium || 0)}</span></div>
        <div class="stat-row"><span>收益占比</span><span class="value">${((st.pnl_ratio || 0) * 100).toFixed(1)}%</span></div>
        <div class="stat-row"><span>平均|Delta|</span><span class="value">${formatNum(st.avg_abs_delta || 0)}</span></div>
        <div class="stat-row"><span>最大回撤</span><span class="value">¥${formatNum(st.max_drawdown || 0)}</span></div>
        <div class="stat-row"><span>操作次数</span><span class="value">${st.total_actions || 0}</span></div>
        <div class="stat-row"><span>对冲次数</span><span class="value">${st.hedge_actions || 0}</span></div>
        <div class="stat-row"><span>期权交易数</span><span class="value">${st.n_option_trades || 0}</span></div>
    `;
}

// ============ 工具函数 ============
function formatNum(n, decimals = 0) {
    if (n === undefined || n === null) return '0';
    return Number(n).toLocaleString('zh-CN', {
        minimumFractionDigits: decimals,
        maximumFractionDigits: decimals,
    });
}

function setPnlColor(elId, val) {
    const el = document.getElementById(elId);
    if (val > 0) el.style.color = 'var(--profit)';
    else if (val < 0) el.style.color = 'var(--loss)';
    else el.style.color = 'var(--text-muted)';
}
