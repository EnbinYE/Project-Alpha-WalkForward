import ccxt
import pandas as pd
import numpy as np
import talib as ta
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

# =====================================================================
# 1. 数据管道 (Data Pipeline) - 压榨 M1 Max 性能，拉取 1000 条 1小时线
# =====================================================================
print("正在从交易所拉取高频数据...")
exchange = ccxt.binance()
# 切换到 1h（1小时线），让样本量暴增，机器学习更好学
bars = exchange.fetch_ohlcv('BTC/USDT', timeframe='1h', limit=1000)
df = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
df.set_index('timestamp', inplace=True)

# =====================================================================
# 2. 特征工程 (Feature Engineering) - 包含特征标准化
# =====================================================================
print("正在构建特征矩阵与标签...")
# 传统指标
df['feat_rsi'] = ta.RSI(df['close'], timeperiod=14)
# 特征标准化：将绝对值的 MACD 除以当时收盘价，转化为变动百分比
macd_raw, _, _ = ta.MACD(df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
df['feat_macd'] = macd_raw / df['close'] 

# 微观结构与时序动量
df['feat_hl_pct'] = (df['high'] - df['low']) / df['close']
df['feat_return_5d'] = df['close'].pct_change(periods=5)
df['feat_vol_10d'] = df['close'].rolling(window=10).std() / df['close']

# 构建标签：预测下一小时是涨(1)还是跌(0)
df['next_day_return'] = df['close'].pct_change().shift(-1) # 下一期的实际收益率
df['target'] = (df['next_day_return'] > 0).astype(int)

# 清理由于滚动产生的空值
df.dropna(inplace=True)

# =====================================================================
# 3. 工业级固定窗口滚动训练 (Fixed Window Rolling Validation)
# =====================================================================
print("开始固定窗口滚动训练与概率过滤...")
feature_cols = ['feat_rsi', 'feat_macd', 'feat_hl_pct', 'feat_return_5d', 'feat_vol_10d']
X = df[feature_cols].values
y = df['target'].values
test_returns = df['next_day_return'].values

train_window = 400  # 永远只用最近的 400 个小时训练
test_window = 50    # 每次向前预测接下来的 50 个小时

all_y_pred = np.zeros(len(y))
test_indices = []

# 开始滚动车轮
for start_idx in range(train_window, len(y) - test_window, test_window):
    # 动态切分训练集（窗口保持固定长度，最老的样本会被丢弃）
    X_train = X[start_idx - train_window : start_idx]
    y_train = y[start_idx - train_window : start_idx]
    
    # 动态切分测试集
    X_test = X[start_idx : start_idx + test_window]
    
    # 训练模型
    model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
    model.fit(X_train, y_train)
    
    # 获取预测概率
    proba = model.predict_proba(X_test)[:, 1]
    
    # 【核心防御】：概率过滤逻辑。模型有54%以上把握涨才买(1)，否则持币观望(0)
    y_pred = (proba > 0.54).astype(int)
    
    # 记录本轮测试集的索引和信号
    current_test_idx = list(range(start_idx, start_idx + test_window))
    all_y_pred[current_test_idx] = y_pred
    test_indices.extend(current_test_idx)

# 【核心解答】：在这里统一把 final_dates 和 final_test_returns 求出来！
test_indices = np.array(test_indices)
final_y_test = y[test_indices]
final_y_pred = all_y_pred[test_indices]
final_test_returns = test_returns[test_indices]  # 对应测试集区间的实际大盘收益率
final_dates = df.index[test_indices]             # 对应测试集区间的日期时间戳

# =====================================================================
# 4. 风控模块 (Risk Control) - 引入日内硬性止损
# =====================================================================
print("开启风控模块，注入硬性止损...")
# 设定硬性止损线：-1.5% (因为变成了1小时线，波动变小，止损线收紧到 -1.5%)
stop_loss_threshold = -0.015
controlled_strategy_returns = np.zeros(len(final_test_returns))
stop_loss_count = 0

for i in range(len(final_test_returns)):
    pred_signal = final_y_pred[i]
    actual_market_return = final_test_returns[i]
    
    if pred_signal == 1: # 满仓持有
        if actual_market_return <= stop_loss_threshold:
            controlled_strategy_returns[i] = stop_loss_threshold
            stop_loss_count += 1
        else:
            controlled_strategy_returns[i] = actual_market_return
    else: # 空仓观望
        controlled_strategy_returns[i] = 0

print(f">>> 风控战报：在测试区间内，策略共触发了 {stop_loss_count} 次硬性止损断臂求生！")

# =====================================================================
# 5. 收益计算与可视化 (Visualization)
# =====================================================================
# 计算累计收益
crypto_cum_return = np.cumprod(1 + final_test_returns)
old_strategy_cum_return = np.cumprod(1 + (final_y_pred * final_test_returns))
new_strategy_cum_return = np.cumprod(1 + controlled_strategy_returns)

# 绘图
plt.figure(figsize=(12, 6))
plt.plot(final_dates, crypto_cum_return, label='Buy & Hold BTC (基准灰线)', color='gray', linestyle='--')
plt.plot(final_dates, old_strategy_cum_return, label='No-Stop Strategy (无风控旧红线)', color='red', alpha=0.5)
plt.plot(final_dates, new_strategy_cum_return, label='Stop-Loss Strategy (加止损新蓝线)', color='blue', linewidth=2)
plt.title('Complete AI Quant System: Fixed Window + Proba Filter + Stop Loss')
plt.xlabel('Time')
plt.ylabel('Equity')
plt.legend()
plt.grid(True)
plt.show()