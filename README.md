# ML-Quant-WalkForward-System
> 一个基于随机森林（Random Forest）与固定窗口滚动训练（Fixed-Window Rolling Validation）的智能量化交易回测框架。

本项目针对金融时间序列（Financial Time Series）的**非平稳性**与**概念漂移（Concept Drift）**痛点，搭建了完整的“数据清洗 -> 动态因子组合 -> 概率信号过滤 -> 动态尾部风控”的量化闭环。

---

## 🚀 核心架构设计 (System Architecture)

针对学术界 AI 模型在量化实盘中常见的过拟合与高回撤问题，本项目在工程上实现了三层防御机制：
1. **时序防御（Anti-Data-Leakage）**：拒绝传统的全局随机洗牌（Shuffle Split），采用滚动窗口确保训练集在时间戳上绝对领先于测试集，从根本上杜绝未来函数（Look-ahead Bias）。
2. **概率防御（Noise Filtering）**：金融数据信噪比极低，模型放弃预测绝对价格，转为预测涨跌概率。引入非对称阈值（如 `Probability > 54%`），在市场方向模糊时强制空仓观望。
3. **尾部风控（Fat-Tail Protection）**：针对金融收益率的非正态分布与极端暴跌风险，在执行层注入日内硬性止损锁（Stop-Loss Trigger），强制截断下行风险。

---

## 📊 数据与特征工程 (Features & Labels)

### 1. 输入特征 (X)
所有与价格绝对值相关的传统指标均经过**无量纲标准化（Standardization）**处理，除以当期收盘价，转化为相对变动率，以适应时序非平稳性。

| 特征名称 | 分类 | 描述 / 标准化公式 |
| :--- | :--- | :--- |
| `feat_rsi` | 动量指标 | 14周期相对强弱指标（天然无量纲） |
| `feat_macd` | 趋势指标 | 原始 MACD 差离值 / $\text{Close}_t$ （百分比化） |
| `feat_hl_pct` | 波动率指标 | $(\text{High}_t - \text{Low}_t) / \text{Close}_t$ （日内振幅） |
| `feat_return_5d` | 动量指标 | 过去 5 个周期的累计资产收益率 |
| `feat_vol_10d` | 风险指标 | 过去 10 个周期收盘价标准差 / $\text{Close}_t$ |

### 2. 预测标签 (y)
* **Target**: $\mathbb{I}(\text{Return}_{t+1} > 0)$。即预测下一周期的收益率是否大于零（二分类任务）。

---

## 🛠️ 环境依赖与运行 (Installation & Usage)

### 环境依赖 (M1/M2/M3 Mac 原生支持)
推荐使用 `Miniforge (conda-forge)` 构建原生 ARM64 Python 环境：

```bash
# 安装原生 TA-Lib C语言库 (Mac 环境必选)
brew install ta-lib

# 安装 Python 依赖
pip install ccxt pandas numpy ta-lib matplotlib scikit-learn
