# 金融助手 Agent 强化学习环境

## 1. 项目简介

本项目旨在为一款集投资交易与个性化理财建议于一体的**金融助手 Agent** 设计并实现一个高性能、可扩展的强化学习 (RL) 环境。该环境遵循 **OpenAI Gymnasium** 标准接口，整合多模态数据输入，并提供精细化的奖励机制，以确保 Agent 能够在追求收益的同时有效管控风险。

## 2. 项目结构

```
financial_agent_rl/
├── env/
│   ├── financial_env.py      # 核心环境类 (基于 Gymnasium 标准接口)
│   ├── feature_engineering.py  # 特征工程模块
│   ├── reward.py             # 奖励函数模块
│   └── simulator.py          # 交易模拟引擎
├── utils/
│   └── metrics.py            # 评估指标
├── examples/
│   ├── run_random_agent.py   # 随机 Agent 示例，演示环境使用
│   └── run_training.py       # 使用 Stable-Baselines3 训练 PPO Agent 的示例
├── data/                     # 存放数据 (例如：历史行情数据)
├── requirements.txt          # 依赖列表
└── README.md                 # 项目说明文档
```

## 3. 核心模块说明

-   **`env/financial_env.py`**: 实现了 `gymnasium.Env` 接口的金融强化学习环境。它整合了特征工程、交易模拟和奖励计算，为 Agent 提供观测、执行动作并接收奖励的接口。
-   **`env/feature_engineering.py`**: 负责从原始金融数据中提取各种技术指标（如移动平均线、RSI、布林带）和模拟情感特征，并对特征进行标准化处理。
-   **`env/reward.py`**: 定义了多目标复合奖励函数，考虑了收益、波动率、交易成本和策略一致性，以引导 Agent 学习稳健的交易策略。
-   **`env/simulator.py`**: 模拟了金融市场的交易机制，包括初始资金、持仓管理、交易佣金和滑点效应，以提供一个真实的交易环境。
-   **`utils/metrics.py`**: 包含用于评估 Agent 性能的多种金融指标，如累计收益率、最大回撤、夏普比率和交易胜率。

## 4. 安装依赖

在运行项目之前，请确保您的 Python 环境已安装所有必要的依赖。建议使用 `pip` 安装：

```bash
pip install -r requirements.txt
```

## 5. 运行示例

### 5.1 运行随机 Agent

此示例演示了如何初始化环境并让一个随机 Agent 在其中进行交互。这有助于验证环境是否正常工作。

```bash
python examples/run_random_agent.py
```

### 5.2 训练 PPO Agent

此示例展示了如何使用 `Stable-Baselines3` 库训练一个 PPO (Proximal Policy Optimization) Agent。训练过程将输出日志，并保存训练好的模型。

```bash
python examples/run_training.py
```

## 6. 数据说明

目前环境使用 `generate_dummy_data` 函数生成模拟的 OHLCV 数据。在实际应用中，您可以替换为真实的金融市场数据，例如通过 `yfinance` 或其他数据源获取。

## 7. 贡献

欢迎对本项目提出改进意见或贡献代码。请遵循以下步骤：

1.  Fork 本仓库。
2.  创建新的功能分支 (`git checkout -b feature/AmazingFeature`)。
3.  提交您的更改 (`git commit -m 'Add some AmazingFeature'`)。
4.  推送到分支 (`git push origin feature/AmazingFeature`)。
5.  打开 Pull Request。

## 8. 许可证

本项目采用 MIT 许可证。详情请参阅 `LICENSE` 文件 (如果存在)。
