# 金融助手 Agent 强化学习环境 (Financial Assistant Agent RL Environment)

## 1. 项目简介

本项目旨在为一款集投资交易与个性化理财建议于一体的**金融助手 Agent** 设计并实现一个高性能、可扩展的强化学习 (RL) 环境。该环境遵循 **OpenAI Gymnasium** 标准接口，整合多模态数据输入，并提供精细化的奖励机制，以确保 Agent 能够在追求收益的同时有效管控风险。

## 2. 项目结构

```
financial_agent_rl/
├── env/
│   ├── financial_env.py      # 核心环境类 (基于 Gymnasium 标准接口)
│   ├── feature_engineering.py  # 特征工程模块
│   ├── reward.py             # 奖励函数模块
│   ├── simulator.py          # 交易模拟引擎
│   ├── task_generator.py     # [NEW] 任务生成器：支持多资产、多情景、多任务类型（股票分析、投资组合、财务规划）随机任务生成
│   └── scorer.py             # [NEW] 打分器：提供 episode 级别的综合评分报告，支持不同任务类型和风险偏好下的评分标准
├── utils/
│   └── metrics.py            # 评估指标
├── examples/
│   ├── run_random_agent.py   # 随机 Agent 示例
│   ├── run_training.py       # 使用 Stable-Baselines3 训练 PPO Agent 示例
│   └── run_task_generator_demo.py # [NEW] 演示任务生成器与打分器功能
├── data/                     # 存放数据
├── requirements.txt          # 依赖列表
└── README.md                 # 项目说明文档
```

## 3. 核心模块说明

-   **`env/financial_env.py`**: 实现了 `gymnasium.Env` 接口。它整合了特征工程、交易模拟、奖励计算、任务生成和综合打分。
-   **`env/task_generator.py`**: **(新功能)** 负责为环境生成多样化的任务。支持从不同资产中采样、随机时间窗口、识别市场情景（牛市、熊市、震荡市）以及环境参数（初始资金、费率）的随机化。**新增支持股票分析、投资组合管理（如 401k 资产配置）和财务规划任务的生成，并包含明确的任务描述和元数据。**
-   **`env/scorer.py`**: **(新功能)** 在 episode 结束时提供综合评分。评分维度包括累计收益、夏普比率、最大回撤和风险控制能力，并支持根据任务难度进行分数归一化。**新增针对股票分析、投资组合管理和财务规划任务的特定评分标准，例如预测准确率、风险分散度、是否符合用户风险偏好和退休目标达成率。**
-   **`env/feature_engineering.py`**: 负责从原始金融数据中提取技术指标和模拟情感特征。
-   **`env/reward.py`**: 定义了多目标复合奖励函数，考虑收益、波动率和交易成本。
-   **`env/simulator.py`**: 模拟金融市场交易机制，包括持仓管理、佣金和滑点。

## 4. 安装依赖

在运行项目之前，请确保您的 Python 环境已安装所有必要的依赖：

```bash
pip install -r requirements.txt
```

## 5. 运行示例

### 5.1 运行任务生成器与打分器演示 (推荐)

演示如何在每次 `reset` 时自动生成不同难度的交易任务，并在结束时获取详细的评分报告。此示例将展示股票分析、投资组合管理和财务规划三种任务类型。

```bash
python examples/run_task_generator_demo.py
```

### 5.2 运行随机 Agent

```bash
python examples/run_random_agent.py
```

### 5.3 训练 PPO Agent

```bash
python examples/run_training.py
```

## 6. 数据说明

目前环境支持通过 `TaskGenerator` 自动管理多资产数据。您可以通过 `create_multi_asset_data` 生成模拟数据，或接入真实的 `yfinance` 数据字典。

## 7. 许可证

本项目采用 MIT 许可证。
