# 金融助手 Agent 强化学习环境 (Financial Assistant Agent RL Environment)

## 1. 项目简介

本项目旨在为一款集投资交易与个性化理财建议于一体的**金融助手 Agent** 设计并实现一个高性能、可扩展的强化学习 (RL) 环境。该环境遵循 **OpenAI Gymnasium** 标准接口，整合多模态数据输入（技术指标、宏观经济数据、新闻情感），并提供精细化的奖励机制和**课程学习 (Curriculum Learning)** 支持，以确保 Agent 能够在追求收益的同时有效管控风险。

## 2. 项目结构

```
financial_agent_rl/
├── env/
│   ├── __init__.py                # 包初始化
│   ├── financial_env.py           # 核心环境类 (基于 Gymnasium 标准接口)
│   ├── feature_engineering.py     # 特征工程模块 (支持宏观经济 + 新闻情感)
│   ├── reward.py                  # 奖励函数模块
│   ├── simulator.py               # 交易模拟引擎
│   ├── task_generator.py          # 任务生成器 (支持难度感知生成)
│   ├── scorer.py                  # 打分器：episode 级别综合评分
│   └── curriculum_scheduler.py    # 课程学习调度器 (NEW)
├── utils/
│   └── metrics.py                 # 评估指标
├── examples/
│   ├── run_random_agent.py        # 随机 Agent 示例
│   ├── run_training.py            # 使用 Stable-Baselines3 训练 PPO Agent 示例
│   ├── run_task_generator_demo.py # 演示任务生成器与打分器功能
│   └── run_enhanced_demo.py       # 演示增强功能：宏观经济 + 新闻 + 课程学习 (NEW)
├── data/                          # 存放数据
├── requirements.txt               # 依赖列表
└── README.md                      # 项目说明文档
```

## 3. 核心模块说明

### 3.1 基础模块

- **`env/financial_env.py`**: 实现了 `gymnasium.Env` 接口。整合特征工程、交易模拟、奖励计算、任务生成、综合打分和课程学习。
- **`env/feature_engineering.py`**: 从原始金融数据中提取技术指标（RSI、布林带等）、宏观经济特征和新闻情感特征。
- **`env/reward.py`**: 多目标复合奖励函数，考虑收益、波动率、交易成本和策略一致性。
- **`env/simulator.py`**: 模拟金融市场交易机制，包括持仓管理、佣金和滑点。
- **`env/task_generator.py`**: 为环境生成多样化任务，支持股票分析、投资组合管理和财务规划。
- **`env/scorer.py`**: Episode 结束时提供综合评分报告。

### 3.2 增强模块 (NEW)

- **`env/curriculum_scheduler.py`**: 课程学习调度器，根据 Agent 历史表现动态调整任务难度，实现渐进式训练。

## 4. 增强功能详解

### 4.1 丰富环境反馈

环境现在支持三种数据源的融合：

| 数据类型 | 特征 | 说明 |
| :--- | :--- | :--- |
| 技术指标 | log_return, volatility_5, volatility_20, sma_ratio, rsi_14, bb_width | 价格动量、均线、RSI、布林带 |
| 新闻情感 | sentiment_score, news_volume, sentiment_momentum | 新闻情感得分、新闻量、情感动量 |
| 宏观经济 | interest_rate_change, cpi_growth, unemployment_change | 利率变化、CPI 增长率、失业率变化 |

**启用方式：**

```python
from env.financial_env import FinancialAssistantEnv
from env.task_generator import (
    TaskGenerator, create_multi_asset_data,
    create_simulated_macro_data, create_simulated_news_data
)

# 创建包含宏观和新闻数据的任务生成器
data_dict = create_multi_asset_data(num_assets=3, days=500)
macro_data = create_simulated_macro_data(days=500)
news_data = create_simulated_news_data(days=500)

task_gen = TaskGenerator(
    data_dict=data_dict,
    macro_data=macro_data,
    news_data=news_data
)

# 启用宏观和新闻特征
env = FinancialAssistantEnv(
    task_generator=task_gen,
    include_macro=True,
    include_news=True
)
# 观测空间维度: 12 (市场特征) + 3 (账户特征) = 15
```

### 4.2 课程学习 (Curriculum Learning)

课程学习调度器根据 Agent 的表现自动调整任务难度：

**难度等级：**

| 难度范围 | 标签 | 特征 |
| :--- | :--- | :--- |
| 0.0 - 0.2 | Beginner | 牛市、低波动、低交易成本 |
| 0.2 - 0.4 | Easy | 牛市/震荡、中等波动 |
| 0.4 - 0.6 | Medium | 震荡市、标准参数 |
| 0.6 - 0.8 | Hard | 熊市、高波动、高交易成本 |
| 0.8 - 1.0 | Expert | 熊市、极端波动、复杂任务 |

**使用方式：**

```python
from env.curriculum_scheduler import CurriculumScheduler

# 创建调度器
scheduler = CurriculumScheduler(
    initial_difficulty=0.1,       # 从简单开始
    promotion_threshold=60.0,     # 平均得分 > 60 则升级
    demotion_threshold=20.0,      # 平均得分 < 20 则降级
    difficulty_step_up=0.1,       # 每次升级增加 0.1
    difficulty_step_down=0.05,    # 每次降级减少 0.05
    exploration_rate=0.1          # 10% 概率探索新难度
)

# 创建带课程学习的环境
env = FinancialAssistantEnv(
    task_generator=task_gen,
    curriculum_scheduler=scheduler,
    include_macro=True,
    include_news=True
)

# 训练循环中自动调整难度
for episode in range(100):
    obs, info = env.reset()
    done = False
    while not done:
        action = your_model.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

    # 课程学习在 episode 结束时自动更新难度
    if "curriculum_stats" in info:
        stats = info["curriculum_stats"]
        print(f"Current Difficulty: {stats['current_difficulty']:.2f} ({stats['difficulty_label']})")
```

### 4.3 难度感知任务生成

任务生成器现在支持 `target_difficulty` 参数，根据难度自动调整：

```python
# 生成特定难度的任务
easy_task = task_gen.generate_task(task_type="stock_analysis", target_difficulty=0.1)
hard_task = task_gen.generate_task(task_type="stock_analysis", target_difficulty=0.8)

# 难度影响的参数：市场类型偏好、时间窗口、交易成本、初始资金
```

## 5. 安装依赖

```bash
pip install -r requirements.txt
```

## 6. 快速开始

### 6.1 运行增强功能演示（推荐）

```bash
cd financial_agent_rl
python examples/run_enhanced_demo.py
```

此演示包含三个部分：
1. **丰富环境反馈演示**：展示宏观经济和新闻情感特征的集成
2. **课程学习演示**：展示 20 个 episode 中难度的动态调整
3. **难度感知任务生成演示**：展示不同难度等级的任务参数差异

### 6.2 运行任务生成器与打分器演示

```bash
cd financial_agent_rl
python examples/run_task_generator_demo.py
```

### 6.3 运行随机 Agent

```bash
cd financial_agent_rl
python examples/run_random_agent.py
```

### 6.4 训练 PPO Agent

```bash
cd financial_agent_rl
python examples/run_training.py
```

## 7. 如何接入自定义模型进行测试与训练

本环境遵循标准的 **Gymnasium** 接口，因此任何兼容 Gymnasium 的 RL 算法或自定义模型都可以轻松接入。

### 7.1 基本接入流程

```python
import sys
sys.path.append("path/to/financial_agent_rl")

from env.financial_env import FinancialAssistantEnv, generate_dummy_data
from env.task_generator import (
    TaskGenerator, create_multi_asset_data,
    create_simulated_macro_data, create_simulated_news_data
)
from env.curriculum_scheduler import CurriculumScheduler

# 方式一：使用固定数据（适合快速测试）
df = generate_dummy_data(num_days=500)
env = FinancialAssistantEnv(df=df, initial_balance=10000)

# 方式二：使用任务生成器 + 课程学习（推荐）
data_dict = create_multi_asset_data(num_assets=5, days=1000)
macro_data = create_simulated_macro_data(days=1000)
news_data = create_simulated_news_data(days=1000)

task_gen = TaskGenerator(
    data_dict=data_dict,
    macro_data=macro_data,
    news_data=news_data,
    risk_profiles=['conservative', 'moderate', 'aggressive']
)
scheduler = CurriculumScheduler(initial_difficulty=0.1)

env = FinancialAssistantEnv(
    task_generator=task_gen,
    curriculum_scheduler=scheduler,
    include_macro=True,
    include_news=True
)
```

### 7.2 接入自定义模型进行测试

```python
class MyCustomModel:
    def predict(self, observation):
        """
        输入: observation (numpy array) - 环境的观测向量
              启用所有特征时为 15 维:
              - 12 个市场特征 (技术指标 + 新闻情感 + 宏观经济)
              - 3 个账户特征 (balance, shares_held, net_worth)

        输出: action (numpy array, shape=(1,)) - 取值范围 [-1, 1]
        """
        import numpy as np
        rsi = observation[4]
        if rsi < -1.0:
            return np.array([0.5])
        elif rsi > 1.0:
            return np.array([-0.5])
        else:
            return np.array([0.0])

model = MyCustomModel()
obs, info = env.reset()
done = False
while not done:
    action = model.predict(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

if "score_report" in info:
    report = info["score_report"]
    print(f"综合评分: {report['final_score']}")
```

### 7.3 接入 Stable-Baselines3

```python
from stable_baselines3 import PPO, SAC

env = FinancialAssistantEnv(task_generator=task_gen, include_macro=True, include_news=True)

# PPO 训练
model = PPO("MlpPolicy", env, verbose=1, learning_rate=3e-4)
model.learn(total_timesteps=50000)
model.save("my_financial_agent")

# SAC 训练（推荐用于连续动作空间）
model = SAC("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=50000)
```

### 7.4 接入 LLM-based Agent

```python
def obs_to_prompt(obs, info):
    """将观测向量转换为自然语言提示"""
    prompt = f"""你是一个金融交易助手。请根据以下市场信息做出交易决策。

当前市场状态:
- 对数收益率: {obs[0]:.4f}
- 短期波动率(5日): {obs[1]:.4f}
- 长期波动率(20日): {obs[2]:.4f}
- 均线比率(SMA10/SMA30): {obs[3]:.4f}
- RSI(14): {obs[4]:.4f}
- 布林带宽度: {obs[5]:.4f}
- 市场情绪得分: {obs[6]:.4f}
- 新闻量: {obs[7]:.4f}
- 情绪动量: {obs[8]:.4f}
- 利率变化: {obs[9]:.4f}
- CPI增长率: {obs[10]:.4f}
- 失业率变化: {obs[11]:.4f}

账户状态:
- 可用余额: {obs[12]:.2f}
- 持仓数量: {obs[13]:.4f}
- 账户净值: {obs[14]:.2f}

请输出一个 -1 到 1 之间的数字作为交易决策。"""
    return prompt
```

### 7.5 环境接口速查表

| 接口 | 说明 | 输入/输出 |
| :--- | :--- | :--- |
| `env.reset()` | 重置环境 | 返回 `(observation, info)` |
| `env.reset(options={"task_type": "..."})` | 指定任务类型重置 | 支持 `stock_analysis`, `portfolio_management`, `financial_planning` |
| `env.step(action)` | 执行一步交易 | 输入 `action: np.array(shape=(1,), range=[-1,1])` |
| `env.observation_space` | 观测空间 | 基础: `Box(shape=(15,))` (12 市场 + 3 账户) |
| `env.action_space` | 动作空间 | `Box(shape=(1,), low=-1, high=1)` |
| `env.current_task_meta` | 当前任务元数据 | 包含 `task_type`, `market_type`, `difficulty_score` 等 |
| `info["score_report"]` | 评分报告 | 包含 `final_score`, `metrics` 等 |
| `info["curriculum_stats"]` | 课程学习统计 | 包含 `current_difficulty`, `promotions`, `demotions` 等 |

## 8. 数据说明

环境支持三种数据源：

```python
# 1. 股票行情数据（必需）
import yfinance as yf
data_dict = {}
for ticker in ["AAPL", "GOOGL", "MSFT"]:
    data_dict[ticker] = yf.download(ticker, start="2020-01-01", end="2024-12-31")

# 2. 宏观经济数据（可选）
# 可通过 FRED API 获取真实数据，或使用模拟数据
from env.task_generator import create_simulated_macro_data
macro_data = create_simulated_macro_data(days=1000)

# 3. 新闻情感数据（可选）
# 可通过 Finnhub/NewsAPI 获取真实数据，或使用模拟数据
from env.task_generator import create_simulated_news_data
news_data = create_simulated_news_data(days=1000)
```

## 9. 许可证

本项目采用 MIT 许可证。
