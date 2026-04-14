# Financial Assistant Agent RL Environment

## 1. Project Overview

This project designs and implements a high-performance, extensible Reinforcement Learning (RL) environment for a **Financial Assistant Agent** that integrates investment trading and personalized financial advice. The environment follows the **OpenAI Gymnasium** standard interface, integrates multimodal data inputs (technical indicators, macro-economic data, news sentiment), and provides a fine-grained multi-objective reward mechanism and **Curriculum Learning** support to ensure the Agent effectively manages risks while pursuing returns.

**Core Design Decisions:**
- **Action Space:** Portfolio weight vector \`Box(0, 1, shape=(n_assets+1,))\`, normalized via internal softmax.
- **Observation Space:** A sliding window of \`W\` time steps, flattened into a 1D vector, compatible with SB3 \`MlpPolicy\`.

## 2. Project Structure

\`\`\`
rl-env-finance-assistant/
├── env/
│   ├── __init__.py                # Package initialization and public exports
│   ├── financial_env.py           # Core environment class (Gymnasium standard interface)
│   ├── feature_engineering.py     # Feature engineering (Technical indicators + Macro + News sentiment)
│   ├── reward.py                  # Multi-objective composite reward function
│   ├── simulator.py               # Trading simulation engine (Commission + Slippage)
│   ├── task_generator.py          # Task generator (Difficulty-aware + Multi-asset)
│   ├── scorer.py                  # Episode comprehensive scorer
│   └── curriculum_scheduler.py    # Curriculum learning scheduler
├── utils/
│   └── metrics.py                 # Evaluation metrics (Returns, Sharpe ratio, Max drawdown, etc.)
├── examples/
│   ├── run_random_agent.py        # Random Agent example
│   ├── run_training.py            # Stable-Baselines3 PPO training example
│   ├── run_task_generator_demo.py # Task generator and scorer demo
│   └── run_enhanced_demo.py       # Enhanced features demo (Macro + News + Curriculum Learning)
├── data/                          # Data directory
├── requirements.txt               # Dependency list
├── ImplementationPlan.md          # Implementation plan (Detailed module specifications)
└── README.md
\`\`\`

## 3. Architecture & Data Flow

\`\`\`
simulator.py   feature_engineering.py   reward.py   scorer.py
      │                  │                  │            │
      └──────────────────┴──────────────────┴────────────┘
                          │
                 task_generator.py   curriculum_scheduler.py
                          │                  │
                          └──────────────────┘
                          │
                   financial_env.py
                          │
                      __init__.py
\`\`\`

**Per-step Data Flow (\`step()\`):**
1. Agent outputs raw action → Softmax normalization → Portfolio weights
2. \`TradingSimulator.rebalance(weights, prices)\` → Execute trades, return transaction costs
3. \`FeatureEngineering.compute_features()\` → Observation feature row
4. \`RewardCalculator.compute()\` → Scalar reward
5. At end of Episode: \`EpisodeScorer.score()\` → Score report → \`CurriculumScheduler.record_episode()\`

## 4. Core Modules Description

| Module | Class | Responsibility |
|------|-----|------|
| \`simulator.py\` | \`TradingSimulator\` | Simulates trading execution: position management, commission, slippage |
| \`feature_engineering.py\` | \`FeatureEngineering\` | Extracts technical indicators, macro features, and news sentiment features |
| \`reward.py\` | \`RewardCalculator\` | Multi-objective composite reward: Return − Volatility − Cost − Drawdown |
| \`scorer.py\` | \`EpisodeScorer\` | Episode-level comprehensive score (0–100), including difficulty factor |
| \`task_generator.py\` | \`TaskGenerator\` | Generates diverse tasks with difficulty awareness and three task types |
| \`curriculum_scheduler.py\` | \`CurriculumScheduler\` | Dynamically adjusts task difficulty based on Agent performance |
| \`financial_env.py\` | \`FinancialAssistantEnv\` | Main Gymnasium environment class, integrating all above modules |

## 5. Environment Interface

### 5.1 Action Space

\`\`\`python
action_space = Box(low=0, high=1, shape=(n_assets+1,), dtype=np.float32)
\`\`\`

The Agent outputs a raw weight vector \`[cash_w, asset1_w, asset2_w, ...]\`, which is normalized into portfolio ratios via internal softmax. In single-asset mode, \`shape=(2,)\` (Cash + 1 Stock).

### 5.2 Observation Space

\`\`\`python
observation_space = Box(low=-inf, high=inf, shape=(window_size × n_features,), dtype=np.float32)
\`\`\`

Sliding window observation, flattened into a 1D vector for SB3 \`MlpPolicy\` compatibility. Feature dimensions:

| Feature Category | Dim per Asset | Description |
|----------|-----------|------|
| Technical Indicators | 6 | \`log_return\`, \`volatility_5\`, \`volatility_20\`, \`sma_ratio\`, \`rsi_14\`, \`bb_width\` |
| Macro Economics | 3 (Global) | \`interest_rate_change\`, \`cpi_growth\`, \`unemployment_change\` |
| News Sentiment | 3 (Global) | \`sentiment_score\`, \`news_volume\`, \`sentiment_momentum\` |
| Portfolio State | n_assets+1 | Holding ratios for each asset + Cash ratio |
| Step Progress | 1 | Current Step / Total Steps |

\`\`\`
n_features = max_assets × 6 + (max_assets + 1) + 1 + (3 if macro) + (3 if news)
\`\`\`

### 5.3 Interface Quick Reference

| Interface | Description | Input/Output |
|------|------|-----------|
| \`env.reset()\` | Resets environment | Returns \`(observation, info)\` |
| \`env.reset(options={"task_type": "..."})\` | Reset with specific task type | Supports \`stock_analysis\`, \`portfolio_management\`, \`financial_planning\` |
| \`env.step(action)\` | Executes one trading step | Input \`action: np.array(shape=(n_assets+1,), range=[0,1])\` |
| \`env.observation_space\` | Observation space | \`Box(shape=(window_size × n_features,))\` |
| \`env.action_space\` | Action space | \`Box(shape=(n_assets+1,), low=0, high=1)\` |
| \`env.simulator\` | Trading simulator | \`TradingSimulator\` instance |
| \`env.current_task_meta\` | Current task metadata | Contains \`task_type\`, \`market_type\`, \`difficulty_score\`, etc. |
| \`info["score_report"]\` | Score report | Contains \`final_score\`, \`metrics\`, \`market_info\`, etc. |

## 6. Features in Detail

### 6.1 Multimodal Data Fusion

The environment supports the fusion of three data sources:

| Data Type | Features | Description |
|----------|------|------|
| Technical Indicators | \`log_return\`, \`volatility_5\`, \`volatility_20\`, \`sma_ratio\`, \`rsi_14\`, \`bb_width\` | Price momentum, volatility, moving average, RSI, Bollinger Bands |
| News Sentiment | \`sentiment_score\`, \`news_volume\`, \`sentiment_momentum\` | Sentiment score, news volume, sentiment momentum |
| Macro Economics | \`interest_rate_change\`, \`cpi_growth\`, \`unemployment_change\` | Interest rate change, CPI growth, unemployment rate change |

**Usage:**

\`\`\`python
from env.financial_env import FinancialAssistantEnv
from env.task_generator import (
    TaskGenerator, create_multi_asset_data,
    create_simulated_macro_data, create_simulated_news_data
)

data_dict = create_multi_asset_data(num_assets=3, days=500)
macro_data = create_simulated_macro_data(days=500)
news_data = create_simulated_news_data(days=500)

task_gen = TaskGenerator(
    data_dict=data_dict,
    macro_data=macro_data,
    news_data=news_data
)

env = FinancialAssistantEnv(
    task_generator=task_gen,
    include_macro=True,
    include_news=True
)
\`\`\`

### 6.2 Curriculum Learning

The curriculum learning scheduler automatically adjusts task difficulty based on Agent performance:

| Difficulty Range | Label | Characteristics |
|----------|------|------|
| 0.0 – 0.2 | Beginner | Bull market, low volatility, low transaction costs |
| 0.2 – 0.4 | Easy | Bull/Side-ways market, moderate volatility |
| 0.4 – 0.6 | Medium | Side-ways market, standard parameters |
| 0.6 – 0.8 | Hard | Bear market, high volatility, high transaction costs |
| 0.8 – 1.0 | Expert | Bear market, extreme volatility, complex tasks |

**Usage:**

\`\`\`python
from env.curriculum_scheduler import CurriculumScheduler

scheduler = CurriculumScheduler(
    initial_difficulty=0.1,
    promotion_threshold=50.0,
    demotion_threshold=10.0,
    difficulty_step_up=0.1,
    difficulty_step_down=0.05,
    exploration_rate=0.1
)

env = FinancialAssistantEnv(
    task_generator=task_gen,
    curriculum_scheduler=scheduler,
    include_macro=True,
    include_news=True
)

# Difficulty automatically updates within the training loop at episode end
for episode in range(100):
    obs, info = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

    # Curriculum updates difficulty automatically; you can check stats
    stats = scheduler.get_stats()
    print(f"Difficulty: {stats['current_difficulty']:.2f} ({stats['difficulty_label']})")
\`\`\`

### 6.3 Difficulty-Aware Task Generation

The task generator supports the \`target_difficulty\` parameter to automatically adjust market type, transaction costs, and time windows:

\`\`\`python
easy_task = task_gen.generate_task(task_type="stock_analysis", target_difficulty=0.1)
hard_task = task_gen.generate_task(task_type="stock_analysis", target_difficulty=0.8)
\`\`\`

**Three Task Types:**

| Task Type | Asset Count | Additional Parameters |
|----------|---------|----------|
| \`stock_analysis\` | 1 | \`risk_profile\` |
| \`portfolio_management\` | 2–3 | \`risk_profile\` |
| \`financial_planning\` | 1–2 | \`risk_profile\`, \`current_age\`, \`retirement_age\` |

### 6.4 Reward Mechanism

Multi-objective composite reward function to balance return and risk:

\`\`\`
reward = w₁ · step_return − w₂ · volatility_penalty − w₃ · transaction_cost_penalty − w₄ · drawdown_penalty
\`\`\`

| Component | Formula | Purpose |
|------|------|------|
| \`step_return\` | \`(curr − prev) / prev\` | Reward positive returns |
| \`volatility_penalty\` | \`std(last 20 returns)\` | Penalize instability |
| \`transaction_cost_penalty\` | \`transaction_cost / prev_net_worth\` | Discourage over-trading |
| \`drawdown_penalty\` | \`max(0, (peak − curr) / peak)\` | Penalize drawdown from peak |

## 7. Installation & Quick Start

### 7.1 Install Dependencies

\`\`\`bash
pip install -r requirements.txt
\`\`\`

### 7.2 Run Enhanced Feature Demo (Recommended)

\`\`\`bash
python examples/run_enhanced_demo.py
\`\`\`

Includes three parts:
1. **Multimodal Data Fusion Demo**: Integration of macro and news sentiment features.
2. **Curriculum Learning Demo**: Dynamic difficulty adjustment over 20 episodes.
3. **Difficulty-Aware Task Generation Demo**: Task parameter differences across difficulty levels.

### 7.3 Other Examples

\`\`\`bash
python examples/run_task_generator_demo.py    # Task generator and scorer
python examples/run_random_agent.py           # Random Agent baseline
python examples/run_training.py              # SB3 PPO training
\`\`\`

## 8. Integrating Custom Models

This environment follows the standard **Gymnasium** interface, so any Gymnasium-compatible RL algorithm can be used.

### 8.1 Basic Integration

\`\`\`python
from env.financial_env import FinancialAssistantEnv, generate_dummy_data
from env.task_generator import (
    TaskGenerator, create_multi_asset_data,
    create_simulated_macro_data, create_simulated_news_data
)
from env.curriculum_scheduler import CurriculumScheduler

# Option 1: Fixed data (suitable for quick testing)
df = generate_dummy_data(num_days=500)
env = FinancialAssistantEnv(df=df, initial_balance=10000)

# Option 2: Task Generator + Curriculum Learning (Recommended)
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
\`\`\`

### 8.2 Testing Custom Models

\`\`\`python
import numpy as np

class MyCustomModel:
    def predict(self, observation):
        """
        Input: observation — Flattened sliding window feature vector
              shape = (window_size × n_features,)
        Output: action — Portfolio weight vector
              shape = (n_assets+1,), range = [0, 1]
              The environment handles softmax normalization internally.
        """
        # Simple example: Equal weighting strategy
        n = env.action_space.shape[0]
        return np.ones(n) / n

model = MyCustomModel()
obs, info = env.reset()
done = False
while not done:
    action = model.predict(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

if "score_report" in info:
    report = info["score_report"]
    print(f"Comprehensive Score: {report['final_score']}")
\`\`\`

### 8.3 Stable-Baselines3

\`\`\`python
from stable_baselines3 import PPO, SAC

env = FinancialAssistantEnv(task_generator=task_gen, include_macro=True, include_news=True)

# PPO Training
model = PPO("MlpPolicy", env, verbose=1, learning_rate=3e-4)
model.learn(total_timesteps=50000)
model.save("my_financial_agent")

# SAC Training (Recommended for continuous action spaces)
model = SAC("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=50000)
\`\`\`

### 8.4 LLM-based Agent

\`\`\`python
def obs_to_prompt(obs, info):
    """Convert observation vector to a natural language prompt"""
    # Note: obs is a flattened window; parse according to n_features.
    # Here is an example mapping features for a single asset, window_size=1.
    prompt = f"""You are a financial trading assistant. Please make a portfolio decision based on the following market info.

Current Market Features (6 technical indicators per asset):
- Log Return (log_return)
- Short-term Volatility (volatility_5)
- Long-term Volatility (volatility_20)
- moving Average Ratio (sma_ratio)
- RSI(14)
- Bollinger Band Width (bb_width)

Portfolio State:
- Cash Ratio
- Asset Holding Ratios
- Step Progress

Please output a portfolio weight vector [cash_weight, asset1_weight, ...],
where each value is in [0, 1]. No normalization required (the environment handles softmax)."""
    return prompt
\`\`\`

## 9. Data Description

The environment supports three data sources:

\`\`\`python
# 1. Stock Market Data (Required)
import yfinance as yf
data_dict = {}
for ticker in ["AAPL", "GOOGL", "MSFT"]:
    data_dict[ticker] = yf.download(ticker, start="2020-01-01", end="2024-12-31")

# 2. Macro-economic Data (Optional)
from env.task_generator import create_simulated_macro_data
macro_data = create_simulated_macro_data(days=1000)

# 3. News Sentiment Data (Optional)
from env.task_generator import create_simulated_news_data
news_data = create_simulated_news_data(days=1000)
\`\`\`

## 10. License

This project is licensed under the MIT License.
