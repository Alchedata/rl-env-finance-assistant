"""
Financial Assistant Environment — Gymnasium-compatible RL environment for financial assistant agents.
"""
import numpy as np
import pandas as pd
import gymnasium
from gymnasium import spaces

from env.simulator import TradingSimulator
from env.feature_engineering import FeatureEngineering
from env.reward import RewardCalculator
from env.scorer import EpisodeScorer
from env.task_generator import TaskGenerator


def _softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def generate_dummy_data(num_days: int = 1260) -> pd.DataFrame:
    """
    Generate a single GBM price series for simple single-asset mode.

    Args:
        num_days: Number of trading days

    Returns:
        DataFrame with open, high, low, close, volume columns
    """
    mu = 0.0002
    sigma = 0.015
    S0 = 100.0

    dates = pd.bdate_range(start='2020-01-01', periods=num_days)
    returns = np.random.normal(mu, sigma, num_days)
    prices = np.zeros(num_days)
    prices[0] = S0
    for t in range(1, num_days):
        prices[t] = prices[t - 1] * np.exp(returns[t])

    high = prices * (1 + np.abs(np.random.normal(0, 0.005, num_days)))
    low = prices * (1 - np.abs(np.random.normal(0, 0.005, num_days)))
    open_prices = prices * (1 + np.random.normal(0, 0.002, num_days))
    volume = np.random.lognormal(mean=15, sigma=1, size=num_days)

    return pd.DataFrame({
        'open': open_prices,
        'high': high,
        'low': low,
        'close': prices,
        'volume': volume,
    }, index=dates)


class FinancialAssistantEnv(gymnasium.Env):
    """Gymnasium environment for financial assistant agent training."""

    metadata = {'render_modes': ['human']}

    def __init__(
        self,
        df: pd.DataFrame = None,
        initial_balance: float = 10000,
        render_mode: str = None,
        task_generator: TaskGenerator = None,
        curriculum_scheduler=None,
        include_macro: bool = False,
        include_news: bool = False,
        window_size: int = 20,
        max_assets: int = None,
    ):
        super().__init__()

        self.render_mode = render_mode
        self.initial_balance = initial_balance
        self.task_generator = task_generator
        self.curriculum_scheduler = curriculum_scheduler
        self.include_macro = include_macro
        self.include_news = include_news
        self.window_size = window_size

        # Store single-asset df if provided
        self._single_df = df

        # Determine max_assets
        if max_assets is not None:
            self.max_assets = max_assets
        elif task_generator is not None:
            self.max_assets = len(task_generator.asset_names)
        else:
            self.max_assets = 1

        # Compute feature dimensions
        self.n_technical = self.max_assets * 6  # 6 technical features per asset
        self.n_portfolio = self.max_assets + 1   # position ratios (cash + assets)
        self.n_progress = 1                      # step progress
        self.n_macro = 3 if include_macro else 0
        self.n_news = 3 if include_news else 0
        self.n_features = self.n_technical + self.n_portfolio + self.n_progress + self.n_macro + self.n_news

        # Define spaces
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.window_size * self.n_features,),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=0, high=1,
            shape=(self.max_assets + 1,),
            dtype=np.float32,
        )

        # Initialize components
        self.simulator = TradingSimulator()
        self.feature_engineering = FeatureEngineering(
            include_macro=include_macro,
            include_news=include_news,
        )
        self.reward_calculator = RewardCalculator()
        self.scorer = EpisodeScorer()

        # Episode state (initialized properly in reset)
        self.current_step = 0
        self.current_task_meta = {}
        self._asset_dfs = {}
        self._feature_dfs = {}
        self._asset_names = []
        self._returns_history = []
        self._net_worth_history = []
        self._max_net_worth = 0.0
        self._total_steps = 0
        self._window_buffer = None
        self._task_macro = None
        self._task_news = None

    def reset(self, seed=None, options=None):
        """Reset the environment."""
        super().reset(seed=seed)

        if seed is not None:
            np.random.seed(seed)

        task_type = 'stock_analysis'
        if options and 'task_type' in options:
            task_type = options['task_type']

        # Generate or use task
        if self.task_generator is not None:
            target_difficulty = None
            if self.curriculum_scheduler is not None:
                target_difficulty = self.curriculum_scheduler.current_difficulty
            task = self.task_generator.generate_task(
                task_type=task_type,
                target_difficulty=target_difficulty,
            )
            self._asset_dfs = task['data_dict']
            self.current_task_meta = task['meta']
            self.initial_balance = task['initial_balance']
            self._asset_names = list(self._asset_dfs.keys())

            # Reset simulator with task params
            self.simulator = TradingSimulator(
                commission_rate=task['commission_rate'],
                slippage_rate=task['slippage_rate'],
            )
            self.simulator.reset(self.initial_balance, self._asset_names)

            # Store macro/news data for feature engineering
            self._task_macro = task.get('macro_data')
            self._task_news = task.get('news_data')
        else:
            # Single-asset mode
            if self._single_df is None:
                self._single_df = generate_dummy_data()
            self._asset_dfs = {'ASSET_0': self._single_df.copy()}
            self._asset_names = ['ASSET_0']
            self.current_task_meta = {
                'task_type': task_type,
                'market_type': 'unknown',
                'difficulty': 'medium',
                'difficulty_score': 0.5,
                'window_size': len(self._single_df),
                'has_macro_data': False,
                'has_news_data': False,
                'risk_profile': 'moderate',
            }
            self.simulator = TradingSimulator()
            self.simulator.reset(self.initial_balance, self._asset_names)
            self._task_macro = None
            self._task_news = None

        # Compute features for all assets
        self._feature_dfs = {}
        for asset_name, df in self._asset_dfs.items():
            features = self.feature_engineering.compute_features(
                df,
                macro_data=self._task_macro,
                news_data=self._task_news,
            )
            self._feature_dfs[asset_name] = features

        # Determine total steps from reference asset
        ref_asset = self._asset_names[0]
        self._total_steps = len(self._asset_dfs[ref_asset])

        # Initialize episode state
        self.current_step = self.window_size  # start after warm-up
        self._returns_history = []
        self._net_worth_history = [self.initial_balance]
        self._max_net_worth = self.initial_balance
        self._window_buffer = None  # will be initialized in _build_observation

        # Build initial observation
        obs = self._build_observation()
        info = {
            'net_worth': self.simulator.net_worth,
            'step': self.current_step,
        }

        return obs, info

    def step(self, action):
        """Execute one step in the environment."""
        # Softmax normalize action → portfolio weights
        weights = _softmax(action.astype(np.float64))

        # Get current prices
        prices = self._get_current_prices()

        # Execute rebalance
        prev_net_worth = self.simulator.net_worth
        transaction_cost = self.simulator.rebalance(weights, prices)
        curr_net_worth = self.simulator.net_worth

        # Track returns
        if prev_net_worth > 0:
            step_return = (curr_net_worth - prev_net_worth) / prev_net_worth
        else:
            step_return = 0.0
        self._returns_history.append(step_return)
        self._net_worth_history.append(curr_net_worth)
        self._max_net_worth = max(self._max_net_worth, curr_net_worth)

        # Compute reward
        reward = self.reward_calculator.compute(
            prev_net_worth=prev_net_worth,
            curr_net_worth=curr_net_worth,
            transaction_cost=transaction_cost,
            returns_history=self._returns_history,
            max_net_worth=self._max_net_worth,
        )

        # Advance step
        self.current_step += 1

        # Check termination
        terminated = self.current_step >= self._total_steps

        # Build observation
        obs = self._build_observation()

        # Build info
        info = {
            'net_worth': curr_net_worth,
            'step': self.current_step,
            'transaction_cost': transaction_cost,
        }

        # Episode end: score and curriculum update
        if terminated:
            score_report = self.scorer.score(
                net_worth_history=self._net_worth_history,
                returns_history=self._returns_history,
                trades_count=self.simulator.trades_count,
                task_meta=self.current_task_meta,
                difficulty=self.current_task_meta.get('difficulty_score', 0.5),
            )
            info['score_report'] = score_report

            if self.curriculum_scheduler is not None:
                self.curriculum_scheduler.record_episode(score_report['final_score'])
                info['curriculum_stats'] = self.curriculum_scheduler.get_stats()

        return obs, reward, terminated, False, info

    def _get_current_prices(self) -> dict:
        """Get current prices for all assets."""
        prices = {}
        for asset_name, df in self._asset_dfs.items():
            if self.current_step < len(df):
                prices[asset_name] = float(df.iloc[self.current_step]['close'])
            else:
                prices[asset_name] = float(df.iloc[-1]['close'])
        return prices

    def _build_observation(self) -> np.ndarray:
        """Build the sliding window observation vector."""
        # Build feature vector for current step
        feature_vector = self._build_feature_vector()

        # Initialize window buffer if needed
        if self._window_buffer is None:
            self._window_buffer = np.zeros((self.window_size, self.n_features), dtype=np.float32)

        # Slide window: append new row, drop oldest
        self._window_buffer = np.roll(self._window_buffer, -1, axis=0)
        self._window_buffer[-1] = feature_vector

        # Flatten and return
        obs = self._window_buffer.flatten().astype(np.float32)

        # Clip for safety
        obs = np.clip(obs, -1e6, 1e6)
        obs = np.nan_to_num(obs, nan=0.0, posinf=1e6, neginf=-1e6)

        return obs

    def _build_feature_vector(self) -> np.ndarray:
        """Build the feature vector for the current step."""
        features = []

        # Technical features for each asset (pad to max_assets)
        for i in range(self.max_assets):
            if i < len(self._asset_names):
                asset_name = self._asset_names[i]
                feat_df = self._feature_dfs[asset_name]
                if self.current_step < len(feat_df):
                    tech_cols = ['log_return', 'volatility_5', 'volatility_20', 'sma_ratio', 'rsi_14', 'bb_width']
                    row = feat_df.iloc[self.current_step]
                    for col in tech_cols:
                        val = row.get(col, 0.0)
                        features.append(float(val) if not pd.isna(val) else 0.0)
                else:
                    features.extend([0.0] * 6)
            else:
                # Padding for missing assets
                features.extend([0.0] * 6)

        # Portfolio state: position ratios
        prices = self._get_current_prices()
        ratios = self.simulator.get_position_ratios(prices)

        # Pad ratios to max_assets + 1
        while len(ratios) < self.max_assets + 1:
            ratios = np.append(ratios, 0.0)
        features.extend(ratios[:self.max_assets + 1].tolist())

        # Step progress
        progress = self.current_step / max(self._total_steps, 1)
        features.append(progress)

        # Macro features
        if self.include_macro:
            ref_asset = self._asset_names[0]
            feat_df = self._feature_dfs[ref_asset]
            if self.current_step < len(feat_df):
                row = feat_df.iloc[self.current_step]
                for col in ['interest_rate_change', 'cpi_growth', 'unemployment_change']:
                    val = row.get(col, 0.0)
                    features.append(float(val) if not pd.isna(val) else 0.0)
            else:
                features.extend([0.0] * 3)

        # News features
        if self.include_news:
            ref_asset = self._asset_names[0]
            feat_df = self._feature_dfs[ref_asset]
            if self.current_step < len(feat_df):
                row = feat_df.iloc[self.current_step]
                for col in ['sentiment_score', 'news_volume', 'sentiment_momentum']:
                    val = row.get(col, 0.0)
                    features.append(float(val) if not pd.isna(val) else 0.0)
            else:
                features.extend([0.0] * 3)

        vec = np.array(features, dtype=np.float32)

        # Replace inf/nan
        vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)

        return vec