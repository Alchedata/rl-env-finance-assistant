"""
Task Generator — generates diverse, difficulty-aware tasks for the RL environment.
"""
import numpy as np
import pandas as pd

from utils.metrics import calculate_max_drawdown


def create_multi_asset_data(num_assets: int = 3, days: int = 500) -> dict:
    """
    Simulate correlated price series via Geometric Brownian Motion.

    Args:
        num_assets: Number of assets to simulate
        days: Number of trading days

    Returns:
        dict mapping asset names to DataFrames with OHLCV columns
    """
    dates = pd.bdate_range(start='2020-01-01', periods=days)
    data_dict = {}

    for i in range(num_assets):
        mu = np.random.uniform(0.0001, 0.0005)
        sigma = np.random.uniform(0.01, 0.025)
        S0 = np.random.uniform(50, 200)

        # GBM simulation
        returns = np.random.normal(mu, sigma, days)
        prices = np.zeros(days)
        prices[0] = S0
        for t in range(1, days):
            prices[t] = prices[t - 1] * np.exp(returns[t])

        # Generate OHLCV
        high = prices * (1 + np.abs(np.random.normal(0, 0.005, days)))
        low = prices * (1 - np.abs(np.random.normal(0, 0.005, days)))
        open_prices = prices * (1 + np.random.normal(0, 0.002, days))
        volume = np.random.lognormal(mean=15, sigma=1, size=days)

        df = pd.DataFrame({
            'open': open_prices,
            'high': high,
            'low': low,
            'close': prices,
            'volume': volume,
        }, index=dates)

        asset_name = f'ASSET_{i}'
        data_dict[asset_name] = df

    return data_dict


def create_simulated_macro_data(days: int = 500) -> pd.DataFrame:
    """
    Simulate macroeconomic data with smooth random walk.

    Args:
        days: Number of trading days

    Returns:
        DataFrame with interest_rate_change, cpi_growth, unemployment_change
    """
    dates = pd.bdate_range(start='2020-01-01', periods=days)

    # Interest rate change: smooth random walk
    ir_changes = np.random.normal(0, 0.001, days)
    ir_changes = pd.Series(ir_changes).rolling(5, min_periods=1).mean().values

    # CPI growth: monthly, forward-filled daily
    cpi_growth = np.random.normal(0.002, 0.001, days)
    cpi_growth = pd.Series(cpi_growth).rolling(22, min_periods=1).mean().values

    # Unemployment change: smooth
    unemp_changes = np.random.normal(0, 0.0005, days)
    unemp_changes = pd.Series(unemp_changes).rolling(10, min_periods=1).mean().values

    return pd.DataFrame({
        'interest_rate_change': ir_changes,
        'cpi_growth': cpi_growth,
        'unemployment_change': unemp_changes,
    }, index=dates)


def create_simulated_news_data(days: int = 500) -> pd.DataFrame:
    """
    Simulate news sentiment data.

    Args:
        days: Number of trading days

    Returns:
        DataFrame with sentiment_score, news_volume, sentiment_momentum
    """
    dates = pd.bdate_range(start='2020-01-01', periods=days)

    # Sentiment score: AR(1) process in [-1, 1]
    sentiment = np.zeros(days)
    sentiment[0] = np.random.uniform(-0.5, 0.5)
    rho = 0.95  # autoregressive coefficient
    for t in range(1, days):
        sentiment[t] = rho * sentiment[t - 1] + (1 - rho) * np.random.normal(0, 0.1)
    sentiment = np.clip(sentiment, -1, 1)

    # News volume: log-normal, normalized to [0, 1]
    raw_volume = np.random.lognormal(mean=0, sigma=1, size=days)
    news_volume = (raw_volume - raw_volume.min()) / (raw_volume.max() - raw_volume.min() + 1e-8)

    # Sentiment momentum: rolling 5-day mean
    sentiment_momentum = pd.Series(sentiment).rolling(5, min_periods=1).mean().values

    return pd.DataFrame({
        'sentiment_score': sentiment,
        'news_volume': news_volume,
        'sentiment_momentum': sentiment_momentum,
    }, index=dates)


def _detect_market_type(prices: np.ndarray) -> str:
    """
    Detect market type from price data.

    Args:
        prices: Array of close prices

    Returns:
        Market type string: bull, bear, volatile, crisis, or sideways
    """
    returns = np.diff(prices) / prices[:-1]
    mean_return = np.mean(returns)
    std_return = np.std(returns)

    # Compute max drawdown
    cummax = np.maximum.accumulate(prices)
    drawdowns = (cummax - prices) / cummax
    max_dd = np.max(drawdowns)

    if max_dd > 0.3:
        return 'crisis'
    elif std_return > 0.02:
        return 'volatile'
    elif mean_return > 0.0003:
        return 'bull'
    elif mean_return < -0.0003:
        return 'bear'
    else:
        return 'sideways'


class TaskGenerator:
    """Generates diverse, difficulty-aware tasks for the RL environment."""

    def __init__(
        self,
        data_dict: dict,
        min_window: int = 100,
        max_window: int = 300,
        initial_balance_range: tuple = (10000, 50000),
        risk_profiles: list = None,
        macro_data: pd.DataFrame = None,
        news_data: pd.DataFrame = None,
    ):
        self.data_dict = data_dict
        self.min_window = min_window
        self.max_window = max_window
        self.initial_balance_range = initial_balance_range
        self.risk_profiles = risk_profiles or ['moderate']
        self.macro_data = macro_data
        self.news_data = news_data
        self.asset_names = list(data_dict.keys())

    def generate_task(self, task_type: str = 'stock_analysis', target_difficulty: float = None) -> dict:
        """
        Generate a task with specified type and difficulty.

        Args:
            task_type: One of 'stock_analysis', 'portfolio_management', 'financial_planning'
            target_difficulty: Target difficulty score (0.0–1.0). If None, randomly sampled.

        Returns:
            Task dict with data_dict, meta, commission_rate, slippage_rate, initial_balance, etc.
        """
        if target_difficulty is None:
            target_difficulty = np.random.uniform(0.0, 1.0)

        # Select assets based on task type
        if task_type == 'stock_analysis':
            selected_assets = [np.random.choice(self.asset_names)]
        elif task_type == 'portfolio_management':
            n = np.random.randint(2, min(4, len(self.asset_names) + 1))
            selected_assets = list(np.random.choice(self.asset_names, size=min(n, len(self.asset_names)), replace=False))
        elif task_type == 'financial_planning':
            n = np.random.randint(1, min(3, len(self.asset_names) + 1))
            selected_assets = list(np.random.choice(self.asset_names, size=min(n, len(self.asset_names)), replace=False))
        else:
            selected_assets = [np.random.choice(self.asset_names)]

        # Determine window size (harder → shorter window)
        window_size = int(self.max_window - target_difficulty * (self.max_window - self.min_window))
        window_size = max(self.min_window, min(self.max_window, window_size))

        # Select a random time window from data
        asset_dfs = {}
        reference_asset = selected_assets[0]
        ref_df = self.data_dict[reference_asset]
        max_start = len(ref_df) - window_size
        if max_start <= 0:
            max_start = 1
        start_idx = np.random.randint(0, max_start)

        for asset_name in selected_assets:
            df = self.data_dict[asset_name]
            sliced = df.iloc[start_idx:start_idx + window_size].copy()
            asset_dfs[asset_name] = sliced

        # Detect market type from the reference asset
        ref_prices = asset_dfs[reference_asset]['close'].values
        market_type = _detect_market_type(ref_prices)

        # Difficulty-based parameters
        commission_rate, slippage_rate = self._get_cost_params(target_difficulty)

        # Initial balance
        initial_balance = np.random.uniform(*self.initial_balance_range)

        # Risk profile
        risk_profile = np.random.choice(self.risk_profiles)

        # Difficulty label
        if target_difficulty <= 0.3:
            difficulty_label = 'easy'
        elif target_difficulty <= 0.6:
            difficulty_label = 'medium'
        else:
            difficulty_label = 'hard'

        # Build description
        description = self._build_description(task_type, market_type, difficulty_label, risk_profile)

        # Slice macro/news data if available
        task_macro = None
        task_news = None
        start_date = asset_dfs[reference_asset].index[0]
        end_date = asset_dfs[reference_asset].index[-1]

        if self.macro_data is not None:
            mask = (self.macro_data.index >= start_date) & (self.macro_data.index <= end_date)
            task_macro = self.macro_data.loc[mask].copy() if mask.any() else None

        if self.news_data is not None:
            mask = (self.news_data.index >= start_date) & (self.news_data.index <= end_date)
            task_news = self.news_data.loc[mask].copy() if mask.any() else None

        # Build meta
        meta = {
            'description': description,
            'task_type': task_type,
            'market_type': market_type,
            'difficulty': difficulty_label,
            'difficulty_score': round(target_difficulty, 2),
            'start_date': start_date,
            'end_date': end_date,
            'window_size': window_size,
            'has_macro_data': task_macro is not None,
            'has_news_data': task_news is not None,
            'risk_profile': risk_profile,
        }

        # Financial planning specific
        if task_type == 'financial_planning':
            meta['current_age'] = np.random.randint(25, 56)
            meta['retirement_age'] = meta['current_age'] + np.random.randint(10, 31)

        return {
            'data_dict': asset_dfs,
            'meta': meta,
            'commission_rate': commission_rate,
            'slippage_rate': slippage_rate,
            'initial_balance': initial_balance,
            'macro_data': task_macro,
            'news_data': task_news,
        }

    def _get_cost_params(self, difficulty: float) -> tuple:
        """Get commission and slippage rates based on difficulty."""
        if difficulty <= 0.3:
            commission = np.random.uniform(0.0005, 0.001)
            slippage = np.random.uniform(0.0001, 0.0003)
        elif difficulty <= 0.6:
            commission = np.random.uniform(0.001, 0.002)
            slippage = np.random.uniform(0.0003, 0.0005)
        else:
            commission = np.random.uniform(0.002, 0.005)
            slippage = np.random.uniform(0.0005, 0.002)
        return commission, slippage

    def _build_description(self, task_type: str, market_type: str, difficulty: str, risk_profile: str) -> str:
        """Build a human-readable task description."""
        task_names = {
            'stock_analysis': 'Stock Analysis',
            'portfolio_management': 'Portfolio Management',
            'financial_planning': 'Financial Planning',
        }
        name = task_names.get(task_type, task_type)
        return f"{name} task in a {market_type} market ({difficulty} difficulty, {risk_profile} risk profile)"