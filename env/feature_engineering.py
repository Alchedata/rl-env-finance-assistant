"""
Feature Engineering — extracts technical, macro, and news features from financial data.
"""
import numpy as np
import pandas as pd


class FeatureEngineering:
    """Extracts technical indicators, macro features, and news sentiment features."""

    def __init__(self, include_macro: bool = False, include_news: bool = False):
        self.include_macro = include_macro
        self.include_news = include_news

    def compute_features(
        self,
        df: pd.DataFrame,
        macro_data: pd.DataFrame = None,
        news_data: pd.DataFrame = None,
    ) -> pd.DataFrame:
        """
        Compute all features from price data and optional macro/news data.

        Args:
            df: DataFrame with at least a 'close' column
            macro_data: DataFrame with macro columns (if include_macro)
            news_data: DataFrame with news columns (if include_news)

        Returns:
            DataFrame with all feature columns, same index as df
        """
        result = pd.DataFrame(index=df.index)

        # Technical features
        close = df['close']

        # log_return
        result['log_return'] = np.log(close / close.shift(1))

        # volatility_5
        result['volatility_5'] = result['log_return'].rolling(5).std()

        # volatility_20
        result['volatility_20'] = result['log_return'].rolling(20).std()

        # sma_ratio
        sma_20 = close.rolling(20).mean()
        result['sma_ratio'] = close / sma_20

        # rsi_14 (manual implementation)
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.inf)
        result['rsi_14'] = 100 - (100 / (1 + rs))

        # bb_width
        sma20 = close.rolling(20).mean()
        std20 = close.rolling(20).std()
        upper = sma20 + 2 * std20
        lower = sma20 - 2 * std20
        result['bb_width'] = (upper - lower) / sma20.replace(0, np.inf)

        # Macro features
        if self.include_macro and macro_data is not None:
            macro_aligned = macro_data.reindex(df.index)
            macro_aligned = macro_aligned.ffill().bfill().fillna(0)
            for col in ['interest_rate_change', 'cpi_growth', 'unemployment_change']:
                if col in macro_aligned.columns:
                    result[col] = macro_aligned[col]
                else:
                    result[col] = 0.0

        # News features
        if self.include_news and news_data is not None:
            news_aligned = news_data.reindex(df.index)
            news_aligned = news_aligned.ffill().bfill().fillna(0)
            for col in ['sentiment_score', 'news_volume', 'sentiment_momentum']:
                if col in news_aligned.columns:
                    result[col] = news_aligned[col]
                else:
                    result[col] = 0.0

        # NaN handling: ffill → bfill → fillna(0)
        result = result.ffill().bfill().fillna(0)

        # Replace inf/nan
        result = result.replace([np.inf, -np.inf], 0.0)
        result = result.fillna(0.0)

        return result