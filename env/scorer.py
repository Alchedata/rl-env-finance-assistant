"""
Episode Scorer — comprehensive episode-level scoring with difficulty scaling.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.metrics import calculate_cumulative_return, calculate_max_drawdown, calculate_sharpe_ratio


class EpisodeScorer:
    """Computes comprehensive score report at episode end."""

    @staticmethod
    def score(
        net_worth_history: list,
        returns_history: list,
        trades_count: int,
        task_meta: dict,
        difficulty: float = 0.5,
    ) -> dict:
        """
        Score an episode and return a comprehensive report.

        Args:
            net_worth_history: List of net worth values over the episode
            returns_history: List of step returns
            trades_count: Number of trades executed
            task_meta: Task metadata from TaskGenerator
            difficulty: Difficulty score (0.0–1.0)

        Returns:
            score_report dict
        """
        # Compute metrics
        total_return = calculate_cumulative_return(net_worth_history)
        sharpe_ratio = calculate_sharpe_ratio(returns_history)
        max_drawdown = calculate_max_drawdown(net_worth_history)

        # Win rate: fraction of positive step returns
        if len(returns_history) > 0:
            win_rate = sum(1 for r in returns_history if r > 0) / len(returns_history)
        else:
            win_rate = 0.0

        # Base score formula
        base_score = 50.0
        base_score += 30.0 * max(-1.0, min(1.0, total_return))  # return contribution (±30)
        base_score += 20.0 * max(-1.0, min(1.0, sharpe_ratio / 3.0))  # risk-adjusted (±20)
        base_score -= 15.0 * max_drawdown  # drawdown penalty (up to -15)
        base_score = max(0.0, min(100.0, base_score))

        # Difficulty multiplier
        difficulty_multiplier = 1.0 + difficulty * 0.5
        final_score = max(0.0, min(100.0, base_score * difficulty_multiplier))

        # Build report
        report = {
            'final_score': round(final_score, 2),
            'base_score': round(base_score, 2),
            'difficulty_multiplier': round(difficulty_multiplier, 2),
            'task_type': task_meta.get('task_type', 'unknown'),
            'metrics': {
                'total_return': round(total_return, 4),
                'sharpe_ratio': round(sharpe_ratio, 4),
                'max_drawdown': round(max_drawdown, 4),
                'win_rate': round(win_rate, 4),
                'trades_count': trades_count,
            },
            'market_info': {
                'market_type': task_meta.get('market_type', 'unknown'),
                'difficulty_score': task_meta.get('difficulty_score', difficulty),
                'window_size': task_meta.get('window_size', 0),
                'has_macro_data': task_meta.get('has_macro_data', False),
                'has_news_data': task_meta.get('has_news_data', False),
            },
        }

        return report