"""
Reward Calculator — multi-objective composite reward function.
"""
import numpy as np


class RewardCalculator:
    """Computes step reward balancing returns, volatility, transaction costs, and drawdowns."""

    def __init__(
        self,
        step_return_weight: float = 1.0,
        volatility_penalty_weight: float = 0.1,
        transaction_cost_penalty_weight: float = 0.5,
        drawdown_penalty_weight: float = 0.2,
    ):
        self.step_return_weight = step_return_weight
        self.volatility_penalty_weight = volatility_penalty_weight
        self.transaction_cost_penalty_weight = transaction_cost_penalty_weight
        self.drawdown_penalty_weight = drawdown_penalty_weight

    def compute(
        self,
        prev_net_worth: float,
        curr_net_worth: float,
        transaction_cost: float,
        returns_history: list,
        max_net_worth: float,
    ) -> float:
        """
        Compute the step reward.

        Args:
            prev_net_worth: Portfolio value at previous step
            curr_net_worth: Portfolio value at current step
            transaction_cost: Total transaction cost this step
            returns_history: List of recent step returns for volatility
            max_net_worth: Peak portfolio value for drawdown penalty

        Returns:
            Scalar reward value
        """
        # Step return
        if prev_net_worth > 0:
            step_return = (curr_net_worth - prev_net_worth) / prev_net_worth
        else:
            step_return = 0.0

        # Volatility penalty
        if len(returns_history) >= 2:
            volatility_penalty = float(np.std(returns_history[-20:]))
        else:
            volatility_penalty = 0.0

        # Transaction cost penalty
        if prev_net_worth > 0:
            transaction_cost_penalty = transaction_cost / prev_net_worth
        else:
            transaction_cost_penalty = 0.0

        # Drawdown penalty
        if max_net_worth > 0:
            drawdown_penalty = max(0.0, (max_net_worth - curr_net_worth) / max_net_worth)
        else:
            drawdown_penalty = 0.0

        reward = (
            self.step_return_weight * step_return
            - self.volatility_penalty_weight * volatility_penalty
            - self.transaction_cost_penalty_weight * transaction_cost_penalty
            - self.drawdown_penalty_weight * drawdown_penalty
        )

        return float(reward)