"""
Trading Simulator — simulates market trading with commission and slippage.
"""
import numpy as np


class TradingSimulator:
    """Simulates financial market trading with position management, commission, and slippage."""

    def __init__(self, commission_rate: float = 0.001, slippage_rate: float = 0.0005):
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate
        self.cash: float = 0.0
        self.positions: dict[str, float] = {}
        self.net_worth: float = 0.0
        self.trades_count: int = 0
        self.total_commission: float = 0.0
        self.total_slippage: float = 0.0

    def reset(self, initial_balance: float, asset_names: list[str]) -> None:
        """Reset the simulator with initial balance and asset names."""
        self.cash = initial_balance
        self.positions = {name: 0.0 for name in asset_names}
        self.net_worth = initial_balance
        self.trades_count = 0
        self.total_commission = 0.0
        self.total_slippage = 0.0

    def rebalance(self, target_weights: np.ndarray, prices: dict[str, float]) -> float:
        """
        Rebalance portfolio to target weights.

        Args:
            target_weights: softmax-normalized weights [cash_w, asset1_w, ...]
            prices: {asset_name: current_price}

        Returns:
            Total transaction cost (commission + slippage)
        """
        portfolio_value = self.get_portfolio_value(prices)
        asset_names = list(self.positions.keys())

        total_cost = 0.0

        # Compute target dollar allocation
        # target_weights[0] = cash weight, target_weights[1:] = asset weights
        for i, asset_name in enumerate(asset_names):
            target_value = portfolio_value * target_weights[i + 1]
            current_value = self.positions[asset_name] * prices[asset_name]
            delta_value = target_value - current_value

            if abs(delta_value) < 1e-6:
                continue

            price = prices[asset_name]
            if price <= 0:
                continue

            # Compute shares delta
            shares_delta = delta_value / price

            # Apply slippage to execution price
            slippage = price * self.slippage_rate * np.sign(shares_delta)
            exec_price = price + slippage

            # Actual trade value
            trade_value = shares_delta * exec_price

            # Apply commission
            commission = abs(trade_value) * self.commission_rate

            # Update cash
            self.cash -= trade_value + commission

            # Update position
            self.positions[asset_name] += shares_delta

            # Track stats
            self.trades_count += 1
            self.total_commission += commission
            self.total_slippage += abs(slippage * shares_delta)
            total_cost += commission + abs(slippage * shares_delta)

        # Recompute net worth
        self.net_worth = self.get_portfolio_value(prices)

        return total_cost

    def get_portfolio_value(self, prices: dict[str, float]) -> float:
        """Compute total portfolio value (cash + positions)."""
        value = self.cash
        for asset_name, shares in self.positions.items():
            value += shares * prices.get(asset_name, 0.0)
        return value

    def get_position_ratios(self, prices: dict[str, float]) -> np.ndarray:
        """
        Compute position ratios [cash_ratio, asset1_ratio, ...] summing to ~1.

        Returns:
            np.ndarray of shape (n_assets + 1,)
        """
        portfolio_value = self.get_portfolio_value(prices)
        if portfolio_value <= 0:
            n = len(self.positions) + 1
            return np.ones(n) / n

        ratios = [self.cash / portfolio_value]
        for asset_name in self.positions:
            asset_value = self.positions[asset_name] * prices.get(asset_name, 0.0)
            ratios.append(asset_value / portfolio_value)

        return np.array(ratios, dtype=np.float64)