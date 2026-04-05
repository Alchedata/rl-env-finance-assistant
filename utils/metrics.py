import numpy as np
import pandas as pd

def calculate_cumulative_return(net_worth_history):
    """
    计算累计收益率
    Calculate Cumulative Return
    """
    if not net_worth_history or len(net_worth_history) < 2:
        return 0.0
    initial_value = net_worth_history[0]
    final_value = net_worth_history[-1]
    return (final_value - initial_value) / initial_value

def calculate_max_drawdown(net_worth_history):
    """
    计算最大回撤
    Calculate Maximum Drawdown
    """
    if not net_worth_history or len(net_worth_history) < 2:
        return 0.0
    
    # 将净值历史转换为 pandas Series 以便计算
    net_worth_series = pd.Series(net_worth_history)
    # 计算累计最大值
    rolling_max = net_worth_series.cummax()
    # 计算回撤
    drawdown = (rolling_max - net_worth_series) / rolling_max
    # 返回最大回撤
    return drawdown.max()

def calculate_sharpe_ratio(returns_history, risk_free_rate=0.0):
    """
    计算夏普比率
    Calculate Sharpe Ratio
    Args:
        returns_history (list or np.array): 每日收益率列表
        risk_free_rate (float): 无风险利率 (年化)
    """
    if not returns_history or len(returns_history) < 2:
        return 0.0
    
    returns = np.array(returns_history)
    # 假设是日收益率，年化因子为 sqrt(252)
    annualization_factor = np.sqrt(252)
    
    # 计算超额收益
    excess_returns = returns - (risk_free_rate / annualization_factor)
    
    # 计算超额收益的平均值和标准差
    avg_excess_return = np.mean(excess_returns)
    std_excess_return = np.std(excess_returns)
    
    if std_excess_return == 0:
        return 0.0
        
    return annualization_factor * (avg_excess_return / std_excess_return)

def calculate_win_rate(trades_info):
    """
    计算交易胜率
    Calculate Win Rate
    Args:
        trades_info (list): 包含每笔交易盈亏的列表。正数表示盈利，负数表示亏损。
    """
    if not trades_info:
        return 0.0
    
    profitable_trades = sum(1 for pnl in trades_info if pnl > 0)
    total_trades = len(trades_info)
    
    return profitable_trades / total_trades if total_trades > 0 else 0.0


if __name__ == '__main__':
    # 示例用法
    net_worth_example = [10000, 10100, 10050, 10200, 9900, 10500, 10300]
    returns_example = [0.01, -0.005, 0.015, -0.029, 0.06, -0.019]
    trades_pnl_example = [50, -20, 100, -10, 30]

    print(f"Cumulative Return: {calculate_cumulative_return(net_worth_example):.4f}")
    print(f"Max Drawdown: {calculate_max_drawdown(net_worth_example):.4f}")
    print(f"Sharpe Ratio (daily): {calculate_sharpe_ratio(returns_example):.4f}")
    print(f"Win Rate: {calculate_win_rate(trades_pnl_example):.4f}")

    net_worth_single = [10000]
    returns_single = [0.01]
    trades_empty = []
    print(f"\nSingle point net worth Cumulative Return: {calculate_cumulative_return(net_worth_single):.4f}")
    print(f"Single point returns Sharpe Ratio: {calculate_sharpe_ratio(returns_single):.4f}")
    print(f"Empty trades Win Rate: {calculate_win_rate(trades_empty):.4f}")
