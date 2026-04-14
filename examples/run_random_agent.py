import gymnasium as gym
import numpy as np
import pandas as pd

from env.financial_env import FinancialAssistantEnv, generate_dummy_data
from utils.metrics import calculate_cumulative_return, calculate_max_drawdown, calculate_sharpe_ratio

def run_random_agent(env, num_episodes=5):
    """
    Runs a random agent example to demonstrate environment usage.
    """
    print("\n--- Running Random Agent Example ---")
    
    all_episode_returns = []
    all_episode_net_worths = []

    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0
        episode_net_worth_history = [env.initial_balance]
        episode_returns_history = []

        print(f"\nEpisode {episode + 1}/{num_episodes}")
        while not done:
            action = env.action_space.sample() # Random action
            prev_net_worth = env.simulator.net_worth
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            current_net_worth = env.simulator.net_worth
            episode_net_worth_history.append(current_net_worth)
            if prev_net_worth != 0:
                episode_returns_history.append((current_net_worth - prev_net_worth) / prev_net_worth)
            else:
                episode_returns_history.append(0.0)

            # env.render() # Uncomment to enable rendering

            if terminated or truncated:
                done = True
        
        all_episode_returns.extend(episode_returns_history)
        all_episode_net_worths.extend(episode_net_worth_history)

        print(f"  Episode finished. Total Reward: {total_reward:.2f}")
        print(f"  Final Net Worth: {env.simulator.net_worth:.2f}")
        print(f"  Trades Count: {env.simulator.trades_count}")
        print(f"  Total Commission: {env.simulator.total_commission:.2f}")
        print(f"  Total Slippage: {env.simulator.total_slippage:.2f}")
        print(f"  Cumulative Return: {calculate_cumulative_return(episode_net_worth_history):.4f}")
        print(f"  Max Drawdown: {calculate_max_drawdown(episode_net_worth_history):.4f}")
        print(f"  Sharpe Ratio (daily): {calculate_sharpe_ratio(episode_returns_history):.4f}")

    print("\n--- Random Agent Example Finished ---")

if __name__ == '__main__':
    # 1. 生成模拟数据
    # Generate dummy data
    dummy_df = generate_dummy_data(num_days=252 * 5) # 5年数据

    # 2. 初始化环境
    # Initialize environment
    env = FinancialAssistantEnv(df=dummy_df, initial_balance=10000, render_mode=None)

    # 3. 运行随机 Agent
    # Run random agent
    run_random_agent(env, num_episodes=3)

    # 4. 关闭环境
    # Close environment
    env.close()
