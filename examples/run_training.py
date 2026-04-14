import gymnasium as gym
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold

from env.financial_env import FinancialAssistantEnv, generate_dummy_data
from utils.metrics import calculate_cumulative_return, calculate_max_drawdown, calculate_sharpe_ratio

def train_ppo_agent(env_id, total_timesteps=100000, eval_freq=10000, n_eval_episodes=5):
    """
    Example of training a PPO Agent using Stable-Baselines3.
    """
    print("\n--- Training PPO Agent Example ---")

    # 1. Create environment
    # For make_vec_env, a function that returns an environment instance is required
    def make_env():
        dummy_df = generate_dummy_data(num_days=252 * 5) # 5 years of data
        return FinancialAssistantEnv(df=dummy_df, initial_balance=10000)

    vec_env = make_vec_env(make_env, n_envs=1)

    # 2. Define callback functions (optional)
    # Stop training when average reward reaches a threshold
    callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=100, verbose=1)
    eval_callback = EvalCallback(vec_env, best_model_save_path="./logs/best_model",
                                 log_path="./logs/", eval_freq=eval_freq,
                                 n_eval_episodes=n_eval_episodes, 
                                 callback_on_new_best=callback_on_best, verbose=1)

    # 3. Initialize PPO model
    model = PPO("MlpPolicy", vec_env, verbose=1, tensorboard_log="./ppo_financial_tensorboard/")

    # 4. Train the model
    model.learn(total_timesteps=total_timesteps, callback=eval_callback)

    # 5. Save the model
    model.save("ppo_financial_agent")
    print("PPO model saved as ppo_financial_agent.zip")

    print("\n--- PPO Agent Training Finished ---")

    # 6. Load and test the model (optional)
    print("\n--- Testing Trained PPO Agent ---")
    loaded_model = PPO.load("ppo_financial_agent")

    obs = vec_env.reset()
    done = False
    total_reward = 0
    episode_net_worth_history = [vec_env.get_attr("initial_balance")[0]]
    episode_returns_history = []

    while not done:
        action, _states = loaded_model.predict(obs, deterministic=True)
        prev_net_worth = vec_env.get_attr("simulator")[0].net_worth
        obs, rewards, dones, infos = vec_env.step(action)
        total_reward += rewards[0]
        
        current_net_worth = vec_env.get_attr("simulator")[0].net_worth
        episode_net_worth_history.append(current_net_worth)
        if prev_net_worth != 0:
            episode_returns_history.append((current_net_worth - prev_net_worth) / prev_net_worth)
        else:
            episode_returns_history.append(0.0)

        if dones[0]:
            done = True

    print(f"  Test Episode Total Reward: {total_reward:.2f}")
    print(f"  Test Episode Final Net Worth: {vec_env.get_attr('simulator')[0].net_worth:.2f}")
    print(f"  Test Episode Cumulative Return: {calculate_cumulative_return(episode_net_worth_history):.4f}")
    print(f"  Test Episode Max Drawdown: {calculate_max_drawdown(episode_net_worth_history):.4f}")
    print(f"  Test Episode Sharpe Ratio (daily): {calculate_sharpe_ratio(episode_returns_history):.4f}")
    print("--- Testing Finished ---")

    vec_env.close()

if __name__ == '__main__':
    train_ppo_agent(env_id="FinancialAssistantEnv-v0", total_timesteps=10000)
