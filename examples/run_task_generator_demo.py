import os
import sys
import pandas as pd
import numpy as np

# Add project root directory to python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.financial_env import FinancialAssistantEnv
from env.task_generator import TaskGenerator, create_multi_asset_data

def run_demo():
    print("=== Financial Assistant RL Env: Task Generator & Scorer Demo ===\n")

    # 1. Prepare multi-asset simulated data
    print("Generating simulated multi-asset data...")
    multi_asset_data = create_multi_asset_data(num_assets=5, days=1000)
    
    # 2. Initialize task generator
    task_gen = TaskGenerator(
        data_dict=multi_asset_data,
        min_window=100,
        max_window=300,
        initial_balance_range=(10000, 50000),
        risk_profiles=['conservative', 'moderate', 'aggressive']
    )

    # 3. Create environment with integrated task generator
    env = FinancialAssistantEnv(task_generator=task_gen)

    # 4. Run multiple episodes to showcase task diversity
    task_types = ["stock_analysis", "portfolio_management", "financial_planning"]
    num_episodes_per_type = 2

    for task_type in task_types:
        print(f"\n--- Task Type: {task_type.replace('_', ' ').title()} Demo ---")
        for episode in range(num_episodes_per_type):
            print(f"\n--- Episode {episode + 1} for {task_type} ---")
            
            # Resetting the environment automatically generates a new task of the specified type
            obs, info = env.reset(options={"task_type": task_type})
            meta = env.current_task_meta
            
            print(f"Task Description: {meta['description']}")
            print(f"Market Type: {meta['market_type']} (Difficulty: {meta['difficulty']})")
            print(f"Time Range: {meta['start_date'].date()} to {meta['end_date'].date()}")
            print(f"Initial Balance: {env.initial_balance:.2f}")
            print(f"Commission Rate: {env.simulator.commission_rate:.4f}")
            if 'risk_profile' in meta: print(f"Risk Profile: {meta['risk_profile']}")
            if 'retirement_age' in meta: print(f"Retirement Age: {meta['retirement_age']}, Current Age: {meta['current_age']}")
            
            done = False
            total_reward = 0
            while not done:
                # Simulate agent with random actions
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                done = terminated or truncated
                
            # End of episode, show score report
            print("\n[Episode End Score Report]")
            report = info["score_report"]
            print(f"Final Score: {report['final_score']}")
            print(f"Base Score: {report['base_score']:.2f} (Difficulty Multiplier: {report['difficulty_multiplier']})")
            print("Detailed Metrics:")
            for k, v in report['metrics'].items():
                print(f"  - {k}: {v}")

    env.close()

if __name__ == "__main__":
    run_demo()
