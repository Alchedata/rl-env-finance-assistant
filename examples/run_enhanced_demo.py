"""
Enhanced Demo: Demonstrates rich environment feedback + curriculum learning features.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from env.financial_env import FinancialAssistantEnv
from env.task_generator import (
    TaskGenerator, create_multi_asset_data,
    create_simulated_macro_data, create_simulated_news_data
)
from env.curriculum_scheduler import CurriculumScheduler


def run_single_episode(env, task_type="stock_analysis"):
    """Run an episode and return results"""
    obs, info = env.reset(options={"task_type": task_type})
    done = False
    total_reward = 0
    steps = 0

    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        if terminated or truncated:
            done = True

    return total_reward, steps, info


def demo_rich_feedback():
    """Demonstrates rich environment feedback (macro economics + news sentiment)"""
    print("=" * 70)
    print("Demo 1: Rich Environment Feedback (Macro + News)")
    print("=" * 70)

    # Create simulated data
    data_dict = create_multi_asset_data(num_assets=3, days=500)
    macro_data = create_simulated_macro_data(days=500)
    news_data = create_simulated_news_data(days=500)

    # Create task generator with macro and news data
    task_gen = TaskGenerator(
        data_dict=data_dict,
        risk_profiles=['conservative', 'moderate', 'aggressive'],
        macro_data=macro_data,
        news_data=news_data
    )

    # Create environment (enabling macro and news features)
    env = FinancialAssistantEnv(
        task_generator=task_gen,
        include_macro=True,
        include_news=True
    )

    print(f"\nObservation space: {env.observation_space.shape}")
    print(f"Action space: {env.action_space.shape}")

    # Run different types of tasks
    for task_type in ["stock_analysis", "portfolio_management", "financial_planning"]:
        print(f"\n--- Task Type: {task_type} ---")
        total_reward, steps, info = run_single_episode(env, task_type)

        print(f"  Steps: {steps}")
        print(f"  Total Reward: {total_reward:.4f}")
        print(f"  Final Net Worth: {info['net_worth']:.2f}")

        if "score_report" in info:
            report = info["score_report"]
            print(f"  Final Score: {report['final_score']}")
            print(f"  Task Type: {report['task_type']}")
            print(f"  Total Return: {report['metrics']['total_return']}")
            print(f"  Sharpe Ratio: {report['metrics']['sharpe_ratio']}")
            print(f"  Max Drawdown: {report['metrics']['max_drawdown']}")

            if report.get("market_info"):
                meta = report["market_info"]
                print(f"  Market Type: {meta.get('market_type', 'N/A')}")
                print(f"  Difficulty Score: {meta.get('difficulty_score', 'N/A')}")
                print(f"  Has Macro Data: {meta.get('has_macro_data', False)}")
                print(f"  Has News Data: {meta.get('has_news_data', False)}")

    env.close()
    print("\nDemo 1 Complete!\n")


def demo_curriculum_learning():
    """Demonstrates curriculum learning functionality"""
    print("=" * 70)
    print("Demo 2: Curriculum Learning")
    print("=" * 70)

    # Create data
    data_dict = create_multi_asset_data(num_assets=3, days=500)
    macro_data = create_simulated_macro_data(days=500)
    news_data = create_simulated_news_data(days=500)

    task_gen = TaskGenerator(
        data_dict=data_dict,
        risk_profiles=['conservative', 'moderate', 'aggressive'],
        macro_data=macro_data,
        news_data=news_data
    )

    # Create curriculum learning scheduler
    scheduler = CurriculumScheduler(
        initial_difficulty=0.1,
        promotion_threshold=50.0,
        demotion_threshold=10.0,
        difficulty_step_up=0.1,
        difficulty_step_down=0.05,
        history_window=5,
        exploration_rate=0.1
    )

    # Create environment with curriculum learning
    env = FinancialAssistantEnv(
        task_generator=task_gen,
        curriculum_scheduler=scheduler,
        include_macro=True,
        include_news=True
    )

    print(f"\nInitial difficulty: {scheduler.current_difficulty:.2f} ({scheduler.get_difficulty_label()})")
    print(f"Promotion threshold: {scheduler.promotion_threshold}")
    print(f"Demotion threshold: {scheduler.demotion_threshold}")
    print(f"\nRunning 20 episodes with curriculum learning...\n")

    for episode in range(20):
        total_reward, steps, info = run_single_episode(env, "stock_analysis")

        score = 0
        if "score_report" in info:
            score = info["score_report"]["final_score"]

        stats = scheduler.get_stats()
        print(f"Episode {episode + 1:2d} | "
              f"Difficulty: {stats['current_difficulty']:.2f} ({stats['difficulty_label']:>8s}) | "
              f"Score: {score:6.2f} | "
              f"Reward: {total_reward:8.4f} | "
              f"Promotions: {stats['promotions']} | "
              f"Demotions: {stats['demotions']} | "
              f"Avg Score: {stats['recent_avg_score']:6.2f}")

    env.close()
    print("\nDemo 2 Complete!\n")


def demo_difficulty_task_generation():
    """Demonstrates difficulty-aware task parameters generation"""
    print("=" * 70)
    print("Demo 3: Difficulty-Aware Task Parameters Generation")
    print("=" * 70)

    data_dict = create_multi_asset_data(num_assets=3, days=1000)
    task_gen = TaskGenerator(data_dict=data_dict)

    difficulties = [0.1, 0.4, 0.7, 0.95]
    
    print(f"{'Difficulty':<12} | {'Market Type':<12} | {'Slippage':<10} | {'Commission':<10} | {'Window':<8}")
    print("-" * 70)

    for diff in difficulties:
        task = task_gen.generate_task(target_difficulty=diff)
        meta = task['meta']
        print(f"{diff:<12.2f} | {meta['market_type']:<12} | {task['slippage_rate']:<10.4f} | {task['commission_rate']:<10.4f} | {meta['window_size']:<8}")

    print("\nDemo 3 Complete!\n")


if __name__ == "__main__":
    demo_rich_feedback()
    demo_curriculum_learning()
    demo_difficulty_task_generation()
