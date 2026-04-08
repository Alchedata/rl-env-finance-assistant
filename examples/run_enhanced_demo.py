"""
Enhanced Demo: 展示丰富环境反馈 + 课程学习功能
Demonstrates rich environment feedback (macro, news) and curriculum learning.
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
    """运行一个 episode 并返回结果"""
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
    """演示丰富环境反馈（宏观经济 + 新闻情感）"""
    print("=" * 70)
    print("Demo 1: Rich Environment Feedback (Macro + News)")
    print("=" * 70)

    # 创建模拟数据
    data_dict = create_multi_asset_data(num_assets=3, days=500)
    macro_data = create_simulated_macro_data(days=500)
    news_data = create_simulated_news_data(days=500)

    # 创建带宏观和新闻数据的任务生成器
    task_gen = TaskGenerator(
        data_dict=data_dict,
        risk_profiles=['conservative', 'moderate', 'aggressive'],
        macro_data=macro_data,
        news_data=news_data
    )

    # 创建环境（启用宏观和新闻特征）
    env = FinancialAssistantEnv(
        task_generator=task_gen,
        include_macro=True,
        include_news=True
    )

    print(f"\nObservation space: {env.observation_space.shape}")
    print(f"Action space: {env.action_space.shape}")

    # 运行不同类型的任务
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
    """演示课程学习功能"""
    print("=" * 70)
    print("Demo 2: Curriculum Learning")
    print("=" * 70)

    # 创建数据
    data_dict = create_multi_asset_data(num_assets=3, days=500)
    macro_data = create_simulated_macro_data(days=500)
    news_data = create_simulated_news_data(days=500)

    task_gen = TaskGenerator(
        data_dict=data_dict,
        risk_profiles=['conservative', 'moderate', 'aggressive'],
        macro_data=macro_data,
        news_data=news_data
    )

    # 创建课程学习调度器
    scheduler = CurriculumScheduler(
        initial_difficulty=0.1,
        promotion_threshold=50.0,
        demotion_threshold=10.0,
        difficulty_step_up=0.1,
        difficulty_step_down=0.05,
        history_window=5,
        exploration_rate=0.1
    )

    # 创建带课程学习的环境
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

    # 打印最终统计
    final_stats = scheduler.get_stats()
    print(f"\n--- Curriculum Learning Summary ---")
    print(f"  Final Difficulty: {final_stats['current_difficulty']:.2f} ({final_stats['difficulty_label']})")
    print(f"  Total Episodes: {final_stats['total_episodes']}")
    print(f"  Total Promotions: {final_stats['promotions']}")
    print(f"  Total Demotions: {final_stats['demotions']}")
    print(f"  Recent Avg Score: {final_stats['recent_avg_score']:.2f}")

    # 展示难度参数推荐
    print(f"\n--- Recommended Task Params at Current Difficulty ---")
    params = scheduler.get_task_params_for_difficulty()
    for key, value in params.items():
        print(f"  {key}: {value}")

    env.close()
    print("\nDemo 2 Complete!\n")


def demo_difficulty_levels():
    """演示不同难度等级的任务生成"""
    print("=" * 70)
    print("Demo 3: Difficulty-Aware Task Generation")
    print("=" * 70)

    data_dict = create_multi_asset_data(num_assets=3, days=500)
    task_gen = TaskGenerator(data_dict=data_dict)

    difficulties = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

    for diff in difficulties:
        task = task_gen.generate_task(task_type="stock_analysis", target_difficulty=diff)
        meta = task["meta"]
        print(f"\nTarget Difficulty: {diff:.1f}")
        print(f"  Market Type: {meta['market_type']}")
        print(f"  Window Size: {meta['window_size']}")
        print(f"  Difficulty Score: {meta['difficulty_score']}")
        print(f"  Commission Rate: {task['commission_rate']:.4f}")
        print(f"  Slippage Rate: {task['slippage_rate']:.5f}")
        print(f"  Initial Balance: {task['initial_balance']:.2f}")

    print("\nDemo 3 Complete!\n")


if __name__ == '__main__':
    demo_rich_feedback()
    demo_curriculum_learning()
    demo_difficulty_levels()
    print("All demos completed successfully!")
