import os
import sys
import pandas as pd
import numpy as np

# 将项目根目录添加到 python 路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.financial_env import FinancialAssistantEnv
from env.task_generator import TaskGenerator, create_multi_asset_data

def run_demo():
    print("=== 金融助手 Agent RL 环境: 任务生成器与打分器演示 ===")
    print("=== Financial Assistant RL Env: Task Generator & Scorer Demo ===\n")

    # 1. 准备多资产模拟数据
    print("正在生成模拟多资产数据...")
    multi_asset_data = create_multi_asset_data(num_assets=5, days=1000)
    
    # 2. 初始化任务生成器
    task_gen = TaskGenerator(
        data_dict=multi_asset_data,
        min_window=100,
        max_window=300,
        initial_balance_range=(10000, 50000),
        risk_profiles=['conservative', 'moderate', 'aggressive']
    )

    # 3. 创建集成任务生成器的环境
    env = FinancialAssistantEnv(task_generator=task_gen)

    # 4. 运行多个 Episode，展示任务多样性
    task_types = ["stock_analysis", "portfolio_management", "financial_planning"]
    num_episodes_per_type = 2

    for task_type in task_types:
        print(f"\n--- 任务类型: {task_type.replace('_', ' ').title()} 演示 ---")
        for episode in range(num_episodes_per_type):
            print(f"\n--- Episode {episode + 1} for {task_type} ---")
            
            # Reset 时会自动生成新任务，并指定任务类型
            obs, info = env.reset(options={"task_type": task_type})
            meta = env.current_task_meta
            
            print(f"任务描述: {meta['description']}")
            print(f"市场类型: {meta['market_type']} (难度: {meta['difficulty']})")
            print(f"时间范围: {meta['start_date'].date()} 至 {meta['end_date'].date()}")
            print(f"初始资金: {env.initial_balance:.2f}")
            print(f"手续费率: {env.simulator.commission_rate:.4f}")
            if 'risk_profile' in meta: print(f"风险偏好: {meta['risk_profile']}")
            if 'retirement_age' in meta: print(f"退休年龄: {meta['retirement_age']}, 当前年龄: {meta['current_age']}")
            
            done = False
            total_reward = 0
            while not done:
                # 使用随机动作模拟 Agent
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                done = terminated or truncated
                
            # Episode 结束，查看打分报告
            print("\n[Episode 结束评分报告]")
            report = info["score_report"]
            print(f"最终得分: {report['final_score']}")
            print(f"基础得分: {report['base_score']:.2f} (难度系数: {report['difficulty_multiplier']})")
            print("详细指标:")
            for k, v in report['metrics'].items():
                print(f"  - {k}: {v}")

    env.close()

if __name__ == "__main__":
    run_demo()
