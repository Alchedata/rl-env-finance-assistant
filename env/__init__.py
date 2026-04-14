"""
env package — Financial Assistant RL Environment.
"""
from env.financial_env import FinancialAssistantEnv, generate_dummy_data
from env.task_generator import TaskGenerator, create_multi_asset_data, create_simulated_macro_data, create_simulated_news_data
from env.curriculum_scheduler import CurriculumScheduler