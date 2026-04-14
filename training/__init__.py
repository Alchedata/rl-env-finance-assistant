"""
Training module — unified trainers for multiple RL algorithms.

Supported algorithms:
- SB3: PPO, SAC, A2C, TD3, DDPG
- Custom: DPO (Direct Preference Optimization), GRPO (Group Relative Policy Optimization)
"""
from training.config import TrainingConfig
from training.sb3_trainer import SB3Trainer
from training.preference_collector import PreferenceCollector
from training.dpo_trainer import DPOTrainer
from training.grpo_trainer import GRPOTrainer

__all__ = [
    'TrainingConfig',
    'SB3Trainer',
    'PreferenceCollector',
    'DPOTrainer',
    'GRPOTrainer',
]