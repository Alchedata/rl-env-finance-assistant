"""
Training Configuration — shared configuration for all trainers.
"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TrainingConfig:
    """Unified training configuration for all algorithms."""

    # --- Environment ---
    env_id: str = "FinancialAssistantEnv-v0"
    num_envs: int = 1
    window_size: int = 20
    initial_balance: float = 10000.0
    include_macro: bool = False
    include_news: bool = False
    num_days: int = 1260  # simulated data length

    # --- Training ---
    total_timesteps: int = 100_000
    seed: int = 42
    device: str = "auto"  # "auto", "cpu", "cuda"
    verbose: int = 1

    # --- Logging ---
    log_dir: str = "./logs"
    tensorboard_log: Optional[str] = None
    save_path: Optional[str] = None

    # --- Evaluation ---
    eval_freq: int = 10_000
    n_eval_episodes: int = 5
    eval_reward_threshold: Optional[float] = None

    # --- Algorithm-specific ---
    learning_rate: float = 3e-4
    batch_size: int = 64
    n_steps: int = 2048  # PPO rollout buffer
    gamma: float = 0.99
    gae_lambda: float = 0.95

    # PPO-specific
    ppo_clip_range: float = 0.2
    ppo_n_epochs: int = 10
    ppo_ent_coef: float = 0.0

    # SAC-specific
    sac_buffer_size: int = 100_000
    sac_learning_starts: int = 1000
    sac_tau: float = 0.005

    # DPO-specific
    dpo_beta: float = 0.1  # DPO regularization strength
    dpo_n_epochs: int = 10
    dpo_reference_free: bool = True

    # GRPO-specific
    grpo_group_size: int = 8  # number of samples per group
    grpo_clip_range: float = 0.2
    grpo_n_epochs: int = 4

    # --- Network ---
    net_arch: Optional[dict] = None  # e.g., {"pi": [256, 256], "vf": [256, 256]}

    def get_save_path(self, algorithm: str) -> str:
        """Get model save path."""
        if self.save_path:
            return self.save_path
        return f"{self.log_dir}/{algorithm}_financial_agent"

    def get_tensorboard_log(self, algorithm: str) -> Optional[str]:
        """Get tensorboard log directory."""
        if self.tensorboard_log:
            return self.tensorboard_log
        return f"{self.log_dir}/{algorithm}_tensorboard/"