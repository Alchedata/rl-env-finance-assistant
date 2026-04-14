"""
Curriculum Scheduler — dynamically adjusts task difficulty based on agent performance.
"""
import numpy as np


class CurriculumScheduler:
    """Adjusts task difficulty based on agent performance history."""

    def __init__(
        self,
        initial_difficulty: float = 0.1,
        promotion_threshold: float = 50.0,
        demotion_threshold: float = 10.0,
        difficulty_step_up: float = 0.1,
        difficulty_step_down: float = 0.05,
        history_window: int = 5,
        exploration_rate: float = 0.1,
    ):
        self.current_difficulty = initial_difficulty
        self.promotion_threshold = promotion_threshold
        self.demotion_threshold = demotion_threshold
        self.difficulty_step_up = difficulty_step_up
        self.difficulty_step_down = difficulty_step_down
        self.history_window = history_window
        self.exploration_rate = exploration_rate

        self._score_history = []
        self.total_episodes = 0
        self.promotions = 0
        self.demotions = 0

    def record_episode(self, score: float) -> None:
        """
        Record an episode score and trigger promotion/demotion check.

        Args:
            score: Episode final score
        """
        self._score_history.append(score)
        self.total_episodes += 1

        # Check for promotion/demotion after enough episodes
        if len(self._score_history) >= self.history_window:
            recent_scores = self._score_history[-self.history_window:]
            recent_avg = sum(recent_scores) / len(recent_scores)

            # Exploration: random difficulty with some probability
            if np.random.random() < self.exploration_rate:
                self.current_difficulty = np.random.uniform(0.0, 1.0)
            elif recent_avg >= self.promotion_threshold:
                self.current_difficulty = min(1.0, self.current_difficulty + self.difficulty_step_up)
                self.promotions += 1
            elif recent_avg <= self.demotion_threshold:
                self.current_difficulty = max(0.0, self.current_difficulty - self.difficulty_step_down)
                self.demotions += 1

    def get_difficulty_label(self) -> str:
        """Get human-readable difficulty label."""
        d = self.current_difficulty
        if d < 0.2:
            return 'beginner'
        elif d < 0.4:
            return 'easy'
        elif d < 0.6:
            return 'medium'
        elif d < 0.8:
            return 'hard'
        else:
            return 'expert'

    def get_stats(self) -> dict:
        """Get current scheduler statistics."""
        recent_avg = 0.0
        if len(self._score_history) > 0:
            window = self._score_history[-self.history_window:]
            recent_avg = sum(window) / len(window)

        return {
            'current_difficulty': round(self.current_difficulty, 2),
            'difficulty_label': self.get_difficulty_label(),
            'total_episodes': self.total_episodes,
            'promotions': self.promotions,
            'demotions': self.demotions,
            'recent_avg_score': round(recent_avg, 2),
        }

    def get_task_params_for_difficulty(self) -> dict:
        """
        Translate current difficulty to recommended TaskGenerator params.

        Returns:
            dict with window_size, commission_rate, slippage_rate, n_assets
        """
        d = self.current_difficulty

        # Window size: easier → longer window
        window_size = int(300 - d * 200)  # 300 at d=0, 100 at d=1
        window_size = max(100, min(300, window_size))

        # Commission and slippage
        if d <= 0.3:
            commission_rate = 0.001
            slippage_rate = 0.0003
        elif d <= 0.6:
            commission_rate = 0.002
            slippage_rate = 0.0005
        else:
            commission_rate = 0.005
            slippage_rate = 0.002

        # Number of assets: harder → more assets
        if d < 0.4:
            n_assets = 1
        elif d < 0.7:
            n_assets = 2
        else:
            n_assets = 3

        return {
            'window_size': window_size,
            'commission_rate': commission_rate,
            'slippage_rate': slippage_rate,
            'n_assets': n_assets,
        }