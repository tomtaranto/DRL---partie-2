from typing import Any

import numpy as np
import tqdm

from ports.monte_carlo_port import MonteCarloAlgo
from ports.types import Env, Episode, Policy


class FirstVisitMCPrediction(MonteCarloAlgo):
    def __init__(self, env: Env, policy: Policy, iterations: int = 5, gamma: float = 0.99) -> None:
        self.env = env
        self.policy = policy
        self.V = {}
        self.Returns = {}
        self.iterations = iterations
        self.gamma = gamma

    def generate_episode(self, policy: Policy) -> Episode:
        current_state, _ = self.env.reset()
        is_terminated = False
        episode: Episode = []
        while not is_terminated:
            action = policy[current_state]
            next_state, reward, terminated, truncated, info = self.env.step(action)
            is_terminated = terminated or truncated
            episode.append((current_state, action, float(reward)))
            current_state = next_state
        return episode

    def train(self) -> Policy:
        for _ in tqdm.tqdm(range(self.iterations)):
            episode = self.generate_episode(self.policy)
            G = 0
            for i, (state, _, reward) in enumerate(episode[::-1]):
                G = self.gamma * reward + G
                if self._is_first_time_in_state(episode, i, state):
                    if state not in self.Returns:
                        self.Returns[state] = [G]
                    else:
                        self.Returns[state].append(G)
                    self.V[state] = float(np.mean(self.Returns[state]))
        print(f"Final value function (node: mean_return): {sorted(self.V.items(), key=lambda item: item[1], reverse=True)=}")

    @staticmethod
    def _is_first_time_in_state(episode, i, state):
        if len(episode[:i]) == 0:
            return True
        for s, _, _ in episode[:i]:
            if s == state:
                return False
        return True
