import numpy as np
import tqdm

from ports.monte_carlo_port import MonteCarloAlgo
from ports.types import Env, Episode, Policy, Action, State


class MCExploringStartOptimized(MonteCarloAlgo):
    def __init__(self, env: Env, iterations: int = 5, gamma: float = 0.99) -> None:
        self.env = env
        self.iterations = iterations
        self.gamma = gamma
        self.epsilon = 0.1

        # Initialize Q and Returns as numpy arrays
        self.state_space_size = self.env.observation_space.n
        self.action_space_size = self.env.action_space.n
        self.Q = np.zeros((self.state_space_size, self.action_space_size))  # Q[s, a] = q
        self.Returns = np.zeros((self.state_space_size, self.action_space_size,
                                 2))  # 2 for sum and count, Returns[s, a, 0] = sum, Returns[s, a, 1] = count

        self.pi = np.random.randint(self.action_space_size, size=self.state_space_size)  # pi[s] = a

    def generate_episode(self, policy: Policy) -> Episode:
        current_state, _ = self.env.reset()
        action = self.env.action_space.sample()
        episode = []
        max_steps = 1000
        for _ in range(max_steps):
            next_state, reward, is_terminated, truncated, _ = self.env.step(action)
            episode.append((current_state, action, float(reward)))
            current_state = next_state
            action = self._get_next_action(current_state)
            if is_terminated or truncated:
                break
        return episode

    def train(self) -> Policy:
        for _ in tqdm.tqdm(range(self.iterations)):
            episode = self.generate_episode(None)
            G = 0
            visited_state_action_pairs = set()
            for i in range(len(episode) - 1, -1, -1):
                state, action, reward = episode[i]
                G = self.gamma * G + reward
                if (state, action) not in visited_state_action_pairs:
                    visited_state_action_pairs.add((state, action))
                    self._update_returns(G, action, state)
                    self._update_q(action, state)
                    self.pi[state] = np.argmax(self.Q[state])
        return self.pi

    def _update_q(self, action: Action, state: State) -> None:
        sum_returns, count_returns = self.Returns[state, action]
        self.Q[state, action] = sum_returns / count_returns

    def _update_returns(self, G: float, action: Action, state: State) -> None:
        self.Returns[state, action, 0] += G
        self.Returns[state, action, 1] += 1

    def _get_next_action(self, state: State) -> Action:
        if np.random.rand() < self.epsilon:
            return self.env.action_space.sample()  # Random action
        return self.pi[state]
