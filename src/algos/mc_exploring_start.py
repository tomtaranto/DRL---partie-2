import numpy as np
import tqdm

from ports.monte_carlo_port import MonteCarloAlgo
from ports.types import Env, Episode, Policy, State, Action


class MCExploringStart(MonteCarloAlgo):
    def __init__(self, env: Env, iterations: int = 5, gamma: float = 0.99) -> None:
        self.env = env
        self.pi = {}
        self.Q = {}
        self.Returns = {}
        self.iterations = iterations
        self.gamma = gamma
        self.epsilon = 0.1

    def generate_episode(self, policy: Policy) -> Episode:
        current_state, _ = self.env.reset()
        action = self.env.action_space.sample()
        is_terminated = False
        episode: Episode = []
        max_steps = 1000
        while not is_terminated and max_steps > 0:
            next_state, reward, is_terminated, truncated, info = self.env.step(action)
            episode.append((current_state, action, float(reward)))
            current_state = next_state
            action = self._get_next_action(current_state)
            max_steps -= 1
        return episode

    def train(self) -> Policy:
        for _ in tqdm.tqdm(range(self.iterations)):
            episode = self.generate_episode(None)
            G = 0.0
            for i, (state, action, reward) in enumerate(episode[::-1]):
                G = self.gamma * reward + G
                if self._should_update(state, action, episode, i):
                    self._update_returns(G, action, state)
                    self._update_q(action, state)
                    self.pi[state] = np.argmax([self.Q[state][a] for a in self.Q[state].keys()])
        print(f"{self.pi=}")
        return self.pi

    def _update_q(self, action: Action, state: State) -> None:
        if state not in self.Q:
            self.Q[state] = {action: np.mean(self.Returns[state][action])}
        else:
            if action not in self.Q[state]:
                self.Q[state][action] = np.mean(self.Returns[state][action])
            else:
                self.Q[state][action] = np.mean(self.Returns[state][action])

    def _update_returns(self, G: float, action: Action, state: State) -> None:
        if state not in self.Returns:
            self.Returns[state] = {action: [G]}
        else:
            if action not in self.Returns[state]:
                self.Returns[state][action] = [G]
            else:
                self.Returns[state][action].append(G)

    def _should_update(self, state: State, action: Action, episode: Episode, i: int) -> bool:
        if len(episode[:-(i+1)]) == 0:
            return True
        for s, a, _ in episode[:-(i+1)]:
            if state == s and action == a:
                return False
        return True

    def _get_next_action(self, state: State) -> Action:
        if state not in self.pi or np.random.rand() < self.epsilon:  # needed to allow exploration
            return self.env.action_space.sample()  # Random action
        return self.pi[state]
