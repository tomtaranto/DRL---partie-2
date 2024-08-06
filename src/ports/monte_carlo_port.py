from typing import Protocol, Any

from ports.types import Episode, Policy


class MonteCarloAlgo(Protocol):
    def generate_episode(self, policy: Policy) -> Episode:
        pass

    def train(self) -> Policy:
        pass
