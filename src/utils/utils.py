from enum import Enum

import numpy as np


class RandomIntDict(dict):
    def __init__(self, actions: int) -> None:
        super().__init__()
        self.actions = actions

    def __getitem__(self, key):
        return np.random.choice(self.actions)


class RandomDict(dict):
    def __init__(self, known_values: dict, max_value: int) -> None:
        super().__init__()
        self.known_values = known_values
        self.max_value = max_value

    def __getitem__(self, key):
        if key not in self.known_values:
            print(f"Key {key} not found in known values")
            return np.random.choice(self.max_value)
        return self.known_values[key]


class RenderMode(Enum):
    HUMAN = "human"
    RGB_ARRAY = "rgb_array"
    ANSI_TEXT = "ansi"
    NO_RENDER = None
