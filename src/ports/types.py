import gymnasium.core
import numpy as np

Action = gymnasium.core.ActType
State = gymnasium.core.ObsType
Reward = float

Episode = list[tuple[State, Action, Reward]]

Env = gymnasium.core.Env

Policy = dict[State, Action] | np.ndarray | None
