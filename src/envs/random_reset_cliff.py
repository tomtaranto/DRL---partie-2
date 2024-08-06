from typing import Optional

import numpy as np
from gymnasium.envs.toy_text.cliffwalking import CliffWalkingEnv


class RandomResetCliff(CliffWalkingEnv):
    explored_starting_states = list(range(37))
    should_force_explore = False

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if self.should_force_explore:
            self.s = np.random.choice(self.explored_starting_states)  # We do not want to start at the terminal state
            self.explored_starting_states.remove(self.s)
            if len(self.explored_starting_states) == 0:
                self.explored_starting_states = list(range(37))
        else:
            self.s = np.random.choice(self.nS)  # We do not want to start at the terminal state

        self.lastaction = None

        if self.render_mode == "human":
            self.render()
        return int(self.s), {"prob": 1}
