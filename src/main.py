import gymnasium as gym
import matplotlib.patches as patches
import matplotlib.pyplot as plt

from algos.first_visit_mc_prediction import FirstVisitMCPrediction
from algos.mc_exploring_start import MCExploringStart
from algos.mc_exploring_start_optimized import MCExploringStartOptimized
from envs.random_reset_cliff import RandomResetCliff
from ports.types import Policy
from utils.utils import RenderMode, RandomIntDict, RandomDict


def first_visit_mc_prediction() -> None:
    env = gym.make("CliffWalking-v0", render_mode=RenderMode.NO_RENDER.value)
    actions = env.action_space.n
    print(f"Nombre d'actions possibles: {actions}")
    policy = RandomIntDict(actions)
    print(f"Policy initiale: {policy}")
    algo = FirstVisitMCPrediction(env, policy, iterations=10)
    algo.train()


def mc_exploring_start() -> None:
    env = RandomResetCliff(render_mode=RenderMode.NO_RENDER.value)

    algo = MCExploringStart(env, iterations=5000)
    pi = algo.train()

    print(f"{pi=}")
    print(f"{sorted(pi.items())}")
    plot_policy(pi)
    play(pi)


def mc_exploring_start_optimized() -> None:
    env = RandomResetCliff(render_mode=RenderMode.NO_RENDER.value)

    algo = MCExploringStartOptimized(env, iterations=5000000)
    pi = algo.train()

    # pi = np.array([1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 2, 2, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1,
    #                1, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0,
    #                0, 0, 1, 1])
    print(f"{pi=}")
    dict_pi = {i: pi[i] for i in range(len(pi))}
    plot_policy(dict_pi)
    print(f"{sorted(dict_pi.items())}")
    play(dict_pi)


def play(policy: Policy) -> None:
    env = gym.make("CliffWalking-v0", render_mode=RenderMode.HUMAN.value)
    policy = RandomDict(policy, env.action_space.n)
    state, infos = env.reset()
    is_terminated = False
    while not is_terminated:
        action = policy[state]
        next_state, reward, is_terminated, truncated, info = env.step(action)
        print(f"{state=}, {action=}, {next_state=}, {reward=}, {truncated=}, {info=}, {is_terminated=}")
        state = next_state
    env.close()


def plot_policy(policy: Policy) -> None:
    rows, cols = 4, 12

    fig, ax = plt.subplots()

    directions = {
        0: (0, -0.4),  # Up
        1: (0.4, 0),  # Right
        2: (0, 0.4),  # Down
        3: (-0.4, 0)  # Left
    }

    arrow_props = dict(facecolor='black', shrink=0, width=0.02, headwidth=3, headlength=4)
    for key, value in policy.items():
        row = key // cols
        col = key % cols
        dx, dy = directions.get(value, (0, 0))
        ax.annotate('', xy=(col + dx, row + dy), xytext=(col, row),
                    arrowprops=arrow_props)

    # Define colors
    highlight_colors = {
        36: 'green',  # Starting cell in green
        47: 'blue',  # Objective cell in blue
    }

    # Add patches for each cell
    for row in range(rows):
        for col in range(cols):
            cell_number = row * cols + col
            color = 'lightgrey'  # Default color
            if cell_number in highlight_colors:
                color = highlight_colors[cell_number]
            elif row == rows - 1:
                color = 'red'
            rect = patches.Rectangle((col - 0.5, row - 0.5), 1, 1, linewidth=1, edgecolor='black', facecolor=color)
            ax.add_patch(rect)

    ax.set_xlim(-0.5, cols - 0.5)
    ax.set_ylim(-0.5, rows - 0.5)
    ax.set_aspect('equal')
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    ax.invert_yaxis()
    plt.show()


def main() -> None:
    first_visit_mc_prediction()
    # mc_exploring_start()
    # mc_exploring_start_optimized()


if __name__ == '__main__':
    main()
