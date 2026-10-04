"""
Analysis driver: runs n-step SARSA and Expected SARSA on the Royal Game of Ur
over grids of hyperparameters and plots
  * the Q-value trajectory of a fixed tracked_state (one line per run, one figure
    per algorithm), and
  * the cumulative win ratio (n-step SARSA only), one line per run.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from enviroment import RoyalGameOfUr
from exp_sarsa import exp_sarsa
from n_step_sarsa import n_step_sarsa


NUM_EPISODES = 250_000
N_VALUES = [1, 2, 5, 10]
EPSILON_VALUES = [0.01, 0.05, 0.1, 0.3]
ALPHA_VALUES = [0.1, 0.01, 0.001]

# Directory where every figure is saved as a PNG.
IMAGES_DIR = Path(__file__).with_name("images")
IMAGES_DIR.mkdir(exist_ok=True)


def run_n_step_sarsa():
    """
    Run n-step SARSA over the (n, alpha) grid. Plots, in two figures:
      Figure 1: tracked_state Q-value trajectory, one coloured line per run.
      Figure 2: cumulative win ratio, one coloured line per run.
    """
    combos = [(n, alpha) for n in N_VALUES for alpha in ALPHA_VALUES]
    # Distinct colour per run, sampled from a continuous colormap.
    colors = plt.cm.viridis(np.linspace(0, 1, len(combos)))

    fig_traj, ax_traj = plt.subplots(figsize=(11, 6))
    fig_win, ax_win = plt.subplots(figsize=(11, 6))

    # Outer bar over the (n, alpha) runs; the per-episode bar inside n_step_sarsa
    # tracks progress within each individual run.
    outer = tqdm(zip(combos, colors), total=len(combos), desc="n-step SARSA runs")
    for (n, alpha), color in outer:
        outer.set_postfix(n=n, alpha=alpha)
        env = RoyalGameOfUr(2)
        Q_trajectory, _rewards, episode_wins, _lengths = n_step_sarsa(
            env, NUM_EPISODES, n, alpha=alpha
        )

        label = f"n={n}, alpha={alpha}"
        ax_traj.plot(Q_trajectory, color=color, label=label, linewidth=1.0)

        # Cumulative win ratio since the start (true 0/1 win flag).
        win_rate_cumulative = pd.Series(episode_wins).expanding().mean()
        ax_win.plot(win_rate_cumulative, color=color, label=label, linewidth=1.0)

    ax_traj.set_xlabel("Episode")
    ax_traj.set_ylabel("Q(tracked_state, tracked_action)")
    ax_traj.set_title("n-step SARSA: tracked-state Q-value trajectory")
    ax_traj.legend(fontsize=8, ncol=2)

    ax_win.axhline(0.5, color="gray", linewidth=0.8, linestyle=":")
    ax_win.set_ylim(0, 1)
    ax_win.set_xlabel("Episode")
    ax_win.set_ylabel("Cumulative win ratio")
    ax_win.set_title("n-step SARSA: cumulative win ratio")
    ax_win.legend(fontsize=8, ncol=2)

    fig_traj.savefig(IMAGES_DIR / "n_step_sarsa_trajectory.png", dpi=150, bbox_inches="tight")
    fig_win.savefig(IMAGES_DIR / "n_step_sarsa_win_ratio.png", dpi=150, bbox_inches="tight")


def run_exp_sarsa():
    """
    Run Expected SARSA over the (epsilon, alpha) grid. Plots, in one figure, the
    tracked_state Q-value trajectory, one coloured line per run.
    """
    combos = [(eps, alpha) for eps in EPSILON_VALUES for alpha in ALPHA_VALUES]
    colors = plt.cm.plasma(np.linspace(0, 1, len(combos)))

    fig_traj, ax_traj = plt.subplots(figsize=(11, 6))

    # Outer bar over the (epsilon, alpha) runs; the per-episode bar inside
    # exp_sarsa tracks progress within each individual run.
    outer = tqdm(zip(combos, colors), total=len(combos), desc="Expected SARSA runs")
    for (eps, alpha), color in outer:
        outer.set_postfix(eps=eps, alpha=alpha)
        env = RoyalGameOfUr(2)
        Q_trajectory, _rewards, _wins, _lengths = exp_sarsa(
            env, NUM_EPISODES, alpha=alpha, epsilon=eps
        )

        label = f"eps={eps}, alpha={alpha}"
        ax_traj.plot(Q_trajectory, color=color, label=label, linewidth=1.0)

    ax_traj.set_xlabel("Episode")
    ax_traj.set_ylabel("Q(tracked_state, tracked_action)")
    ax_traj.set_title("Expected SARSA: tracked-state Q-value trajectory")
    ax_traj.legend(fontsize=8, ncol=2)

    fig_traj.savefig(IMAGES_DIR / "exp_sarsa_trajectory.png", dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    run_n_step_sarsa()
    run_exp_sarsa()
    plt.show()
