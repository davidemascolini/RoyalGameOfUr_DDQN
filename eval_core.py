"""
Standalone evaluation harness for trained Royal Game of Ur policies.

This module deliberately does NOT import DQN_network.py: that module runs the
whole training setup at import time (builds two nets, calls env.reset(), opens a
pyplot window). The network definition and the state/action encodings are small
enough to restate here, and they are kept byte-compatible with the originals:

    DQN            <-> DQN_network.py:53-66
    encode_state   <-> DQN_network.py:82-97
    all_actions    <-> DQN_network.py:101-108
    greedy_policy  <-> DQN_network.py:210-224 (make_opponent_policy)

A "policy" here is the callable signature the environment already expects for
opponent_policy (enviroment.py:163-166):

    policy(raw_state: np.ndarray, legal_actions: list[tuple]) -> (start, dest)

where raw_state is always written from the *moving* player's point of view, so
the same callable works for either side.
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from enviroment import RoyalGameOfUr

# The policy in final_policy.pt was trained at N=7, and N=7 is also what the
# solved Finkel lookup table assumes. Nothing here is valid at other N.
N_PIECES = 7
BOARD_SIZE = 16
DICE_SIZE = 5

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)


#######################################
#              NETWORK                #
#######################################

class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


#######################################
#       STATE / ACTION ENCODING       #
#######################################

def encode_state(state, n_pieces=N_PIECES, board_size=BOARD_SIZE, dice_size=DICE_SIZE):
    """
    state: (x1..xN, y1..yN, d) -> concatenated position histograms plus a one-hot
    dice roll. 16 + 16 + 5 = 37 floats. The histogram makes the encoding
    independent of piece ordering (and of N, in dimension only).
    """
    x = state[0:n_pieces]
    y = state[n_pieces:2 * n_pieces]
    d = state[2 * n_pieces]

    x_hist = np.bincount(x, minlength=board_size)
    y_hist = np.bincount(y, minlength=board_size)
    dice = np.eye(dice_size)[d]

    return np.concatenate([x_hist, y_hist, dice]).astype(np.float32)


all_actions = [
    (start, destination)
    for start in range(BOARD_SIZE)
    for destination in range(BOARD_SIZE)
]
action_to_index = {action: idx for idx, action in enumerate(all_actions)}
index_to_action = {idx: action for idx, action in enumerate(all_actions)}
N_ACTIONS = len(all_actions)          # 256
N_OBSERVATIONS = 2 * BOARD_SIZE + DICE_SIZE   # 37


#######################################
#              POLICIES               #
#######################################

def load_policy(path="final_policy.pt", dev=None):
    """Rebuild the 37 -> 128 -> 128 -> 256 net and load a saved state_dict."""
    dev = dev or device
    net = DQN(N_OBSERVATIONS, N_ACTIONS).to(dev)
    net.load_state_dict(torch.load(path, map_location=dev))
    net.eval()
    return net


def greedy_policy(net, dev=None):
    """
    Greedy argmax over Q, with illegal actions masked to -inf. Identical to
    make_opponent_policy in DQN_network.py, but without the module-global net.
    """
    dev = dev or device

    def policy(raw_state, legal_actions):
        state = encode_state(raw_state)
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32, device=dev).unsqueeze(0)
            q_values = net(state_tensor)

        legal_indices = [action_to_index[action] for action in legal_actions]
        mask = torch.full((q_values.shape[1],), -float("inf"), device=dev)
        mask[legal_indices] = 0.0
        masked_q_values = q_values + mask.unsqueeze(0)
        action_idx = masked_q_values.max(1).indices.view(1, 1)
        return index_to_action[int(action_idx)]

    return policy


def random_policy(rng=None):
    """Uniform over legal actions. The floor of the ladder."""
    rng = rng or np.random.default_rng()

    def policy(raw_state, legal_actions):
        return legal_actions[int(rng.integers(len(legal_actions)))]

    return policy


def heuristic_policy(rosettes=(4, 8, 14), public_cells=tuple(range(5, 13)), scored_cell=15):
    """
    A simple but non-trivial hand-written opponent, in priority order:
      1. score a piece
      2. capture an enemy piece (only possible in the shared middle lane)
      3. land on a rosette (extra turn)
      4. otherwise advance the piece that is furthest along

    raw_state is from the mover's perspective, so the enemy pieces are the
    second block of N_PIECES entries.
    """

    def policy(raw_state, legal_actions):
        enemy = set(int(v) for v in raw_state[N_PIECES:2 * N_PIECES])

        def score(action):
            start, dest = action
            if start == dest:            # the pass action (15, 15)
                return (-1, 0)
            if dest == scored_cell:
                return (4, dest)
            if dest in public_cells and dest in enemy:
                return (3, dest)
            if dest in rosettes:
                return (2, dest)
            return (1, dest)

        return max(legal_actions, key=score)

    return policy


#######################################
#           MATCH RUNNING             #
#######################################

def play_match(p1_policy, p2_policy, env=None, seed=None, max_steps=2000):
    """
    Play one full game. Returns True if player 1 won, False if player 2 won,
    None if the game hit max_steps without a winner.

    env.step() always advances P1 and runs P2's entire turn internally via
    opponent_policy, so p1_policy drives the loop and p2_policy is handed to the
    environment. Winner is read off the board rather than the reward sign,
    because reward carries a shaping term (enviroment.py:231).
    """
    if env is None:
        env = RoyalGameOfUr(N_PIECES)

    state, info = env.reset(seed=seed)
    legal_actions = env.get_legal_moves()

    for _ in range(max_steps):
        # env.step sets current_player = 1 itself, but only once it is called --
        # after a P2 turn the flag is still 2 here, and a policy that needs to
        # know whose move it is (ur_bridge.lut_policy) would read it stale.
        env.current_player = 1
        action = p1_policy(state, legal_actions)
        state, reward, terminated, truncated, info = env.step(action, opponent_policy=p2_policy)

        if terminated:
            if all(piece == env.scored_cell for piece in env.player1_loc):
                return True
            if all(piece == env.scored_cell for piece in env.player2_loc):
                return False
            return None

        legal_actions = info["legal_actions"]

    return None


#######################################
#             STATISTICS              #
#######################################

def wilson_interval(wins, n, z=1.96):
    """
    Wilson score interval for a binomial proportion. Preferred over the normal
    approximation because it stays inside [0, 1] and behaves sanely for the
    lopsided win rates we expect against a solved opponent.
    """
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = wins / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (p, max(0.0, centre - margin), min(1.0, centre + margin))


def evaluate(policy, opponent, n_games=1000, seed=0, swap_sides=True, max_steps=2000,
             env=None):
    """
    Play `policy` against `opponent` and return a result dict.

    The environment is NOT side-symmetric (P1 always moves first), so by default
    half the games are played with `policy` as P1 and half with it as P2. The
    per-side win rates are reported separately: a large gap between them means
    the first-move advantage matters more than the policy difference, which is
    worth knowing before reading anything into the aggregate.

    `env` matters for policies that inspect the live environment rather than
    raw_state (ur_bridge.lut_policy has to, because raw_state is sorted and
    cannot say which side is which). Such a policy closes over an env, and it
    must be *this* env -- otherwise it reads a board that never advances. Pass
    the same object you gave the policy.
    """
    if env is None:
        env = RoyalGameOfUr(N_PIECES)

    if not swap_sides:
        wins = sum(
            1 for i in range(n_games)
            if play_match(policy, opponent, env=env, seed=seed + i, max_steps=max_steps) is True
        )
        p, lo, hi = wilson_interval(wins, n_games)
        return {"wins": wins, "games": n_games, "rate": p, "lo": lo, "hi": hi,
                "as_p1": p, "as_p2": None, "unfinished": 0}

    half = n_games // 2
    unfinished = 0

    wins_p1 = 0
    for i in range(half):
        r = play_match(policy, opponent, env=env, seed=seed + i, max_steps=max_steps)
        if r is True:
            wins_p1 += 1
        elif r is None:
            unfinished += 1

    # Sides swapped: the baseline drives the loop as P1, `policy` is P2. A P2 win
    # (play_match returning False) is a win for `policy`.
    wins_p2 = 0
    for i in range(n_games - half):
        r = play_match(opponent, policy, env=env, seed=seed + 10_000 + i, max_steps=max_steps)
        if r is False:
            wins_p2 += 1
        elif r is None:
            unfinished += 1

    wins = wins_p1 + wins_p2
    p, lo, hi = wilson_interval(wins, n_games)
    return {
        "wins": wins,
        "games": n_games,
        "rate": p,
        "lo": lo,
        "hi": hi,
        "as_p1": wins_p1 / half if half else None,
        "as_p2": wins_p2 / (n_games - half) if n_games - half else None,
        "unfinished": unfinished,
    }


def format_row(label, res):
    """One line of the results table."""
    side = ""
    if res["as_p2"] is not None:
        side = f"   (P1 {res['as_p1']:.3f} / P2 {res['as_p2']:.3f})"
    warn = f"  [{res['unfinished']} unfinished]" if res.get("unfinished") else ""
    return (f"{label:<28} {res['rate']:.3f}  "
            f"[{res['lo']:.3f}, {res['hi']:.3f}]  "
            f"{res['wins']}/{res['games']}{side}{warn}")
