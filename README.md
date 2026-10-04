# Reinforcement Learning on the Royal Game of Ur

Reinforcement Learning assignment (Sapienza): a Gymnasium environment for the
Royal Game of Ur plus three agents that learn to play it — tabular n-step SARSA,
tabular Expected SARSA, and a Double DQN with self-play.

The assignment text is in [ReinforcementLearning_Assignment2.pdf](ReinforcementLearning_Assignment2.pdf).

## The environment

[enviroment.py](enviroment.py) implements `RoyalGameOfUr(N)`, where `N` is the
number of pieces per player. The agent is always player 1; player 2 is driven
inside `step()` by an opponent policy (random by default).

**Encoding.** Each piece has a position in `0..15`: `0` = not yet entered,
`1..14` = the path, `15` = scored. Cells `5..12` are public (captures happen
there), `4`, `8`, `14` are rosettes (landing on one grants another roll), and
cell `8` is the shared rosette, which is a safe square.

- **Observation** — `MultiDiscrete`: `N` sorted player-1 positions, `N` sorted
  player-2 positions, then the dice roll `0..4`. Sorting collapses
  permutation-symmetric states (`[3,7]` and `[7,3]` are the same position),
  roughly halving the effective state space for the tabular methods.
- **Action** — a `(start, destination)` pair, with `destination == start + roll`.
  The pass action, used when no move is legal, is `(15, 15)`.
- **Dice** — `Binomial(4, 0.5)`, the four-binary-dice roll of the original game.
- **Reward** — `+1` for a win, `-1` for a loss, plus a potential-based shaping
  term `Phi(s') - Phi(s)` scaled by `0.01 / N`, where `Phi` is player 1's total
  board progress minus player 2's. Potential-based shaping (Ng, Harada & Russell
  1999) is policy-invariant, so it densifies the signal without changing the
  optimal policy. Because of it, the terminal reward is not exactly `±1` — check
  its sign, not its value, when scoring episodes.

`step()` runs player 2's whole turn (including extra rolls from rosettes) before
returning, so one environment step is one player-1 decision.

## Agents

| File | Method | Notes |
| --- | --- | --- |
| [n-step sarsa.py](n-step%20sarsa.py) | n-step SARSA | Current version; legal-action masking, random tie-breaking. Filename has a space, so it is loaded via `importlib`. |
| [n-step_sarsa.py](n-step_sarsa.py) | n-step SARSA | Older version, kept for reference; uses a rejection loop for legality. |
| [exp_sarsa.py](exp_sarsa.py) | Expected SARSA | Bootstraps against the epsilon-greedy distribution over legal actions. |
| [DQN_network.py](DQN_network.py) | Double DQN | Network, replay buffer, training loop and self-play evaluation, all in one file. |

Both tabular methods keep `Q` as a `defaultdict(state -> np.zeros(256))` over the
flat action index `start * 16 + dest`, and track the Q-value of one fixed state
(`(0,0,0,0,4)`, action `(0,4)`) across episodes so convergence can be plotted.

### DQN details

- **Input** — `encode_state` turns the observation into a 37-dim float vector:
  a 16-bin histogram of own pieces, a 16-bin histogram of the opponent's, and a
  one-hot dice roll.
- **Output** — Q-values over all 256 `(start, destination)` pairs; illegal
  actions are masked with `-inf` before the argmax, both when acting and when
  bootstrapping.
- **Architecture** — 2 hidden layers of 128 units (ReLU), AdamW at `lr=1e-5`,
  Huber loss, gradient clipping at norm 10, soft target updates (`TAU=0.005`),
  `GAMMA=1`, replay capacity 100k, batch 128, epsilon decaying `0.9 -> 0.01`.
- **Self-play** — the snapshot at episode 1000 is saved as `old_net.pt` and used
  as the fixed evaluation benchmark. From then on, every 500 episodes a new
  snapshot joins the pool of opponents sampled at the start of each episode, and
  a 200-match evaluation against the benchmark is logged.
- **Artifacts** — `final_policy.pt` (end of training) and `old_net.pt`
  (episode-1000 benchmark). Both are checked in and `N = 7`, so they only load
  into a net built for that piece count.

`DQN_network.py` refuses to train without a GPU (it sets `num_episodes = 0` and
prints a warning) — edit that branch if you want to run it on CPU.

## Running things

The project uses [uv](https://docs.astral.sh/uv/) and Python 3.12:

```bash
uv sync
```

Note that `pyproject.toml` does not yet list `torch` and `tqdm`, which the DQN
and the training scripts import; install them as well:

```bash
uv add torch tqdm
```

Then:

```bash
uv run python DQN_network.py   # train the DQN (needs a GPU), writes final_policy.pt
uv run python analysis.py      # hyperparameter sweeps for both SARSA variants
uv run python play_test.py     # watch random-vs-random episodes on the rendered board
uv run python match_test.py    # play against the trained DQN in the terminal
uv run python eval_ladder.py   # win rates vs random / heuristic / old_net, with CIs
uv run python eval_optimal.py  # vs the solved game (needs finkel.rgu, see below)
```

`match_test.py` prompts for the *index* of your move in the printed list of
legal moves, not the move itself.

## Evaluating a trained policy
vv
Training only ever measured the agent against itself, which cannot say how
strong it is in absolute terms. Three scripts answer that, in increasing order
of cost and rigour.

### Quick ladder — no downloads

```bash
uv run python eval_ladder.py --games 2000 --seed 0
```

[eval_core.py](eval_core.py) is a standalone harness: it restates the network
and encodings from `DQN_network.py` rather than importing it (that module builds
nets, resets the env and opens a pyplot window at import time). It provides
`load_policy`, `greedy_policy`, `random_policy`, `heuristic_policy`,
`play_match` and `evaluate`. Games are played with **sides swapped** for half the
sample, because player 1 moves first and the env is not symmetric, and win rates
carry a **Wilson score interval**.

Results for the checked-in `final_policy.pt`, 2000 games per opponent:

| Opponent | Win rate | 95% CI |
| --- | --- | --- |
| random | 0.993 | [0.988, 0.995] |
| heuristic (score > capture > rosette > advance) | 0.791 | [0.773, 0.808] |
| `old_net.pt` (episode-1000 snapshot) | 0.731 | [0.711, 0.749] |

The heuristic itself beats random 0.933 of the time, so the 0.791 is a real
margin over a non-trivial opponent, and beating its own earlier snapshot 0.731
confirms the second half of training was not wasted.

### Against perfect play

RoyalUr.net strongly solved the Finkel ruleset by value iteration and published
the lookup table, so the game has a known optimal opponent and a ground-truth
win probability for every position. [ur_bridge.py](ur_bridge.py) connects this
environment to it.

```bash
uv run python -m pip install "git+https://github.com/RoyalUr/RoyalUr-Python.git"
# download finkel.rgu (827MB) from https://huggingface.co/sothatsit/RoyalUrModels
uv run python ur_bridge.py --positions 5000   # verification gate, must PASS
uv run python eval_optimal.py --games 2000 --analysis-games 600
```

The PyPI release of `royalur` has no `lut` module — install from GitHub. The
table is gitignored.

**The rulesets genuinely coincide.** The `.rgu` header reports `board_shape:
Standard, paths: Bell, dice: FourBinary, start_pieces: 7, safe_rosettes: true,
rosettes_grant_rolls: true, captures_grant_rolls: false`, which is exactly what
[enviroment.py](enviroment.py) implements. Our linear cell `c` in `1..14` is Bell
path index `c - 1`, and the shared middle column is path indices `4..11`, i.e.
our `public_cells` `5..12`. Valid at `N = 7` only.

`ur_bridge.py` encodes positions straight into LUT keys instead of building
`royalur.Game` objects, which is what makes thousands of games practical. Its
`--positions` gate cross-checks every key against the library's own encoder;
that check is not ceremony, it caught a real bug (the side-lane bit order is the
reverse of path order, which otherwise produces valid-looking keys that are
silently the wrong position).

Reported by [eval_optimal.py](eval_optimal.py), 2000 games / 600 analysis games:

| Metric | Value |
| --- | --- |
| Win rate vs. optimal play | 0.015 [0.010, 0.021] |
| Random vs. optimal play (floor) | 0.000 [0.000, 0.010] |
| Optimal-move agreement | 0.578 |
| Mean regret per decision | 0.0055 win probability |
| Blunder rate (costing > 0.05) | 0.016 |

Perfect play wins 0.5154 from the start position, which matches the published
first-player advantage and is itself a check that the bridge is correct.

**Read the second half of that table, not the first.** A 2% win rate against a
solved opponent is close to uninformative — Ur is a dice game and one match is
roughly one bit. The move-quality metrics score *every* decision against the
table's win probabilities, so 600 games yield ~53k scored decisions and much
tighter conclusions. Mean regret of 0.0055 says the policy is usually close to
optimal even when it does not pick the best move, and the median regret is 0;
agreement of 0.578 with a 0.016 blunder rate says the remaining loss is
concentrated in a few bad decisions rather than spread thinly.

Regret is also broken down by game phase, and it roughly doubles from opening
(0.0047) to endgame (0.0095) — the endgame is where this policy is weakest,
which is where to look next.

`--show-blunders` dumps the worst positions, and they share a shape: almost all
of them are races where the opponent has one piece left on the board, around
cells 10-11, and the policy introduces a new piece from home (`(0, 3)`) or
shuffles a back piece instead of moving the piece on cell 8 to 11 to block or
contest. The single worst case costs 0.199 win probability by playing `(8, 11)`
when `(12, 15)` would have borne a piece off. In short: the policy under-values
finishing and over-values developing new pieces once the race is decided — a
concrete, fixable weakness that a win rate alone would never have surfaced.

## Sweeps and figures

[analysis.py](analysis.py) runs both tabular methods over hyperparameter grids
(`n ∈ {1,2,5,10}`, `epsilon ∈ {0.01,0.05,0.1,0.3}`, `alpha ∈ {0.1,0.01,0.001}`,
250k episodes, `N = 2`) and saves the tracked-state Q-trajectories and cumulative
win ratios to [images/](images/).

## Board rendering

`env.render(p1, p2)` prints the 8x3 board: `X` = player 1, `O` = player 2,
`R` = rosette, `S` / `E` = start / end squares, `-` = empty. It takes only
positions in `1..14`, so filter out home and scored pieces first (see
`active_board_positions` in [play_test.py](play_test.py#L6)).

## Repository layout

- [start_action/](start_action/) — snapshot of the earlier design, where an
  action was just the starting cell rather than a `(start, destination)` pair.
- [DQN_training.py](DQN_training.py) — scratch file, not part of the pipeline.
