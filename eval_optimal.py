"""
Measure a trained policy against perfect play, using the solved Finkel lookup
table from RoyalUr.net.

    python eval_optimal.py --games 500 --lut finkel.rgu

Two metrics are reported:

1. Win rate vs. the optimal agent. Honest but blunt: Ur is a dice game, so a
   single game carries about one bit of information and even a strong policy
   will lose heavily to perfect play.

2. Move quality, which is why the LUT beats any other opponent. At every
   decision the table gives the win probability of *every* legal move, so each
   move yields a real-valued score rather than one bit per game:

     agreement   how often the policy picks a LUT-optimal move
     regret      mean win-probability given up per decision
     blunders    decisions costing more than a threshold, dumped for inspection

   Regret is also broken down by game phase, which localises weakness in a way
   a win rate cannot.
"""

import argparse
import time

import numpy as np

from enviroment import RoyalGameOfUr
from eval_core import (
    N_PIECES,
    evaluate,
    format_row,
    greedy_policy,
    load_policy,
    random_policy,
    wilson_interval,
)
from ur_bridge import SolvedUr, lut_policy, move_values

SCORED_CELL = 15


def analyse_moves(net, solved, n_games=200, seed=42, blunder_threshold=0.05,
                  opponent="lut", max_steps=2000):
    """
    Play games and score every decision the policy makes against the LUT.

    The policy plays P1; `opponent` selects P2's play. Scoring the policy's
    moves does not depend on who the opponent is, but playing against the
    optimal agent keeps the sampled positions realistic for strong play.
    """
    env = RoyalGameOfUr(N_PIECES)
    policy = greedy_policy(net)

    if opponent == "lut":
        p2_policy = lut_policy(solved, env)
    elif opponent == "random":
        p2_policy = random_policy(np.random.default_rng(seed))
    else:
        p2_policy = None

    regrets = []
    optimal_hits = 0
    decisions = 0
    blunders = []
    # Progress is measured by total pieces advanced, as a proxy for game phase.
    phase_regret = {"opening": [], "midgame": [], "endgame": []}

    wins = 0
    finished = 0

    for g in range(n_games):
        state, _ = env.reset(seed=seed + g)
        legal_actions = env.get_legal_moves()

        for _ in range(max_steps):
            # See the note in eval_core.play_match: env.step sets this itself,
            # but not until it is called, so a LUT opponent would read it stale.
            env.current_player = 1

            # Only score real decisions: a forced move carries no information.
            if len(legal_actions) > 1:
                mover = list(env.player1_loc)
                other = list(env.player2_loc)
                values = move_values(solved, mover, other, legal_actions)

                best = max(values)
                chosen_action = policy(state, legal_actions)
                chosen_value = values[legal_actions.index(chosen_action)]

                regret = best - chosen_value
                regrets.append(regret)
                decisions += 1
                if regret <= 1e-9:
                    optimal_hits += 1

                progress = (sum(mover) + sum(other)) / (2 * N_PIECES * SCORED_CELL)
                phase = ("opening" if progress < 0.33
                         else "midgame" if progress < 0.66
                         else "endgame")
                phase_regret[phase].append(regret)

                if regret > blunder_threshold:
                    blunders.append({
                        "game": g,
                        "p1": mover,
                        "p2": other,
                        "roll": int(env.roll),
                        "chosen": chosen_action,
                        "best": legal_actions[int(np.argmax(values))],
                        "regret": regret,
                    })
                action = chosen_action
            else:
                action = legal_actions[0]

            state, reward, terminated, truncated, info = env.step(
                action, opponent_policy=p2_policy
            )

            if terminated:
                finished += 1
                if all(p == SCORED_CELL for p in env.player1_loc):
                    wins += 1
                break

            legal_actions = info["legal_actions"]

    return {
        "decisions": decisions,
        "agreement": optimal_hits / decisions if decisions else 0.0,
        "mean_regret": float(np.mean(regrets)) if regrets else 0.0,
        "median_regret": float(np.median(regrets)) if regrets else 0.0,
        "p90_regret": float(np.percentile(regrets, 90)) if regrets else 0.0,
        "blunders": sorted(blunders, key=lambda b: -b["regret"]),
        "blunder_rate": len(blunders) / decisions if decisions else 0.0,
        "phase_regret": {k: (float(np.mean(v)) if v else 0.0, len(v))
                         for k, v in phase_regret.items()},
        "wins": wins,
        "games": finished,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="final_policy.pt")
    parser.add_argument("--lut", default="finkel.rgu")
    parser.add_argument("--games", type=int, default=500)
    parser.add_argument("--analysis-games", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--blunder-threshold", type=float, default=0.05)
    parser.add_argument("--show-blunders", type=int, default=5)
    parser.add_argument("--skip-sanity", action="store_true")
    args = parser.parse_args()

    print(f"loading {args.lut} (this takes a moment, it is ~827MB) ...")
    t0 = time.perf_counter()
    solved = SolvedUr(args.lut)
    print(f"loaded {len(solved._keys):,} positions in {time.perf_counter() - t0:.1f}s")

    start_p = solved.win_prob([0] * N_PIECES, [0] * N_PIECES, light_to_move=True)
    print(f"perfect-play win probability for the first player: {start_p:.4f}\n")

    net = load_policy(args.policy)
    policy = greedy_policy(net)

    # Sanity gate: the optimal agent must crush random. If it does not, the
    # bridge is wrong and every other number here is meaningless.
    if not args.skip_sanity:
        env_sanity = RoyalGameOfUr(N_PIECES)
        print("sanity check: optimal agent vs random ...")
        res = evaluate(
            lut_policy(solved, env_sanity),
            random_policy(np.random.default_rng(args.seed)),
            n_games=100, seed=args.seed, env=env_sanity,
        )
        print(format_row("  LUT vs random", res))
        if res["rate"] < 0.95:
            print("\n  WARNING: the optimal agent is not dominating random play.")
            print("  The bridge is probably wrong; do not trust the numbers below.\n")
        else:
            print("  ok\n")

    # 1. Win rate against perfect play.
    env_eval = RoyalGameOfUr(N_PIECES)
    print(f"win rate vs optimal play ({args.games} games) ...")
    t0 = time.perf_counter()
    res = evaluate(policy, lut_policy(solved, env_eval), n_games=args.games,
                   seed=args.seed, env=env_eval)
    print(format_row("  policy vs LUT", res) + f"   {time.perf_counter() - t0:.1f}s")

    # A reference point: how does random do against perfect play? This is the
    # floor, and it makes the policy's number interpretable.
    env_ref = RoyalGameOfUr(N_PIECES)
    ref = evaluate(
        random_policy(np.random.default_rng(args.seed + 7)),
        lut_policy(solved, env_ref),
        n_games=max(100, args.games // 5), seed=args.seed, env=env_ref,
    )
    print(format_row("  random vs LUT (floor)", ref))

    # 2. Move-quality analysis.
    print(f"\nmove quality over {args.analysis_games} games ...")
    t0 = time.perf_counter()
    stats = analyse_moves(
        net, solved,
        n_games=args.analysis_games,
        seed=args.seed + 1000,
        blunder_threshold=args.blunder_threshold,
    )
    print(f"  {stats['decisions']:,} scored decisions in {time.perf_counter() - t0:.1f}s")
    print(f"  optimal-move agreement : {stats['agreement']:.3f}")
    print(f"  mean regret            : {stats['mean_regret']:.4f} win probability / decision")
    print(f"  median regret          : {stats['median_regret']:.4f}")
    print(f"  90th pct regret        : {stats['p90_regret']:.4f}")
    print(f"  blunder rate (>{args.blunder_threshold:.2f})   : {stats['blunder_rate']:.3f}")

    print("\n  regret by phase:")
    for phase in ("opening", "midgame", "endgame"):
        mean, count = stats["phase_regret"][phase]
        print(f"    {phase:<9} {mean:.4f}   ({count:,} decisions)")

    if args.show_blunders and stats["blunders"]:
        print(f"\n  worst {min(args.show_blunders, len(stats['blunders']))} blunders:")
        for b in stats["blunders"][:args.show_blunders]:
            print(f"    roll {b['roll']}  chose {b['chosen']} instead of {b['best']}"
                  f"  (-{b['regret']:.3f} win prob)")
            print(f"      P1 {sorted(b['p1'])}")
            print(f"      P2 {sorted(b['p2'])}")


if __name__ == "__main__":
    main()
