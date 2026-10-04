"""
Cheap evaluation ladder for a trained policy. No downloads, no dependencies
beyond what training already needed.

    python eval_ladder.py --games 2000 --seed 0

Rungs, in increasing order of strength:
    random      uniform over legal moves
    heuristic   score > capture > rosette > furthest advance
    old_net     the episode-1000 snapshot, if old_net.pt exists

Use this as the smoke test before investing in the 827MB solved lookup table:
if the policy cannot beat random convincingly, something is wrong with the
checkpoint or the encoding and nothing downstream will mean anything.
"""

import argparse
import os
import time

import numpy as np

from eval_core import (
    evaluate,
    format_row,
    greedy_policy,
    heuristic_policy,
    load_policy,
    random_policy,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="final_policy.pt")
    parser.add_argument("--games", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-swap", action="store_true",
                        help="play every game with the policy as P1")
    args = parser.parse_args()

    net = load_policy(args.policy)
    policy = greedy_policy(net)

    opponents = [
        ("vs random", random_policy(np.random.default_rng(args.seed))),
        ("vs heuristic", heuristic_policy()),
    ]

    if os.path.exists("old_net.pt"):
        opponents.append(("vs old_net (ep. 1000)", greedy_policy(load_policy("old_net.pt"))))
    else:
        print("note: old_net.pt not found, skipping the training-progress check\n")

    print(f"{args.policy}  |  {args.games} games/opponent  |  seed {args.seed}")
    print(f"{'opponent':<28} {'win rate':<7} {'95% CI':<18} record")
    print("-" * 78)

    for label, opponent in opponents:
        t0 = time.perf_counter()
        res = evaluate(
            policy, opponent,
            n_games=args.games,
            seed=args.seed,
            swap_sides=not args.no_swap,
        )
        print(format_row(label, res) + f"   {time.perf_counter() - t0:.1f}s")

    # A reference point: how do the baselines do against each other? This tells
    # you how much of the policy's win rate is skill rather than the heuristic
    # simply being weak.
    print()
    res = evaluate(
        heuristic_policy(), random_policy(np.random.default_rng(args.seed + 1)),
        n_games=args.games, seed=args.seed, swap_sides=not args.no_swap,
    )
    print(format_row("(heuristic vs random)", res))


if __name__ == "__main__":
    main()
