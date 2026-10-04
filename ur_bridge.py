"""
Bridge between this repo's `enviroment.py` and the solved Finkel lookup table
published by RoyalUr.net.

Why this works at all
---------------------
`enviroment.py` encodes each player's pieces as a linear track 0..15:
    0 = home (not entered), 1..14 = the 14 path squares, 15 = scored.
RoyalUr's Bell path is also 14 tiles per player, so our cell `c` in 1..14 is
exactly path index `c - 1`. Verified geometry (royalur.model.path.BellPathPair):

    light path: (1,4)(1,3)(1,2)(1,1) (2,1)(2,2)(2,3)(2,4)(2,5)(2,6)(2,7)(2,8) (1,8)(1,7)
    dark  path: (3,4)(3,3)(3,2)(3,1) (2,1)(2,2)(2,3)(2,4)(2,5)(2,6)(2,7)(2,8) (3,8)(3,7)

The shared middle column x=2 is path indices 4..11, i.e. our cells 5..12 --
precisely `public_cells` in enviroment.py:30. Rosettes at our cells 4, 8, 14 map
to path indices 3, 7, 13 = tiles (1,1)/(3,1), (2,4), (1,7)/(3,7), which are the
standard board's rosettes. So the rulesets coincide and the LUT is valid here,
**at N = 7 only** (the LUT's middle-lane compression hardcodes 7 pieces a side).

Rather than construct royalur `Game` objects (LutAgent.play copies a Game per
candidate move, which is far too slow for thousands of games, and it also
returns None in a completely lost position because it initialises
highest_value = 0 and tests `value > highest_value`), this module encodes our
position lists straight into a LUT key, replicating
SimpleGameStateEncoding.encode_board bit for bit.

Key layout (royalur/lut/board_encoder.py:82-99):
    bits  0-5   right lane  (dark's private squares, x=3)
    bits  6-18  middle lane (compressed, 13 bits)
    bits 19-24  left lane   (light's private squares, x=1)
    bits 25-27  dark pieces still at home
    bits 28-30  light pieces still at home
Values are uint16 win probabilities for LIGHT: 0 = light loses, 65535 = light wins.
The table only stores light-to-move positions, so a dark-to-move position is
encoded with the two sides swapped and the value read back as 65535 - value.
"""

import numpy as np

from royalur.lut.board_encoder import SimpleGameStateEncoding
from royalur.lut.reader import LutReader

N_PIECES = 7
HOME_CELL = 0
SCORED_CELL = 15
MAX_VALUE = 65535

# Our cells 1..14 are path indices 0..13. The side lanes are the parts of the
# path outside the shared middle: path indices 0..3 and 12..13, i.e. our cells
# 1..4 and 13..14.
#
# The bit ORDER is not the path order. encode_side_lane (board_encoder.py:65-80)
# walks the board column top-to-bottom, bit `index` = board row 0,1,2,3,6,7 --
# but the Bell path enters at board row 3 and runs *up* the column, so the path
# visits those rows in the order 3,2,1,0 then 7,6:
#
#   bit index   0     1     2     3     4     5
#   board row   0     1     2     3     6     7
#   path idx    3     2     1     0     13    12
#   our cell    4     3     2     1     14    13
#
# Getting this backwards still produces valid-looking keys that exist in the
# table, so it fails silently -- hence the cross-check against the library's own
# encoder in check_equivalence().
SIDE_LANE_CELLS = (4, 3, 2, 1, 14, 13)     # -> encode_side_lane's bit 0..5
MIDDLE_LANE_CELLS = (5, 6, 7, 8, 9, 10, 11, 12)   # -> middle lane index 0..7


class SolvedUr:
    """
    Wraps the solved Finkel LUT and exposes win probabilities for positions
    expressed in `enviroment.py` coordinates.
    """

    def __init__(self, lut_path="finkel.rgu"):
        self._encoding = SimpleGameStateEncoding()
        self._middle_compression = self._encoding.middle_lane_compression

        lut = LutReader(lut_path).read()
        self._lut = lut
        # Sorted key array + value array, so lookups can be done with a binary
        # search instead of the library's per-call Python lookup.
        #
        # The file stores big-endian ('>u4'/'>u2') and the arrays are read-only
        # views onto it. np.searchsorted on a byte-swapped dtype falls off the
        # fast path and is thousands of times slower, so convert once, up front,
        # to native order. This costs ~550MB + ~275MB of RAM and a few seconds,
        # and turns a lookup from milliseconds into microseconds.
        self._keys = np.ascontiguousarray(lut.keys_as_numpy(0), dtype=np.uint32)
        self._values = np.ascontiguousarray(lut.values_as_numpy(0), dtype=np.uint16)

    # ---------------------------------------------------------------- encoding

    def encode(self, light_loc, dark_loc):
        """
        Encode a light-to-move position into a LUT key.

        light_loc / dark_loc: iterables of 7 cells in 0..15 (our coordinates).
        """
        light = list(light_loc)
        dark = list(dark_loc)

        light_on = set(c for c in light if HOME_CELL < c < SCORED_CELL)
        dark_on = set(c for c in dark if HOME_CELL < c < SCORED_CELL)

        # Side lanes: 1 bit per square, occupancy only (a player's own lane can
        # only ever hold their own pieces).
        left_lane = 0    # light's private squares
        right_lane = 0   # dark's private squares
        for index, cell in enumerate(SIDE_LANE_CELLS):
            if cell in light_on:
                left_lane |= 1 << index
            if cell in dark_on:
                right_lane |= 1 << index

        # Middle lane: 2 bits per square, 1 = dark, 2 = light.
        middle = 0
        for index, cell in enumerate(MIDDLE_LANE_CELLS):
            if cell in dark_on:
                middle |= 1 << (2 * index)
            elif cell in light_on:
                middle |= 2 << (2 * index)

        compressed = self._middle_compression[middle]
        if compressed == -1:
            raise ValueError(f"Illegal middle lane state: {middle}")

        state = right_lane | (compressed << 6) | (left_lane << 19)
        state |= sum(1 for c in dark if c == HOME_CELL) << 25
        state |= sum(1 for c in light if c == HOME_CELL) << 28
        return state

    # ---------------------------------------------------------------- lookups

    def _raw_lookup(self, key):
        idx = np.searchsorted(self._keys, np.uint32(key))
        if idx >= len(self._keys) or self._keys[idx] != key:
            raise KeyError(f"Position {key} not present in the lookup table")
        return int(self._values[idx])

    def _raw_lookup_many(self, keys):
        """Vectorised form of _raw_lookup for a batch of keys."""
        arr = np.asarray(keys, dtype=np.uint32)
        idx = np.searchsorted(self._keys, arr)
        idx_clipped = np.clip(idx, 0, len(self._keys) - 1)
        if not np.array_equal(self._keys[idx_clipped], arr):
            missing = arr[self._keys[idx_clipped] != arr]
            raise KeyError(f"Positions not present in the lookup table: {missing[:5]}")
        return self._values[idx_clipped].astype(np.float64)

    def win_prob(self, light_loc, dark_loc, light_to_move=True):
        """
        Probability that the player to move wins, in [0, 1], under perfect play
        by both sides.

        Terminal positions are answered directly: the table stores only
        positions with moves remaining.
        """
        light = list(light_loc)
        dark = list(dark_loc)

        if all(c == SCORED_CELL for c in light):
            return 1.0 if light_to_move else 0.0
        if all(c == SCORED_CELL for c in dark):
            return 0.0 if light_to_move else 1.0

        if light_to_move:
            value = self._raw_lookup(self.encode(light, dark))
        else:
            # Swap the sides so the mover is "light", then the returned value is
            # already from the mover's point of view.
            value = self._raw_lookup(self.encode(dark, light))

        return value / MAX_VALUE


# ---------------------------------------------------------------------- policy

def _apply_move(mover_loc, other_loc, action, public_cells=frozenset(range(5, 13))):
    """
    Apply `action` to a copy of the position, mirroring enviroment.update_board
    (enviroment.py:120-146): move the piece, and capture an enemy piece if the
    destination is in the shared middle lane.
    """
    mover = list(mover_loc)
    other = list(other_loc)
    start, dest = action
    if start == dest:            # pass action (15, 15)
        return mover, other

    mover[mover.index(start)] = dest
    if dest in public_cells and dest in other:
        other[other.index(dest)] = HOME_CELL
    return mover, other


def move_values(solved, mover_loc, other_loc, legal_actions, rosettes=(4, 8, 14)):
    """
    Win probability *for the mover* after each legal action, as a list aligned
    with `legal_actions`.

    Landing on a rosette grants another turn, so the resulting position is still
    the mover's to play; otherwise the turn passes and the value must be flipped
    to stay in the mover's frame.
    """
    values = [None] * len(legal_actions)
    keys = []
    slots = []     # (index into values, whether the value needs flipping)

    for i, action in enumerate(legal_actions):
        after_mover, after_other = _apply_move(mover_loc, other_loc, action)

        if all(c == SCORED_CELL for c in after_mover):
            values[i] = 1.0
            continue

        start, dest = action
        keeps_turn = (start != dest) and (dest in rosettes)

        if keeps_turn:
            # Still the mover's turn: the value is already in the mover's frame.
            keys.append(solved.encode(after_mover, after_other))
            slots.append((i, False))
        else:
            # Opponent to move: encode from their side and flip the result.
            keys.append(solved.encode(after_other, after_mover))
            slots.append((i, True))

    if keys:
        raw = solved._raw_lookup_many(keys) / MAX_VALUE
        for (i, flip), v in zip(slots, raw):
            values[i] = (1.0 - v) if flip else v

    return values


def lut_policy(solved, env):
    """
    An optimal opponent in the `opponent_policy(raw_state, legal_actions)` form
    that `enviroment.step` expects.

    Which side is moving is worked out from the position itself rather than from
    env.current_player: as P1 the policy is called by the caller's loop *before*
    env.step() sets current_player = 1, so that flag is stale at exactly the
    moment it would be consulted. The legal actions, on the other hand, are
    always generated for the mover, so the side whose piece list contains every
    move's start cell is by definition the one to play.

    raw_state is not usable here either -- it is sorted, so it cannot say which
    block belongs to which player once the policy is playing as P2.
    """

    def policy(raw_state, legal_actions):
        if len(legal_actions) == 1:
            return legal_actions[0]

        starts = set(start for start, _ in legal_actions)
        p1_ok = starts.issubset(set(env.player1_loc))
        p2_ok = starts.issubset(set(env.player2_loc))

        if p1_ok and not p2_ok:
            mover, other = env.player1_loc, env.player2_loc
        elif p2_ok and not p1_ok:
            mover, other = env.player2_loc, env.player1_loc
        else:
            # Ambiguous (both sides could own these starts): fall back to the
            # environment's own idea of whose turn it is.
            if env.current_player == 1:
                mover, other = env.player1_loc, env.player2_loc
            else:
                mover, other = env.player2_loc, env.player1_loc

        values = move_values(solved, mover, other, legal_actions)
        return legal_actions[int(np.argmax(values))]

    return policy


# ------------------------------------------------------------ equivalence test

def _build_royalur_game(light_loc, dark_loc, roll):
    """
    Build a royalur Game whose board matches our position, with `roll` already
    rolled and light to move. Used only by the verification gate -- it is far
    too slow for evaluation, which is the whole reason this module encodes
    positions directly.
    """
    from royalur.game import Game
    from royalur.model.board import Piece
    from royalur.model.path import BellPathPair
    from royalur.model.player import PlayerType

    game = Game.create_finkel()
    state = game.get_current_state()
    board = state.board
    board.clear()

    paths = BellPathPair()
    for loc_list, player, path in (
        (light_loc, PlayerType.LIGHT, paths.light),
        (dark_loc, PlayerType.DARK, paths.dark),
    ):
        on_board = 0
        for cell in loc_list:
            if HOME_CELL < cell < SCORED_CELL:
                board.set(path[cell - 1], Piece(player, cell - 1))
                on_board += 1
        player_state = state.light_player if player == PlayerType.LIGHT else state.dark_player
        player_state._piece_count = sum(1 for c in loc_list if c == HOME_CELL)
        player_state._score = sum(1 for c in loc_list if c == SCORED_CELL)

    return game, state


def check_equivalence(solved, n_positions=10_000, seed=0, verbose=True):
    """
    The gate that must pass before any number from eval_optimal.py means
    anything.

    Two independent checks at every position reached by random play:

      (a) The position encodes and is present in the table. A miss means our
          rules let the environment reach a position the solved game considers
          impossible, i.e. the rulesets differ.

      (b) Our direct encoding agrees exactly with the key that the library's own
          SimpleGameStateEncoding produces from a real royalur Game built to the
          same position. This is the check that actually validates the bit
          layout, independently of our own assumptions.

    Note what is deliberately NOT checked: win_prob(A, B) + win_prob(B, A) == 1.
    Those are two different positions -- both ask "what does the player to move
    score?", and in both the named player moves next. From the start position
    both are 0.5154, which is the published first-player advantage, not a bug.
    """
    from enviroment import RoyalGameOfUr

    env = RoyalGameOfUr(N_PIECES)
    rng = np.random.default_rng(seed)
    checked = 0
    encoding_mismatches = 0
    failures = []

    # Cross-checking against a real Game object is slow, so only do it on a
    # sample; the presence check runs on every position.
    cross_check_every = max(1, n_positions // 300)

    while checked < n_positions:
        env.reset(seed=int(rng.integers(1 << 30)))

        for _ in range(500):
            if all(c == SCORED_CELL for c in env.player1_loc) or \
               all(c == SCORED_CELL for c in env.player2_loc):
                break

            try:
                solved.win_prob(env.player1_loc, env.player2_loc, True)
                solved.win_prob(env.player2_loc, env.player1_loc, True)

                if checked % cross_check_every == 0:
                    ours = solved.encode(env.player1_loc, env.player2_loc)
                    _, state = _build_royalur_game(
                        env.player1_loc, env.player2_loc, env.roll
                    )
                    theirs = solved._encoding.encode_board(state.board)
                    theirs |= state.dark_player._piece_count << 25
                    theirs |= state.light_player._piece_count << 28
                    if ours != theirs:
                        encoding_mismatches += 1
                        failures.append((
                            list(env.player1_loc), list(env.player2_loc),
                            f"encoding mismatch: ours={ours} theirs={theirs}", None,
                        ))
            except (KeyError, ValueError) as exc:
                failures.append((list(env.player1_loc), list(env.player2_loc), repr(exc), None))

            checked += 1
            if checked >= n_positions:
                break

            legal = env.get_legal_moves()
            action = legal[int(rng.integers(len(legal)))]
            try:
                _, _, terminated, _, info = env.step(action)
            except ValueError:
                break
            if terminated:
                break

    if verbose:
        print(f"checked {checked} positions, {len(failures)} failures "
              f"({encoding_mismatches} encoding mismatches)")
        for f in failures[:10]:
            print("  ", f)

    return checked, failures


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--lut", default="finkel.rgu")
    parser.add_argument("--positions", type=int, default=10_000)
    args = parser.parse_args()

    print(f"loading {args.lut} ...")
    solved = SolvedUr(args.lut)
    print("metadata:", solved._lut.get_metadata())
    print(f"table size: {len(solved._keys):,} positions")

    start = solved.win_prob([0] * 7, [0] * 7, light_to_move=True)
    print(f"\nwin probability for the player to move from the start: {start:.4f}")

    checked, failures = check_equivalence(solved, n_positions=args.positions)
    print("\nPASS" if not failures else "\nFAIL")
