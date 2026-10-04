import numpy as np
import gymnasium as gym
from gymnasium import spaces
import sys


# Royal game of Ur
class RoyalGameOfUr(gym.Env):

    def __init__(self, N):
        super().__init__()
        if N <= 0: 
            raise ValueError("N must be a positive integer")

        # Number of pieces per player.
        self.N = N

        # Scale for the potential-based shaping term (see potential() / step()).
        # The raw potential can swing by ~tens of cells over a game, while the
        # terminal win reward is +/-1; scaling the shaping down keeps it a gentle
        # guidance signal rather than something that dominates the win/loss reward.
        self.shaping_scale = 0.01 *(1/self.N)

        # Board constants used by the transition logic. Positions are encoded as:
        # 0 = not entered, 1..14 = board path, 15 = scored.
        self.home_cell = 0
        self.scored_cell = 15
        
        self.private_cells = tuple(list(range(0, 5)) + [13, 14, 15])
        self.public_cells = tuple(range(5, 13))

        self.rosettes = (4, 8, 14)
        self.all_cells = tuple(range(0, 16))

        # 2*N is the total number of pieces for both players
        # Each piece position has len(self.all_cells) possible values (0..15),
        # and the dice roll has 5 possible values (0..4).
        self.observation_space = spaces.MultiDiscrete([len(self.all_cells)] * (2 * self.N) + [5])

        # Actions are pairs (start, destination), following the assignment formulation.
        # The pass action is represented by (15, 15).
        self.pass_action = (self.scored_cell, self.scored_cell)
        self.action_space = spaces.MultiDiscrete([len(self.all_cells), len(self.all_cells)])

        self.reset()
        
    def roll_dice(self):
        return int(self.np_random.binomial(4, 0.5))

    def reset(self, seed=None):
        super().reset(seed=seed)

        self.player1_loc = [self.home_cell] * self.N
        self.player2_loc = [self.home_cell] * self.N
        self.current_player = 1
        self.last_landed_position = None
        self.roll = self.roll_dice()
        self.state = self._make_state()

        return self.state, {}

    def _make_state(self):
        """
        Build the observation the agent keys on. Piece positions are sorted per
        player so that game-equivalent positions (e.g. [3, 7] and [7, 3]) map to a
        single state. This is purely a relabelling of the observation: the internal
        player1_loc / player2_loc lists keep their order for the transition logic,
        so update_board's .index() lookups are unaffected. Collapsing these
        permutation-symmetric states roughly halves the effective state space and
        multiplies the number of visits per state, which a tabular method needs.
        """
        return np.array(
            sorted(self.player1_loc) + sorted(self.player2_loc) + [self.roll],
            dtype=np.int64,
        )

    def check_win(self):
        """
        Checks if current player won
        """
        pieces = self.player1_loc if self.current_player == 1 else self.player2_loc
        return all(piece == self.scored_cell for piece in pieces)

    def normalize_action(self, action):
        """
        Convert a Gymnasium action to a (start, destination) tuple of ints.
        """
        if isinstance(action, np.ndarray):
            action = action.tolist()
        if isinstance(action, (list, tuple)) and len(action) == 2:
            return int(action[0]), int(action[1])
        raise ValueError(f"Action must be a (start, destination) pair, got {action}")

    def get_legal_moves(self):
        """
        Gets legal actions for current player
        """
        pieces = self.player1_loc if self.current_player == 1 else self.player2_loc
        enemy_pieces= self.player2_loc if self.current_player == 1 else self.player1_loc
        legal_actions = []
        shared_rosette= next(iter(set(self.public_cells) & set(self.rosettes)))
        for start in sorted(set(pieces)):
            if self.roll == 0 or start == self.scored_cell:
                continue

            destination = start + self.roll
            if destination > self.scored_cell:
                continue

            # The only pieces that can share the same cells are the ones in the home or score cell; otherwise the action is illegal.
            if destination not in (self.home_cell, self.scored_cell) and destination in pieces:
                continue
            if destination==shared_rosette and shared_rosette in enemy_pieces: 
                continue

            legal_actions.append((start, destination))

        return legal_actions or [self.pass_action]

    def update_board(self, action):
        """
        Update current player pieces positions.
        Checks if current player captured any enemy pieces and in case resets their position
        """
        self.last_landed_position = None
        action = self.normalize_action(action)
        if action == self.pass_action:
            return

        own_pieces = self.player1_loc if self.current_player == 1 else self.player2_loc
        enemy_pieces = self.player2_loc if self.current_player == 1 else self.player1_loc

        start, destination = action
        expected_destination = start + self.roll
        if expected_destination > self.scored_cell:
            raise ValueError(f"Illegal overshoot from {start} with roll {self.roll}")
        if destination != expected_destination:
            raise ValueError(
                f"Illegal destination {destination} from {start} with roll {self.roll}; expected {expected_destination}"
            )

        own_pieces[own_pieces.index(start)] = destination
        self.last_landed_position = destination

        if destination in self.public_cells and destination in enemy_pieces:
            enemy_pieces[enemy_pieces.index(destination)] = self.home_cell

    def is_on_rosette(self):
        """
        Checks if current player is on a rosette, if not updates the current player
        """
        return self.last_landed_position in self.rosettes
                      
    def sample_legal_action(self):
        legal_actions = self.get_legal_moves()
        return legal_actions[int(self.np_random.integers(len(legal_actions)))]

    def move_p2(self,opponent_policy=None):
        self.current_player = 2

        while self.current_player == 2:
            self.roll = self.roll_dice()
            if opponent_policy is not None:
                p2_state = np.array(sorted(self.player2_loc) + sorted(self.player1_loc) + [self.roll],dtype=np.int64)
                legal_moves = self.get_legal_moves()
                p2_action = opponent_policy(p2_state,legal_moves)

            
            else:
                p2_action = self.sample_legal_action()
            self.update_board(p2_action)

            if self.check_win():
                return True

            if not self.is_on_rosette():
                self.current_player = 1

        return False

    def potential(self):
        """
        Potential Phi(s) for potential-based reward shaping: P1's total board
        progress minus P2's, summed over pieces (0 = home, 1..14 = path, 15 =
        scored). Higher means P1 is further ahead. Used to add a dense shaping
        term Phi(s') - Phi(s) to the sparse +/-1 win reward.

        Potential-based shaping (Ng, Harada & Russell 1999) is provably
        policy-invariant: adding gamma*Phi(s') - Phi(s) to the reward does not
        change which policy is optimal, it only densifies the learning signal so
        the agent gets feedback every move (advancing/scoring is good, getting
        captured back to home is bad) instead of only at the terminal step.
        """
        return float(sum(self.player1_loc) - sum(self.player2_loc))

    def step(self, action,opponent_policy=None):
        self.current_player = 1
        action = self.normalize_action(action)
        legal_actions = self.get_legal_moves()

        if action not in legal_actions:
            raise ValueError(f"Illegal action {action} in state {self.state.tolist()}")

        # Potential before the transition, for potential-based reward shaping.
        phi_before = self.potential()

        self.update_board(action)
        terminated = self.check_win()
        # +1 for a P1 win, -1 for a P2 win, 0 otherwise. The loss penalty gives the
        # agent a gradient to avoid positions that let P2 finish the game.
        reward = 1 if terminated else 0

        if not terminated:
            if self.is_on_rosette():
                self.roll = self.roll_dice()
            else:
                p2_won = self.move_p2(opponent_policy)
                if p2_won:
                    terminated = True
                    reward = -1
                else:
                    self.roll = self.roll_dice()

        self.current_player = 1
        self.state = self._make_state()

        # Add the shaping term gamma*Phi(s') - Phi(s) (gamma = 1). On terminal
        # transitions Phi(s') is taken as 0 by convention, which keeps the shaped
        # return's optimal policy identical to the unshaped one.
        phi_after = phi_before if terminated else self.potential()
        reward = reward + self.shaping_scale * (phi_after - phi_before)

        info = {"legal_actions": [] if terminated else self.get_legal_moves()}
        return self.state, reward, terminated, False, info



    def render(self, player1_loc, player2_loc):
        """
        The input for this render functions are lists or sets player1_loc and player2_loc.
        These should contain the location of the pieces for the respective player.
        The location should be denoted by the number of the square on the path of that player.
        More precisely, we can number the squares for both players on their path from 1 to 14, with 1 the first square on the
        path, and 14 the last square on the path, which is the final one before leaving the board.
        If player1_loc then equals [2, 4] for example, then he has two pieces in play. The first is on square 2, i.e., the
        second square on their path, while the second is on square 4, i.e., the fourth square on their path.
        """
        
        # Obtain states
        seq_1 = [(i, 1) for i in range(4, 0, -1)] + [(i, 2) for i in range(1, 9)] + [(8, 1), (7, 1)]
        seq_2 = [(i, 3) for i in range(4, 0, -1)] + [(i, 2) for i in range(1, 9)] + [(8, 3), (7, 3)]
        self.encode_states = [{idx: s for s, idx in zip(seq_1, range(1, 15))},
                          {idx: s for s, idx in zip(seq_2, range(1, 15))}]
        board_states_1 = [self.encode_states[0][i] for i in player1_loc]
        board_states_2 = [self.encode_states[1][i] for i in player2_loc]
        
        
        outfile = sys.stdout
        for i in range(1, 9):
            for j in range(1, 4):
                if (i, j) in board_states_1:
                    output = " X "
                elif (i, j) in board_states_2:
                    output = " O "
                elif (i, j) in [(5, 1), (5, 3)]:
                    output = " S "
                elif (i, j) in [(6, 1), (6, 3)]:
                    output = " E "
                elif (i, j) in [(1, 1), (1, 3), (7, 1), (7, 3), (4, 2)]:
                    output = " R "
                else:
                    output = " - "
    
                if j == 1:
                    output = output.lstrip()
                if j == 3:
                    output = output.rstrip()
                    output += "\n"
    
                outfile.write(output)
        outfile.write("\n")

               
        return

