from enviroment import RoyalGameOfUr
# eval_core rather than DQN_network: importing the latter runs the whole
# training setup (builds two nets, resets its own env, opens a pyplot window).
from eval_core import greedy_policy, load_policy

env = RoyalGameOfUr(7)
state, info = env.reset()

def active_board_positions(positions):
    return [position for position in positions if 1 <= position <= 14]


def describe_action(env, action):
    action = tuple(action)
    if action == env.pass_action:
        return "pass"
    return f"move from {action[0]} to {action[1]}"

# creating our opponent by loading the policy

bot = greedy_policy(load_policy("final_policy.pt"))
#Initliazie the match parameters
terminated = False
truncated = False
total_reward = 0
step_count = 0
reward = 0
winner = "none"

# ACTUAL MATCH 

def ask_action(legal_moves):
    """Read the *index* of a move in legal_moves from the terminal."""
    options = "  ".join(f"[{i}] {describe_action(env, m)}" for i, m in enumerate(legal_moves))
    while True:
        raw = input(f"Your turn. {options}\nMove index: ")
        # int() has to be inside the try: a non-numeric entry raises ValueError
        # here, not at the indexing step, and would otherwise kill the game.
        try:
            idx = int(raw)
            if idx < 0:
                raise IndexError
            return legal_moves[idx]
        except (ValueError, IndexError):
            print(f"Please type a move index between 0 and {len(legal_moves) - 1}.")


while not terminated:
    legal_moves = env.get_legal_moves()
    action = ask_action(legal_moves)

    print(
        f"Step {step_count + 1}: roll={env.roll}, "
        f"legal={legal_moves}, action={action} ({describe_action(env, action)})"
    )

    observation, reward, terminated, truncated, info = env.step(action,opponent_policy=bot)
    total_reward += reward
    step_count += 1

    print(
        f"Observation: {observation.tolist()}, reward={reward}, "
        f"terminated={terminated}, next_legal={info['legal_actions']}"
    )
    print(f"p1={env.player1_loc}, p2={env.player2_loc}")
    env.render(active_board_positions(env.player1_loc), active_board_positions(env.player2_loc))

    if all(piece == env.scored_cell for piece in env.player1_loc):
        winner = "player 1"
    elif all(piece == env.scored_cell for piece in env.player2_loc):
        winner = "player 2"
    else:
        winner = "none"
print(
    f"Finished after {step_count} steps. "
    f"winner={winner}, total_reward={total_reward}, truncated={truncated}"
)