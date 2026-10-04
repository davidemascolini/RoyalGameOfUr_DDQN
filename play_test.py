import time

from enviroment import RoyalGameOfUr
from tqdm import tqdm 

def active_board_positions(positions):
    return [position for position in positions if 1 <= position <= 14]


def describe_action(env, action):
    action = tuple(action)
    if action == env.pass_action:
        return "pass"
    return f"move from {action[0]} to {action[1]}"


def play_episode(N=3, seed=None, delay=0.5, max_steps=200):
    env = RoyalGameOfUr(N)
    observation, _ = env.reset(seed=seed)
    terminated = False
    truncated = False
    total_reward = 0
    step_count = 0
    reward = 0

    #print(f"Initial observation: {observation.tolist()}")
    #env.render(active_board_positions(env.player1_loc), active_board_positions(env.player2_loc))

    while not (terminated or truncated) and step_count < max_steps:
        legal_actions = env.get_legal_moves()
        action = env.sample_legal_action()

        print(
            f"Step {step_count + 1}: roll={env.roll}, "
            f"legal={legal_actions}, action={action} ({describe_action(env, action)})"
        )

        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step_count += 1

        print(
            f"Observation: {observation.tolist()}, reward={reward}, "
            f"terminated={terminated}, next_legal={info['legal_actions']}"
        )
        print(f"p1={env.player1_loc}, p2={env.player2_loc}")
        env.render(active_board_positions(env.player1_loc), active_board_positions(env.player2_loc))
        time.sleep(delay)

    if step_count >= max_steps and not terminated:
        truncated = True

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
    return winner=="player 1"

if __name__ == "__main__":
    count=0
    for seed in tqdm(range(10)):
        if play_episode(N=3, seed=seed):
            count+=1
    print(f"partite vinte da p1: {count}")
        

