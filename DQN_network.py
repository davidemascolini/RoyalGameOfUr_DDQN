from collections import namedtuple, deque, Counter, defaultdict
from itertools import count
from enviroment import RoyalGameOfUr
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()
# if GPU is to be used
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)
N=7
env = RoyalGameOfUr(N)
#######################################
#          EXPERIENCE REPLAY          #
#######################################

# tuple Transition store all the (s,a,r,s') experiences
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward','next_state_mask'))

#replay memory is the queue from where we sample the experiences
class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
#######################################
#        POLICY & TARGET NETWORK      #
#######################################

class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)
##########################################
# HYPERPARAMTERES & SUPPORTING FUNCTIONS #
##########################################


BATCH_SIZE = 128 # the number of transitions sampled from the replay buffer
GAMMA = 1 # the discount factor as mentioned in the previous section
EPS_START = 0.9 #the starting value of epsilon
EPS_END = 0.01 # the final value of epsilon
EPS_DECAY = 140_000 # controls the rate of exponential decay of epsilon, higher means a slower decay
TAU = 0.005 #  the update rate of the target network
LR = 1e-5 #  the learning rate of the ``AdamW`` optimizer



def encode_state(state, n_pieces= N , board_size=16, dice_size=5):
    """
    state: tuple (x1, ..., xN, y1, ..., yN, d), we want to convert this to an histogram vector 
    of dimension board_size where every position indicates how many pieces you have in that position
    """
    x = state[0:n_pieces]
    y = state[n_pieces:2*n_pieces]
    
    d = state[2*n_pieces]
    
    x_hist = np.bincount(x, minlength=board_size)
    y_hist = np.bincount(y, minlength= board_size)

    dice = np.eye(dice_size)[d]

    return np.concatenate([x_hist, y_hist, dice]).astype(np.float32)


# Build a discrete action index space from the tuple-based actions used by the environment.
all_actions = [
    (start, destination)
    for start in range(env.scored_cell + 1)
    for destination in range(env.scored_cell + 1)
]
action_to_index = {action: idx for idx, action in enumerate(all_actions)}
index_to_action = {idx: action for idx, action in enumerate(all_actions)}
n_actions = len(all_actions)

# Get the number of state observations
state, info = env.reset()
state = encode_state(state)
n_observations = len(state)

policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(100_000)


steps_done = 0


def select_action(state,test=False):
    global steps_done
    sample = random.random()
    steps_done_tensor = torch.tensor(float(steps_done), device=device, dtype=torch.float32)
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        torch.exp(-steps_done_tensor / EPS_DECAY) if not test else 0
    steps_done += 1 if not test else 0

    if sample > eps_threshold:
        with torch.no_grad():
            if isinstance(state, torch.Tensor):
                state_tensor = state
            else:
                state_tensor = torch.tensor(encode_state(state), dtype=torch.float32, device=device).unsqueeze(0)

            legal_actions = env.get_legal_moves()
            legal_indices = [action_to_index[action] for action in legal_actions]
            q_values = policy_net(state_tensor)
            mask = torch.full((q_values.shape[1],), -float("inf"), device=device)
            mask[legal_indices] = 0.0
            masked_q_values = q_values + mask.unsqueeze(0)
            return masked_q_values.max(1).indices.view(1, 1), eps_threshold
    else:
        legal_actions = env.get_legal_moves()
        legal_indices = [action_to_index[action] for action in legal_actions]
        action_idx = random.choice(legal_indices)
        return torch.tensor([[action_idx]], device=device, dtype=torch.long), eps_threshold 


episode_durations = []
# evaluation history: (episode index, win ratio) for every evaluation test
eval_episodes = []
eval_win_ratios = []


def plot_rewards(episode,eps,show_result=False):
    plt.figure(1)
    # Number of completed episodes (episode_durations used as a counter)
    num_plotted = len(episode_durations)
    if num_plotted == 0:
        return
    # Use the rewards array and plot only the completed episodes
    rewards_t = torch.tensor(rewards[:num_plotted], dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        progress = int(episode/num_episodes*100)
        plt.clf()
        plt.title(f'Training... (Progress:{progress} % , current_epsilon: {eps:.3f})')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.plot(rewards_t.numpy())
    # Take 100 episode averages and plot them too
    if num_plotted >= 100:
        means = rewards_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())


def plot_win_ratios():
    """Plot the win ratio measured at every evaluation test (once training is over)."""
    if not eval_win_ratios:
        print("No evaluation test was run, nothing to plot.")
        return
    plt.figure(2)
    plt.clf()
    plt.title('Evaluation: win ratio vs frozen opponent')
    plt.xlabel('Episode')
    plt.ylabel('Win ratio')
    plt.ylim(0, 1)
    plt.axhline(0.5, color='gray', linestyle='--', linewidth=1, label='50% (parity)')
    plt.plot(eval_episodes, eval_win_ratios, marker='o', label='Win ratio')
    plt.legend()
    plt.grid(alpha=0.3)


def make_opponent_policy(net):
    def opponent_policy(raw_state,legal_actions):
        state=encode_state(raw_state)
        #forward pass
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32,device=device).unsqueeze(0)  # aggiunge la dimensione di batch
            q_values = net(state_tensor)
        #time to build the mask for the actions
        legal_indices = [action_to_index[action] for action in legal_actions]
        mask = torch.full((q_values.shape[1],), -float("inf"), device=device)
        mask[legal_indices] = 0.0
        masked_q_values = q_values + mask.unsqueeze(0)
        action_idx=masked_q_values.max(1).indices.view(1, 1)
        return index_to_action[int(action_idx)]
    return opponent_policy

def play(p2_policy,n_matches):
    rewards = np.zeros(n_matches)
    p1_wins=0
    steps = np.zeros(n_matches)
    for i in range(n_matches):
        state, _ = env.reset()
        terminated = False
        step_count = 0
        reward=0
        while not terminated:

            action_idx,_ = select_action(state,test=True)
            action = index_to_action[int(action_idx)]

            observation, reward, terminated, truncated, info = env.step(action,opponent_policy=p2_policy)
            rewards[i] += reward
            step_count += 1
            state = observation
        if all(piece == env.scored_cell for piece in env.player1_loc):
            winner = "player 1"
            p1_wins+=1
        elif all(piece == env.scored_cell for piece in env.player2_loc):
            winner = "player 2"
        else:
            winner = "none"
        '''print(
            f"Finished after {step_count} steps. "
            f"winner={winner}, total_reward={rewards[i]}, duration={step_count}"  
        )'''
        steps[i]= step_count
    return p1_wins/n_matches, np.mean(rewards),np.mean(steps)

##########################################
#        OPTIMIZATION FUNCTION           #
##########################################
def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    # masks need to be filtered in order to be associated jsut with non final states
    next_state_mask_batch = torch.cat([s for s in batch.next_state_mask
                                                if s is not None])
    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1).values
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        #action is chosen based on the policy network
        q_values = policy_net(non_final_next_states)
        masked_q_values = q_values + next_state_mask_batch
        next_actions = masked_q_values.argmax(dim=1, keepdim=True)
        #the selected action is evaluated by the target network (consistent with DDQN mechanics) only on the non final state
        next_state_values[non_final_mask] = target_net(non_final_next_states).gather(1, next_actions).squeeze(1)
    # Compute the expected Q values (the bellman equation)
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()

    # Gradient clipping with a more robust implementation for this setup.
    max_norm = 10
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm)

    optimizer.step()
################################################
#                  TRAINING LOOP               #
################################################

if __name__ == "__main__":
    if torch.cuda.is_available() or torch.backends.mps.is_available():
        num_episodes = 50_000
    else:
        print("GPU NOT AVAILABLE!!!")
        num_episodes = 0
    rewards = np.zeros(num_episodes)
    wins = np.zeros(num_episodes)
    completed_episodes = 0
    #choose your fighter!
    opponents = defaultdict()
    opponents["Random"] = None 
    for i_episode in range(num_episodes):
        # in the initial fase we decide wich opponent policy will be used for training
        opponent = random.choice(list(opponents.values()))
        # Initialize the environment and get its state
        raw_state, info = env.reset()
        state = encode_state(raw_state)
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        step_reward=0
        for t in count():
            action_idx,eps = select_action(state)
            action = index_to_action[int(action_idx.item())]
            observation, reward, terminated, truncated, _ = env.step(action,opponent_policy=opponent)
            step_reward += reward
            reward = torch.tensor([reward], device=device)
            done = terminated or truncated

            if terminated:
                # The environment adds a shaping term to the terminal reward, so the
                # raw win/loss signal is better inferred from the sign of the reward.
                wins[completed_episodes] = 1 if reward ==1  else 0
                completed_episodes += 1
                next_state_mask = None
                next_state = None

            else:
                legal_actions = env.get_legal_moves()
                legal_indices = [action_to_index[a] for a in legal_actions]
                next_state_mask = torch.full((n_actions,), -float("inf"), device=device).unsqueeze(0)
                next_state_mask[0][legal_indices] = 0.0

                observation = encode_state(observation)
                next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

            # Store the transition in memory
            memory.push(state, action_idx, next_state, reward,next_state_mask)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the policy network)
            optimize_model()

            # Soft update of the target network's weights
            # θ′ ← τ θ + (1 −τ )θ′
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
            target_net.load_state_dict(target_net_state_dict)

            if done:
                rewards[i_episode] = step_reward
                episode_durations.append(t + 1)
                plot_rewards(i_episode,eps)
            
                #EVALUATION PROCESS

                #initialization
                n_matches = 200

                if i_episode==1000:
                    #our benchmark will be the deterministic 1000 episodes policy
                    torch.save(policy_net_state_dict,"old_net.pt")
                    #initialize the test policy
                    old_net=DQN(n_observations, n_actions).to(device)
                    old_net.load_state_dict(torch.load("old_net.pt"))
                    old_net.eval()
                    test_policy = make_opponent_policy(old_net)

                if completed_episodes % 500 == 0 and i_episode>1000:
                    #we can now save a network snapshot here and add it to the opponents batch
                    opp_net=DQN(n_observations, n_actions).to(device)
                    opp_net.load_state_dict(policy_net_state_dict)
                    new_opp = make_opponent_policy(opp_net)
                    opponents[f"Policy {i_episode+1}"] = new_opp

                    print(f"EVALUATION PROCESS: policy at episode 1000 vs policy at episode {i_episode+1}")
                
                    win_ratio, avg_reward, avg_duration = play(test_policy,n_matches=n_matches)
                    eval_episodes.append(i_episode + 1)
                    eval_win_ratios.append(win_ratio)
                    print("RESULTS:")
                    print(f"Win ratio: {win_ratio}, average reward: {avg_reward}, average duration: {avg_duration}")
                    # overwriting the current state on the old network
                    #torch.save(policy_net_state_dict,"old_net.pt")

                break

    print('Complete')
    plot_rewards(None,None,show_result=True)
    plot_win_ratios()
    plt.ioff()
    plt.show()
    torch.save(policy_net_state_dict,"final_policy.pt")