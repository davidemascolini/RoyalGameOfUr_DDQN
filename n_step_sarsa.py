import numpy as np
from collections import defaultdict
from tqdm import tqdm


def legal_indices(legal_moves):
    """
    Convert a list of legal (start, destination) moves into their flat action
    indices. The action map is all_actions = [(i, j) for i in range(16) for j in
    range(16)], so index((i, j)) == i * 16 + j (no dict lookup needed).
    """
    return [start * 16 + dest for (start, dest) in legal_moves]


def e_greedy(Q, state, legal_idx, epsilon=0.1):
    """
    Epsilon-greedy action selection restricted to the legal actions of `state`.

    legal_idx: precomputed list of legal action indices for `state`. Selection is
    masked to these directly, so it is equivalent to the previous rejection-loop
    (uniform random among legal actions w.p. epsilon, otherwise the legal action
    with the highest Q value) but without the per-call list/dict allocation or the
    re-sampling loop.
    """
    if np.random.random() < epsilon:
        return int(np.random.choice(legal_idx))
    q = Q[state][legal_idx]
    # Break ties uniformly at random among the best legal actions instead of always
    # taking the lowest index. In an unlearned state every legal action has Q = 0, so
    # np.argmax would deterministically pick the first legal move; that systematic bias
    # makes the greedy half of the policy play a fixed (near-arbitrary) strategy rather
    # than exploring. Random tie-breaking turns those ties into genuine exploration.
    best = np.flatnonzero(q == q.max())
    return int(legal_idx[np.random.choice(best)])





def n_step_sarsa(env, num_episodes,n, discount_factor=1.0, alpha=0.01, epsilon=0.1):
    """
    n-step SARSA algorithm: On-policy TD(n) control. Finds the optimal epsilon-greedy policy.
    
    Args:
        env: Gymnasium environment.
        num_episodes: Number of episodes to run for.
        n: number of steps before updating target
        discount_factor: Gamma discount factor.
        alpha: TD learning rate.
        epsilon: Chance the sample a random action. Float betwen 0 and 1.
    
    Returns:
        A tuple (Q, stats).
        Q is the optimal action-value function, a dictionary mapping state -> action values.
        stats is an EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
    """
    
    nA = env.action_space
    # The final action-value function.
    # A nested dictionary that maps state -> (action -> action-value).
    # Every key is a state, while the value is a numpy array of values per action of that state. 
    all_actions = [(i, j) for i in range(16) for j in range(16)]
    idx_to_action = {idx: a for idx, a in enumerate(all_actions)} 
    action_to_idx = {a: idx for idx, a in enumerate(all_actions)}
    Q = defaultdict(lambda: np.zeros(len(all_actions)))  # 256 values

    tracked_state = (0, 0, 0, 0, 4)
    tracked_action = action_to_idx[(0, 4)]  
    Q_trajectory = np.zeros(num_episodes)
  

        
    # Sum of (shaped) rewards per episode. With potential-based shaping this is no
    # longer a clean 0/1 win flag, so it is kept only as a learning-signal diagnostic.
    episode_rewards = np.zeros(num_episodes)

    # True win flag per episode (1 = P1 won, 0 otherwise), set from the SIGN of the
    # terminal reward. The shaping term added at the terminal step is small relative
    # to the +/-1 win/loss reward, so its sign is unaffected. Use THIS for the
    # win-ratio plot, not episode_rewards.
    episode_wins = np.zeros(num_episodes)

    #episode lengths should give the length of the episode.
    episode_lengths = np.zeros(num_episodes)



    # Loop over the episodes
    for i in tqdm(range(num_episodes)):
        #new episode, we reset to the initial state
        obs, info = env.reset()
        state = tuple(obs)
        T = np.inf #we don't know yet how much the episode will last
        t=-1
        #let's inizialize an action based on the greedy policy (restricted to legal moves)
        legal_idx = legal_indices(env.get_legal_moves())
        action_idx = e_greedy(Q, state, legal_idx, epsilon)
        action = idx_to_action[action_idx]
        #in order to update the target based on the (s,a) pairs we saw n-1 step before we have to create two buffers
        recent_states = [None] * (n+1)
        recent_actions = [None] * (n+1)
        # rewards buffer: recent_rewards[k % (n+1)] holds R_k (reward received entering step k)
        recent_rewards = [0.0] * (n+1)
        #we save the first (s,a) pair
        recent_states[0]= state
        recent_actions[0] = action_idx
        next_state = state          # so the bootstrap term is well-defined before the first step
        next_action_idx = action_idx
        while True:
            t+=1 # t=0,1,2... keep track of the current state
            if t<T:
                # observe S{t+1}, R{t+1} and store them in the buffers
                next_obs, reward, done, truncated, info = env.step(action)
                next_state = tuple(next_obs)                 # tuple it BEFORE using it as a dict key
                recent_states[(t+1) % (n+1)] = next_state
                recent_rewards[(t+1) % (n+1)] = reward
                episode_rewards[i] += reward
                episode_lengths[i]+= 1
                if done: # if terminal, the next iteration will be the last one (we no longer enter this branch)
                    T=t+1
                    next_action_idx = None
                    # record the true win flag from the sign of the terminal reward
                    episode_wins[i] = 1.0 if reward > 0 else 0.0
                else:    #otherwise select the next action (from the NEXT state) and store it in the buffer
                    # reuse the legal moves the env already computed for the next state
                    next_legal_idx = legal_indices(info["legal_actions"])
                    next_action_idx = e_greedy(Q, next_state, next_legal_idx, epsilon)
                    recent_actions[(t+1) % (n+1)] = next_action_idx
                state = next_state
                action_idx = next_action_idx
                action = idx_to_action[action_idx] if action_idx is not None else None
            # we want know to update the (s,a) we saw n-1 steps ago, if we took enough steps (e.g. if t-(n-1)>=0)
            tau = t-(n-1) # our indicator for the current updated pair
            if tau>=0:
                # we store in G all the rewards (discounted) from tau to tau+n
                G = sum(recent_rewards[j % (n+1)] * discount_factor**(j-tau-1) for j in range(tau+1,min(tau+n,T)+1))
                # if we're not in the last update we bootstrap off (S_{tau+n}, A_{tau+n})
                if (tau+n) < T:
                    boot_state  = recent_states[(tau+n) % (n+1)]
                    boot_action = recent_actions[(tau+n) % (n+1)]
                    G = G + discount_factor**n * Q[boot_state][boot_action]
                S_tau, A_tau = recent_states[tau % (n+1)], recent_actions[tau % (n+1)]
                # finally updating the target Q(S_{tau},A_{tau})
                Q[S_tau][A_tau] += alpha * (G - Q[S_tau][A_tau])
            if tau==T-1: #our episode is over
                break
        Q_trajectory[i] = Q[tracked_state][tracked_action]

    
        
            
    return Q_trajectory, episode_rewards, episode_wins, episode_lengths
