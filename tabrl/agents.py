import numpy as np
from numba import njit, prange


@njit
def set_seed(seed):
    return np.random.seed(seed)


@njit
def get_probas(qval, epsilon):
    """
    Epsilon-greedy probabilities
    """
    qmax = qval.max()
    n_actions = len(qval)
    mask = qmax == qval #  mask for equal values

    if np.all(mask):
        # Random selection
        probas = np.ones(n_actions) / n_actions
    elif sum(mask) > 1:
        # Random selection in unvisited states
        n_equal = sum(mask)
        probas = epsilon * np.ones(n_actions) /  n_actions
        probas = ((1 - epsilon) / n_equal + epsilon / n_actions) * mask +  probas * (1 - mask)
    else:
        # Epsilon-greedy state
        action_max = qval.argmax()
        probas = epsilon * np.ones(n_actions) / n_actions
        probas[action_max] = 1 - epsilon + epsilon / n_actions

    return probas


@njit
def choose_action(ix, Q, epsilon):
    """
    Choose action based on epsilon-greedy
    """
    probas = get_probas(Q[ix], epsilon)
    action = np.random.multinomial(1, probas).argmax()
    return action


@njit
def choose_action_double_q(ix, Qs, epsilon):
    """
    Choose action based on average of Q1 and Q2
    and epsilon-greedy.
    Use only for double-Q learning
    """
    Q1, Q2 = Qs[0], Qs[1]
    Q = (Q1 + Q2) / 2
    action = choose_action(ix, Q, epsilon)
    return action


@njit
def nstep_sarsa_update(states, actions, rewards, Q, alpha, gamma, end_state_reached):
    """
    n-step SARSA update
    """
    Q = np.copy(Q)
    # Remove nan elements
    # This happens is buffer is not filled and we have reached an end state
    map_take_rewards = ~np.isnan(rewards)
    map_take_SA = ~np.isnan(actions)

    # Return Q if all elements are NaN
    if not map_take_rewards.any():
        return Q

    states = states[map_take_SA].astype(np.int32)
    actions = actions[map_take_SA].astype(np.int32)
    rewards = rewards[map_take_rewards]


    state_init, state_end = states[0], states[-1]
    action_init, action_end = actions[0], actions[-1]

    buffer_size = len(rewards)
    gamma_values = np.power(gamma, np.arange(buffer_size))

    target = (rewards * gamma_values).sum()
    target = target + np.power(gamma, buffer_size) * Q[state_end, action_end] * (1 - end_state_reached)

    td_err =  target - Q[state_init, action_init]
    q_next = Q[state_init, action_init] + alpha * td_err
    Q[state_init, action_init] = q_next
    return Q
    

@njit
def sarsa_update(s, a, r, s_next, a_next, Q, alpha, gamma, epsilon):
    """
    SARSA update
    :(s, a) -> (r, s_new, a_new):
    """
    td_err = r + gamma * Q[s_next, a_next] - Q[s, a]
    q_next = Q[s, a] + alpha * td_err
    Q[s, a] = q_next
    return Q


@njit
def expected_sarsa_update(s, a, r, s_next, a_next, Q, alpha, gamma, epsilon):
    """
    Expected SARSA update
    :(s, a) -> (r, s_new, a_new):
    """
    # Expected value of Q[s_next, :]
    td_err = r + gamma * (probas * Q[s_next, :]).sum() - Q[s, a]
    probas = get_probas(Q[s_next], epsilon)
    q_next = Q[s, a] + alpha * td_err
    Q[s, a] = q_next
    return Q


@njit
def qlearning_update(s, a, r, s_next, a_next, Q, alpha, gamma, epsilon):
    """
    s -> a -> (s_new, r)
    """
    td_err = r + gamma * Q[s_next, :].max() - Q[s, a]
    q_next = Q[s, a] + alpha * td_err
    Q[s, a] = q_next
    return Q


@njit
def double_q_learning_update(s, a, r, s_next, a_next, Qs, alpha, gamma, epsilon):
    """
    We have a duplicate Q-value function Qs = [Q1, Q2].
    At each update, we flip a coin to decide which Q-value function to update and provide an estimate
    of the action.  The other Q-value function is used to choose the action.
    """
    iq = np.random.randint(2)
    Q1, Q2 = Qs[iq], Qs[1 - iq]

    # Update Q1 using Q2 to determine the value of the action
    td_err = r + gamma * Q2[s_next, Q1[s_next, :].argmax()] - Q1[s, a]
    q_next = Q1[s, a] + alpha * td_err
    Qs[iq, s, a] = q_next
    return Qs


@njit
def step_and_update_agent(
    Q, s, a, epsilon, alpha, gamma,
    gridworld, movements, update_fn, policy_fn
):
    """
    Parameters
    ---------
    epsilon: (0, 1)
        exploration rate
    alpha: (0,1), float
        learning rate
    gamma: (0,1), float
        discount factor
    """
    # 1. take step, obtain reward and new state
    r, s_next = gridworld.step(s, a, movements)
    
    # 2. Choose 'a_new' based on Q and 's_new'
    a_next = policy_fn(s_next, Q, epsilon)

    # 3. Take an update of the action-value
    Q = Q.copy()
    Q = update_fn(s, a, r, s_next, a_next, Q, alpha, gamma, epsilon)

    return (s_next, a_next, r), Q


def run_agent(
        ix_init,
        gridworld, n_actions, n_steps,
        epsilon, alpha, gamma, movements,
        update_fn, policy_fn=None,
        seed=314, Q=None,
):
    if policy_fn is None:
        policy_fn = choose_action

    if Q is None:
        Q = np.zeros((gridworld.n_states, n_actions))

    ix = ix_init
    ix_hist = [ix]
    # TODO: fix this --- don't use first action, but make it compatible with all agents
    # action = choose_action(ix, Q, epsilon)
    action = 0
    action_hist = [action]
    reward_hist = []

    for n in range(n_steps):
        set_seed(seed + n)

        (ix, action, r), Q = step_and_update_agent(
            Q, ix, action, epsilon, alpha, gamma, gridworld, movements, update_fn, policy_fn
        )

        ix_hist.append(ix)
        action_hist.append(action)
        reward_hist.append(r)
    
    ix_hist = np.array(ix_hist)
    action_hist = np.array(action_hist)
    reward_hist = np.array(reward_hist)

    hist = {
        "ix": ix_hist,
        "action": action_hist,
        "reward": reward_hist,
    }
    return hist, Q


@njit
def run_agent_return(
        ix_init,
        gridworld, n_actions, n_episodes,
        epsilon, alpha, gamma, movements,
        update_fn, policy_fn=None,
        seed=314, Q=None
):
    if policy_fn is None:
        policy_fn = choose_action

    if Q is None:
        Q = np.zeros((gridworld.n_states, n_actions))

    ix = ix_init
    action = choose_action(ix, Q, epsilon)   

    return_hist = np.zeros(n_episodes)
    t, n = 0, 0
    episode_return = 0.0
    while True:
        set_seed(seed + t)
        (ix, action, r), Q = step_and_update_agent(
            Q, ix, action, epsilon, alpha, gamma, gridworld, movements, update_fn, policy_fn
        )

        episode_return += r
        if r == gridworld.reward_goal:
            return_hist[n] = episode_return
            n = n + 1
            episode_return = 0.0

            if n == n_episodes:
                break
        
        t = t + 1
    return return_hist, Q


@njit(parallel=True)
def simulations_run_agent_return(
        ix_init,
        gridworld, n_actions, n_episodes,
        epsilon, alpha, gamma, movements,
        update_fn, policy_fn=None,
        seed=314, n_simulations=100, Q=None
):
    return_hist = np.zeros((n_simulations, n_episodes))
    for s in prange(n_simulations):
        return_hist[s], _ = run_agent_return(
            ix_init, gridworld, n_actions, n_episodes,
            epsilon, alpha, gamma, movements, update_fn, policy_fn,
            Q=Q, seed=seed + s
        )
    return return_hist
