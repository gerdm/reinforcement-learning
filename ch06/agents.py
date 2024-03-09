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
    mask = qmax == qval

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
    probas = get_probas(Q[ix], epsilon)
    action = np.random.multinomial(1, probas).argmax()
    return action


@njit
def sarsa_step(s, a, Q, epsilon, alpha, gamma, gridworld, movements):
    """
    SARSA step
    :(s, a) -> (r, s_new, a_new):
    """
    # take action 'a', observe 'r' and s_new'
    step = movements[a]
    s_new, r = gridworld.move_and_reward(s, step)
    # Choose 'a_new' based on Q and 's_new'
    a_new = choose_action(s_new, Q, epsilon)

    q_new = Q[s, a] + alpha * (r + gamma * Q[s_new, a_new] - Q[s, a])
    return (r, s_new, a_new), q_new


@njit
def expected_sarsa_step(s, Q, epsilon, alpha, gamma, gridworld, movements):
    """
    :Expected SARSA step:
    s -> a -> (s_new, r_new)
    """
    # choose action 'a' based on Q and 's'
    a = choose_action(s, Q, epsilon)
    step = movements[a]
    # observe 'r' and 's_new'
    s_new, r = gridworld.move_and_reward(s, step)

    # Expected value of Q[s_new, :]
    probas = get_probas(Q[s_new], epsilon)
    q_new = Q[s, a] + alpha * (r + gamma * (probas * Q[s_new, :]).sum() - Q[s, a])
    return (r, s_new, a), q_new


@njit
def qlearning_step(s, Q, epsilon, alpha, gamma, gridworld, movements):
    """
    :Q-learning step:
    s -> a -> (s_new, r_new)
    """
    # choose action 'a' based on Q and 's'
    a = choose_action(s, Q, epsilon)
    step = movements[a]
    # observe 'r' and 's_new'
    s_new, r = gridworld.move_and_reward(s, step)

    q_new = Q[s, a] + alpha * (r + gamma * Q[s_new, :].max() - Q[s, a])
    return (r, s_new, a), q_new


@njit
def double_q_learning_step(s, Qs, epsilon, alpha, gamma, gridworld, movements):
    """
    We have a duplicate Q-value function Qs = [Q1, Q2].
    At each step, we flip a coin to decide which Q-value function to update and provide an estimate
    of the action.  The other Q-value function is used to choose the action.
    """
    Q1, Q2 = Qs
    ...


@njit
def sarsa_step_and_update(
    s, a, Q, epsilon, alpha, gamma, gridworld, movements
):
    Q = Q.copy()
    (r, s_new, a_new), q_new = sarsa_step(s, a, Q, epsilon, alpha, gamma, gridworld, movements)
    Q[s, a] = q_new
    return (r, s_new, a_new), Q


@njit
def expected_sarsa_step_and_update(
    s, a, Q, epsilon, alpha, gamma, gridworld, movements
):
    Q = Q.copy()
    (r, s_new, a), q_new = expected_sarsa_step(s, Q, epsilon, alpha, gamma, gridworld, movements)
    Q[s, a] = q_new
    return (r, s_new, a), Q


@njit
def qlearning_step_and_update(
    s, a_prev, Q, epsilon, alpha, gamma, gridworld, movements
):
    """
    Note: a_prev is not used in the function,
    but it is included for compatibility with the SARSA function
    """
    Q = Q.copy()
    (r, s_new, a), q_new = qlearning_step(s, Q, epsilon, alpha, gamma, gridworld, movements)
    Q[s, a] = q_new
    return (r, s_new, a), Q


def run_agent(
        ix_init,
        gridworld, n_actions, n_steps,
        epsilon, alpha, gamma, movements,
        step_and_update_fn,
        seed=314,
):
    Q = np.zeros((gridworld.n_states, n_actions))

    ix = ix_init
    ix_hist = [ix]
    action = choose_action(ix, Q, epsilon)
    action_hist = [action]
    reward_hist = []

    for n in range(n_steps):
        set_seed(seed + n)
        (r, ix, action), Q = step_and_update_fn(ix, action, Q, epsilon, alpha, gamma, gridworld, movements)

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
    step_and_update_fn,
    seed=314,
):
    """
    Run the agent until it reaches the goal n_episodes times
    and return the total reward for each episode and the final Q-value function
    """
    Q = np.zeros((gridworld.n_states, n_actions))
    ix = ix_init
    action = choose_action(ix, Q, epsilon)

    return_hist = np.zeros(n_episodes)

    t, n = 0, 0
    episode_return = 0.0
    while True:
        set_seed(seed + t)
        (reward, ix, action), Q = step_and_update_fn(ix, action, Q, epsilon, alpha, gamma, gridworld, movements)

        episode_return += reward
        # if gridworld.is_goal(ix) # TODO: refactor to this
        if reward == gridworld.reward_goal:
            return_hist[n] = episode_return
            n = n + 1
            episode_return = 0.0

            if n == n_episodes:
                break

        t = t + 1
    return return_hist, Q



@njit(parallel=True)
def simulations_run_agent_return(
    ix_init, gridworld, n_actions, n_episodes, epsilon, alpha, gamma, movements, step_and_update_fn,
    n_simulations=100, seed=314,
):
    return_hist = np.zeros((n_simulations, n_episodes))
    for s in prange(n_simulations):
        return_hist[s], _ = run_agent_return(
            ix_init, gridworld, n_actions, n_episodes,
            epsilon, alpha, gamma, movements, step_and_update_fn,
            seed=seed + s
        )
    return return_hist