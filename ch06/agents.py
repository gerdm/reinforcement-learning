import numpy as np
from numba import njit

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



def run_agent_sarsa(
        ix_init,
        gridworld, n_actions, n_steps,
        epsilon, alpha, gamma, movements,
        seed=314
):
    Q = np.zeros((gridworld.n_states, n_actions))
    
    ix = ix_init
    ix_hist = [ix]
    action = choose_action(ix, Q, epsilon)
    action_hist = [action]
    reward_hist = []

    for n in range(n_steps):
        set_seed(seed + n)
        (r, ix_new, action_new), q_new = sarsa_step(ix, action, Q, epsilon, alpha, gamma, gridworld, movements)
        Q[ix, action] = q_new
        ix, action = ix_new, action_new

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


def run_agent_qlearning(
        ix_init,
        gridworld, n_actions, n_steps,
        epsilon, alpha, gamma, movements,
        seed=314
):
    Q = np.zeros((gridworld.n_states, n_actions))
    
    ix = ix_init
    ix_hist = [ix]
    action_hist = []
    reward_hist = []

    for n in range(n_steps):
        set_seed(seed + n)
        (r, ix_new, action), q_new = qlearning_step(ix, Q, epsilon, alpha, gamma, gridworld, movements)
        Q[ix, action] = q_new
        ix = ix_new

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

