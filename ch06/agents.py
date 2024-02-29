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
        probas = ((1 - epsilon) / n_actions + epsilon) * mask +  probas * (1 - mask)
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

    q_new = Q[s, a] + alpha * (r + gamma * Q[s_new, :].max(axis=-1) - Q[s, a])
    return (r, s_new, a), q_new