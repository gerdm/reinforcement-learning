import numpy as np
from numba import njit


@njit
def init_buffers_mdp(buffer_size):
    buffer_actions = np.zeros(buffer_size + 1) * np.nan
    buffer_states = np.zeros(buffer_size + 1) * np.nan
    buffer_rewards = np.zeros(buffer_size) * np.nan

    buffers = (buffer_states, buffer_actions, buffer_rewards)
    return buffers


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
def update_buffer(buffer, new_element):
    """
    Update buffer FIFO mode.
    We assume that elements in the buffer are time-ordered
    along the first axis.

    Example:
    buffer = [b1, b2, b3]
    new_element = x
    -> buffer = [b2, b3, x]

    Parameters
    ----------
    buffer: np.array
        (B, ...)
    new_element: np.array
        (...,)
    """
    buffer = np.roll(buffer, -1)
    buffer[-1] = new_element
    return buffer


@njit
def choose_action(self, ix, Q):
    """
    Choose action based on epsilon-greedy
    """
    probas = get_probas(Q[ix], self.epsilon)
    action = np.random.multinomial(1, probas).argmax()
    return action


class TabularAgent:
    def __init__(self, alpha, gamma, epsilon):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def choose_action(self, ix, Q):
        """
        Choose action based on epsilon-greedy
        """
        probas = get_probas(Q[ix], self.epsilon)
        action = np.random.multinomial(1, probas).argmax()
        return action

    
    def unpack(self, Q, states, actions, rewards):
        Q = np.copy(Q)
        # Remove nan elements
        # This happens is buffer is not full and we have reached an end state
        map_take_rewards = ~np.isnan(rewards)
        map_take_SA = ~np.isnan(actions)
    
        states = states[map_take_SA].astype(np.int32)
        actions = actions[map_take_SA].astype(np.int32)
        rewards = rewards[map_take_rewards]

        return Q, states, actions, rewards


class ContextualBanditAgent(TabularAgent):
    def __init__(self, alpha, epsilon):
        super().__init__(alpha, 0.0, epsilon)

    def _update_single(self, Q, states, actions, rewards, end_state_reached):
        Q = np.copy(Q)
        # Remove nan elements
        # This happens is buffer is not full and we have reached an end state
        map_take_rewards = ~np.isnan(rewards)
        map_take_SA = ~np.isnan(actions)
    
        # Return Q if all elements are NaN
        if not map_take_rewards.any():
            return Q
    
        states = states[map_take_SA].astype(np.int32)
        actions = actions[map_take_SA].astype(np.int32)
        rewards = rewards[map_take_rewards]

        state_init = states[0]
        action_init = actions[0]
        target = rewards[0]
    
        td_err =  target - Q[state_init, action_init]
        q_next = Q[state_init, action_init] + self.alpha * td_err
        Q[state_init, action_init] = q_next

        return Q


    def update(self, Q, buffers, end_state_reached):
        buffer_states, buffer_actions, buffer_rewards = buffers
        buffer_size = len(buffer_rewards)
        
        Q = self._update_single(
            Q, buffer_states, buffer_actions, buffer_rewards, end_state_reached
        )
    
        buffers = buffer_states, buffer_actions, buffer_rewards
        return Q, buffers


class MonteCarloAgent(TabularAgent):
    def __init__(self, alpha, gamma, epsilon):
        super().__init__(alpha, gamma, epsilon)

    def _update_single(self, Q, states, actions, rewards):
        Q, states, actions, rewards = self.unpack(Q, states, actions, rewards)

        state_init, action_init = states[0], actions[0]
    
        buffer_size = len(rewards)
        gamma_values = np.power(self.gamma, np.arange(buffer_size))
    
        target = (rewards * gamma_values).sum()
        
        td_err =  target - Q[state_init, action_init]
        q_next = Q[state_init, action_init] + self.alpha * td_err
        Q[state_init, action_init] = q_next
        return Q


    def update(self, Q, buffers, end_state_reached):
        buffer_states, buffer_actions, buffer_rewards = buffers
        buffer_size = len(buffer_rewards)
        
        for b in np.arange(buffer_size):
            Q = self._update_single(
                Q, buffer_states, buffer_actions, buffer_rewards,
            )
    
            # Flush oldest element from buffers
            buffer_actions = update_buffer(buffer_actions, np.nan)
            buffer_states = update_buffer(buffer_states, np.nan)
            buffer_rewards = update_buffer(buffer_rewards, np.nan)
    
            # Stop updating if the number of rewards in the buffer
            # is less than the size of the buffer
            if np.isnan(buffer_rewards).all():
                break
    
        buffers = buffer_states, buffer_actions, buffer_rewards
        return Q, buffers


class NSarsaAgent(TabularAgent):
    def __init__(self, alpha, gamma, epsilon):
        super().__init__(alpha, gamma, epsilon)
        

    def _update_single(self, Q, states, actions, rewards, end_state_reached):
        Q, states, actions, rewards = self.unpack(Q, states, actions, rewards)

        state_init, state_end = states[0], states[-1]
        action_init, action_end = actions[0], actions[-1]
    
        buffer_size = len(rewards)
        gamma_values = np.power(self.gamma, np.arange(buffer_size))
    
        target = (rewards * gamma_values).sum()
        target = target + np.power(self.gamma, buffer_size) * Q[state_end, action_end] * (1 - end_state_reached)
    
        td_err =  target - Q[state_init, action_init]
        q_next = Q[state_init, action_init] + self.alpha * td_err
        Q[state_init, action_init] = q_next

        return Q


    def update(self, Q, buffers, end_state_reached):
        buffer_states, buffer_actions, buffer_rewards = buffers
        buffer_size = len(buffer_rewards)
        
        # 4.1 Update Q-values using the elmements in the buffer + bootstrap
        if not end_state_reached:
            Q = self._update_single(
                Q, buffer_states, buffer_actions, buffer_rewards, end_state_reached
            )
        
        # 4.2. Update the Q-values MC-style, i.e., we do not bootstrap
        else:
            for b in np.arange(buffer_size):
                Q = self._update_single(
                    Q, buffer_states, buffer_actions, buffer_rewards, end_state_reached
                )
        
                # Flush oldest element from buffers
                buffer_actions = update_buffer(buffer_actions, np.nan)
                buffer_states = update_buffer(buffer_states, np.nan)
                buffer_rewards = update_buffer(buffer_rewards, np.nan)
    
                # Stop updating if the number of rewards in the buffer
                # is less than the total reward.
                if np.isnan(buffer_rewards).all():
                    break
    
        buffers = buffer_states, buffer_actions, buffer_rewards
        return Q, buffers
