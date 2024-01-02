import numpy as np
from numba import njit, prange

DISCOUNT = 1.0
CARD_MIN = 1
CARD_MAX = 10
PLAY_MINVAL = 4
PLAY_MAXVAL = 21
DECK = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10])


@njit
def clip(a, a_min, a_max):
    return np.minimum(a_max, np.maximum(a, a_min))


@njit
def set_seed(value):
    """
    See: https://github.com/numba/numba/issues/6002
    """
    np.random.seed(value)
    

@njit
def draw_card():
    card = np.random.choice(DECK)
    return card


@njit
def draw_card_uniform():
    card = np.random.choice(np.arange(CARD_MIN, CARD_MAX + 1))
    return card


@njit
def compute_reward(value_cards_player, value_cards_dealer):
    if value_cards_player > 21:
        reward = -1 # lose
    elif value_cards_dealer > 21:
        reward = 1 # win
    elif value_cards_player == value_cards_dealer:
        reward = 0 # draw
    elif value_cards_player < value_cards_dealer:
        reward = -1 # lose
    else:
        reward = 1 # win
    
    return reward


@njit
def init_game():
    value_cards_player = np.array([draw_card(), draw_card()])
    has_usable_ace = (1 in value_cards_player) * 1
    value_cards_player = value_cards_player.sum() + 10 * has_usable_ace
    
    dealers_card = draw_card() # Dealer's one-showing card
    
    return value_cards_player, has_usable_ace, dealers_card


@njit
def init_game_es():
    """
    Initialize the game for the exploring starts method
    """
    # value_cards_player = np.array([draw_card_uniform(), draw_card_uniform()])
    value_cards_player = np.random.choice(np.arange(4, 22))
    has_usable_ace = np.random.choice(np.array([0, 1]))

    if has_usable_ace == 1 and value_cards_player <= 10:
        value_cards_player += 11
    
    dealers_card = draw_card_uniform() # Dealer's one-showing card
    
    return value_cards_player, has_usable_ace, dealers_card


@njit
def update_hand(value_cards):
    """
    Update the value of the hand for either the player
    or the dealer
    """
    new_card = draw_card()
    
    if value_cards == 1: # if dealer starts with an ace
        value_cards = 11 # Make use of the ace
        has_usable_ace = True
    elif (new_card == 1) and (value_cards <= 10):
        has_usable_ace = True
        value_cards = value_cards + 11 # Make use of the ace
    else:
        has_usable_ace = False
        value_cards = value_cards + new_card

    return value_cards, has_usable_ace


@njit
def state_to_ix(value_cards_player, has_usable_ace, dealers_card):
    ix_value_cards = clip(value_cards_player - PLAY_MINVAL, 0, PLAY_MAXVAL - PLAY_MINVAL)
    ix_has_usable_ace = int(has_usable_ace)
    ix_dealers_card = dealers_card - CARD_MIN
    
    ixs = ix_value_cards, ix_has_usable_ace, ix_dealers_card
    return ixs


@njit
def step_player(
    value_cards_player,
    has_usable_ace,
    dealers_card,
    policy
):
    continue_play = True
    # turn into indices
    ixs = state_to_ix(value_cards_player, has_usable_ace, dealers_card)
    ix_value_cards, ix_has_usable_ace, ix_dealers_card = ixs
    proba_actions = policy[ix_value_cards, ix_has_usable_ace, ix_dealers_card]
    action = np.random.multinomial(1, proba_actions).argmax()

    if action == 1:
        value_cards_player, has_usable_ace_new = update_hand(value_cards_player)
        has_usable_ace = has_usable_ace or has_usable_ace_new
        
    if (value_cards_player >= 21) or (action == 0):
        continue_play = False
    
    return value_cards_player, has_usable_ace, action, continue_play


@njit
def dealer_strategy(dealers_card):
    """
    The dealer hits or sticks according to a fixed strategy:
    Stick on any sum of 17 or greater and hit otherwise
    """
    value_cards_dealer = dealers_card
    while value_cards_dealer < 17:
        value_cards_dealer, _ = update_hand(value_cards_dealer)
        
    return value_cards_dealer

    
@njit
def play_single(value_cards_player, has_usable_ace, dealers_card, policy):
    reward = 0
    
    continue_play = True
    while continue_play:
        value_cards_player, has_usable_ace, action, continue_play = step_player(
            value_cards_player, has_usable_ace, dealers_card, policy
        )
    
    value_cards_dealer = dealer_strategy(dealers_card)
    
    reward = compute_reward(value_cards_player, value_cards_dealer)
    return reward, (value_cards_player, value_cards_dealer)


@njit
def play_single_hist(value_cards_player, has_usable_ace, dealers_card, policy):
    """
    Play game and store intermediate steps
    """
    reward = 0
    
    states = [[value_cards_player, has_usable_ace, dealers_card]]
    rewards = []
    actions = []
    
    continue_play = True
    while continue_play:
        value_cards_player, has_usable_ace, action, continue_play = step_player(
            value_cards_player, has_usable_ace, dealers_card, policy
        )
        
        rewards.append(0)
        actions.append(action)
        states.append([value_cards_player, has_usable_ace, dealers_card])
    
    value_cards_dealer = dealer_strategy(dealers_card)
    
    reward = compute_reward(value_cards_player, value_cards_dealer)
    rewards.append(reward)
    actions.append(0) # final dummy action
    
    hist = (
        states,
        actions,
        rewards
    )
    
    return reward, (value_cards_player, value_cards_dealer), hist


@njit
def single_first_visit_mc(policy, exploring_starts=False):    
    if exploring_starts:
        player_value_cards, ace, dealer_card = init_game_es()
    else:
        player_value_cards, ace, dealer_card = init_game()
        
    reward, values, hist = play_single_hist(player_value_cards, ace, dealer_card, policy=policy)
    hist_state, hist_actions, hist_reward = hist
    T = len(hist_state)

    elements = []
    sim_reward = 0
    for t in range(-1, -(T + 1), -1):
        sim_reward = DISCOUNT * sim_reward + hist_reward[t]
        current_state = hist_state[t]
        current_action = hist_actions[t]
        previous_states = hist_state[:t]
        value_cards_player, has_usable_ace, dealers_card = current_state
        
        if value_cards_player > 21:
            # print("Bug")
            continue
        
        first_visit = current_state not in previous_states
        if first_visit:
            ixs = state_to_ix(value_cards_player, has_usable_ace, dealers_card)
                        
            element = (ixs, sim_reward, current_action)
            elements.append(element)
        
    return elements, values, reward


@njit(parallel=True)
def first_visit_mc_eval(policy, n_sims, exploring_starts=False):
    grid_rewards = np.zeros(policy.shape[:-1])
    grid_count = np.zeros(policy.shape[:-1])
    
    for s in prange(n_sims):
        hits, _, _ = single_first_visit_mc(policy, exploring_starts=exploring_starts)
        for element in hits:
            ixs, sim_reward, _ = element
            ix_value_cards, ix_has_usable_ace, ix_dealers_card = ixs
            if (ix_value_cards + PLAY_MINVAL) > 21:
                continue
                        
            grid_rewards[ix_value_cards, ix_has_usable_ace, ix_dealers_card] += sim_reward
            grid_count[ix_value_cards, ix_has_usable_ace, ix_dealers_card] += 1

    
    return grid_rewards, grid_count
