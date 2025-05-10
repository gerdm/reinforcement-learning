import gymnasium as gym
import numpy as np

def phi(s):
    sprime = np.concatenate([
        np.ones(1), s, np.sin(s), np.cos(s)
    ])
    return sprime


def action_value(W, s, a):
    w = W[a]
    return w @ phi(s)


def eval_action_value(W, s, actions):
    q_a = np.array([action_value(W, s, ai) for ai in actions])
    return q_a

def choose_max_action(action_value_est):
    vmax = action_value_est.max()
    if np.sum(vmax == action_value_est) == 1:
        return np.argmax(action_value_est)
    else:
        actions = np.arange(len(action_value_est))
        av_sub = actions[action_value_est == vmax]
        return np.random.choice(av_sub)


def eps_greedy_choice(W, s, eps, actions):
    u = np.random.uniform()
    if u < eps:
        a = np.random.choice(4)
    else:
        action_value_estimates = eval_action_value(W, s, actions)
        a = choose_max_action(action_value_estimates)
    return a


n_steps = 1000
eps = 1e-5
W = np.load("weights.npy")
actions = np.arange(4)

env = gym.make("LunarLander-v3", render_mode="human")
observation, info = env.reset(seed=314)
action = eps_greedy_choice(W, observation, eps, actions)

episode_reward = 0.0
reset = False

while not reset:
    observation, reward, terminated, truncated, info = env.step(action)
    episode_reward += reward
    action = eps_greedy_choice(W, observation, eps, actions)
    reset = terminated or truncated

print(f"Episode reward: {episode_reward}")
env.close()
