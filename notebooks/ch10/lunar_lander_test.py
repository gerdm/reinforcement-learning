import matplotlib.pyplot as plt
import gymnasium as gym
import numpy as np

def phi(s):
    normalization_constants = np.array([
        1.0 / 3.0,     # x-position
        1.0 / 1.5,     # y-position
        1.0 / 4.0,     # x-velocity
        1.0 / 4.0,     # y-velocity
        1.0 / 6.28,    # angle (theta)
        1.0 / 10.0,    # angular velocity
        1.0,           # left leg contact
        1.0            # right leg contact
    ])

    s = s * normalization_constants

    s_inner = np.array([
        s[-1] * s[-2], # contact
        s[0] * s[1], # x/y position
        s[2] * s[3], # x/y velocity,
        s[4] * s[5], # angle and angular velocity
    ])
    
    sprime = np.concatenate((
        np.ones(1),
        s, s ** 2,
        # np.sin(2 * np.pi * s[:-2]),
        # np.cos(2 * np.pi * s[:-2]),
        np.sin(4 * np.pi * s),
        np.cos(4 * np.pi * s),
        s_inner,
    ))
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
eps = 1e-4
W = np.load("weights.npy")
actions = np.arange(4)

env = gym.make("LunarLander-v3", render_mode="human")
observation, info = env.reset(seed=314)
action = eps_greedy_choice(W, observation, eps, actions)

episode_reward = 0.0
reset = False

rewards = []
while not reset:
    observation, reward, terminated, truncated, info = env.step(action)
    episode_reward += reward
    action = eps_greedy_choice(W, observation, eps, actions)
    reset = terminated or truncated
    rewards.append(reward)

print(f"Episode reward: {episode_reward}")
env.close()

plt.plot(np.cumsum(rewards))
plt.title("Cumulative rewards")
plt.savefig("rewards-lunar-lander.png")
