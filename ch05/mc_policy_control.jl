using Revise
include("blackjack.jl")

using Random
using Plots


"""
Selection action that maximises the state-action-value function
with ties broken randomly. 
"""
function select_action(action_values)
    range_values = 1:length(action_values)
    map_values = maximum(action_values) .== action_values
    choice_values = range_values[map_values]
    return sample(choice_values)
end


"""
Initialise the policy with hitting whenever the sum is less than 20.
and sticking whenever the sum is 20 or 21.
"""
function initialise_policy()
    policy = zeros((11, 2, 10))
    policy[1:end-2, :, :] .= 1 # value of cards less than 20 => hits
    policy[end-2:end, :, :] .= 0 # value of cards ∈ {20,21} => stick
    return policy
end


# ToDo: group this into a function
Random.seed!(314)
policy = initialise_policy()
state_action_size = (2, size(policy)...)
# total sum of backward-cumulative returns
returns = zeros(state_action_size)
# number of times an action-state was visited
count_visit = ones(state_action_size)

num_simulations = 1_000_000
# Decide whether to store the results of each simulation
# policy_hist = zeros(num_simulations, size(policy)...)

for i in 1:num_simulations
    episode_hist, _ = blackjack_exploring_starts(policy)
    episode_hist = reverse(episode_hist, dims=1)
    visited_action_states = [0 0 0 0] # dummy initial state
    
    cumulative_reward = 0
    for step in eachrow(episode_hist)
        action, player_val, ace_val, dealer_card, _, reward = step
        cumulative_reward += reward

        action_ix = action + 1
        player_ix = min(player_val - 11, 11)
        ace_ix = ace_val + 1
        dealer_ix = dealer_card

        action_state_ix = [action_ix player_ix ace_ix dealer_ix]

        # Estimate value function and be greedy with respect to estimated value function
        # if state hasn't been observed and we are inside a valid state
        if !array_in_arrays(action_state_ix, visited_action_states)
            visited_action_states = vcat(visited_action_states, action_state_ix)
            
            returns[action_ix, player_ix, ace_ix, dealer_ix] += cumulative_reward
            count_visit[action_ix, player_ix, ace_ix, dealer_ix] += 1
            
            returns_state_action = returns[:, player_ix, ace_ix, dealer_ix]
            count_state_action = count_visit[:, player_ix, ace_ix, dealer_ix]
            Q_action = returns_state_action ./ max.(count_state_action, 1.0)
            
            # stick(0) or hit(1)
            new_action = select_action(Q_action) - 1
            # new_action = argmax(Q_action) - 1
            policy[player_ix, ace_ix, dealer_ix] = new_action

            # policy_hist[i, ..] = policy
        end
    end
end


begin
    dealer_range = 1:10
    player_range = 12:21
    policy_play = policy[begin:end-1, :, :]

    p1 = heatmap(dealer_range, player_range, policy_play[:, 1, :], title="Usable ace")
    p2 = heatmap(dealer_range, player_range, policy_play[:, 2, :], title="No usable ace")

    plot(p1, p2, size=(800, 300))
end

# Only run if we store the policy hist
@gif for it ∈ 1:num_simulations
    heatmap(policy_hist[it, begin:end-1, 1, :], title="Policy (@it$it). Usable ace")
end every 100



# Total number of visits when we decide to stick
# and we have a usable ace.
# The bottom rows are zeros. This should not
# be the case. 
# ToDo: Fix this.

action = 1 # Stick
usable_ace = true + 1 # Don't have a usable ace
returns_action_ace = returns[action, :, usable_ace, :]
count_action_ace = count_visit[action, :, usable_ace, :]
Q_action_ace = returns_action_ace ./ max.(1.0, count_action_ace)
plot(Q_action_ace[1:end-1, :], st=:wireframe, camera=(20, 60))

pol_values = returns[:, :, usable_ace, :] ./ max.(1.0, count_visit[:, :, usable_ace, :])
argmax(pol_values, dims=1)
size(pol_values)
