using Plots
include("blackjack.jl")


# Value player | has usable ace | dealer's showing card
policy = zeros((11, 2, 10))
policy[1:end-2, :, :] .= 1 # Sum less than 20 => hits
policy[end-2:end, :, :] .= 0 # Sum == 20 or sum == 21 => stickend

# total sum of returns for all history
returns = zeros(size(policy))
# count of times a state was visited
state_count = zeros(size(policy))

Random.seed!(314)
for _ in 1:500_000
    episode_hist, _ = blackjack(policy)
    episode_hist = reverse(episode_hist, dims=1)
    visited_states = [0 0 0]
    
    G = 0
    for row in eachrow(episode_hist)
        action, player_val, ace_val, dealer_card, _, reward = row
        G = G + reward
        
        player_ix = player_val - 11
        ace_ix = ace_val + 1
        dealer_ix = dealer_card

        state_ix = [player_ix ace_ix dealer_ix]

        # Update if state hasn't been observed and we are inside
        # a valid state
        if !array_in_arrays(state_ix, visited_states)
            player_ix = min(player_ix, 11)
            visited_states = vcat(visited_states, state_ix)
            returns[player_ix, ace_ix, dealer_ix] += G
            state_count[player_ix, ace_ix, dealer_ix] += 1
        end
    end
end

begin
    Vπ = (returns ./ state_count)[begin:end-1, :, :]
    p1 = plot(1:10, 12:21, Vπ[:, 1, :], st=:wireframe, title="Usable ace", zlim=(-1, 1), camera=(20, 60))
    p2 = plot(1:10, 12:21, Vπ[:, 2, :], st=:wireframe, title="No usable ace", zlim=(-1, 1), camera=(20, 60))
    plot(p1, p2, size=(800, 250))
end