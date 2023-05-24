def initialize():
    global total_demand, low_cost, high_cost, total_stages, num_adv_history, lr, gamma, num_actions
    global action_step, num_stochastic_iter, num_episodes, num_episodes_reset, episode_adv_increase
    global rewards_division_const,replay_buffer_size,prob_break_limit_ln,converge_break, print_step
    global batch_update_size,buffer_play_coefficient, num_cores

    total_demand = 400
    low_cost = 57
    high_cost = 71
    total_stages = 25
    num_adv_history = 3
    lr = 0.000005
    gamma = 1
    num_actions = 20
    action_step = 3
    num_stochastic_iter = 10
    rewards_division_const = 10000

    # episodes for learning the last stage, then for 2nd to last stage 2*numEpisodes. In total:300*numEpisodes
    num_episodes = 2000
    num_episodes_reset = num_episodes
    # increase in num of episodes for each adv in support
    episode_adv_increase = 1000

    replay_buffer_size= 500_000

    prob_break_limit_ln=-0.001
    converge_break=True

    # if details of game after this many episodes need to be printed. None means no printing
    print_step=None
    # size of the batch of games that would be played in each process and then would update nn with other processes resultstogether
    batch_update_size=50
    # proportion of episodes from buffer ( (1-coefficient)*episode will be new games)
    buffer_play_coefficient=0.5

    num_cores=4
