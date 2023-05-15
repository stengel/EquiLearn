def initialize():
    global totalDemand,lowCost,highCost,totalStages,adversaryHistroy,lr,gamma,numActions,actionStep,numStochasticIter,numEpisodes,numEpisodesReset, episodeIncreaseAdv
    totalDemand = 400
    lowCost = 57
    highCost = 71
    totalStages = 25
    adversaryHistroy = 3
    lr = 0.000005
    gamma = 1
    numActions = 20
    actionStep = 3
    numStochasticIter = 10

    # episodes for learning the last stage, then for 2nd to last stage 2*numEpisodes. In total:300*numEpisodes
    numEpisodes = 3000
    numEpisodesReset = numEpisodes
    # increase in num of episodes for each adv in support
    episodeIncreaseAdv = 1000