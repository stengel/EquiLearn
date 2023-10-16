from learningBase import ReinforceAlgorithm
import environmentModelBase as model
from neuralNetworkSimple import NNBase
import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np
import matplotlib.pyplot as plt

const95= model.Strategy(model.StrategyType.static, model.const,"const95", firstPrice=95 )
actionStep=3
adv= model.MixedStrategy()
adv._strategies.append(const95)
adv._strategyProbs=torch.ones(1)
game = model.Model(totalDemand = 400, 
               tupleCosts = (57, 71),
              totalStages = 25, adversary=adv, stateAdvHistory=1)

hyperParams=[0.0001, 1, 0]
codeParams=[1, 10000, 1, 1]

neuralNet=NNBase(num_input=game.T+2+game.stateAdvHistory, lr=hyperParams[0],num_actions=18,action_step=actionStep, adv_hist=game.stateAdvHistory)
algorithm = ReinforceAlgorithm(game, neuralNet, numberIterations=2, numberEpisodes=3_000_000, discountFactor =hyperParams[1])


# algorithm.solver(print_step=100_000,options=codeParams,converge_break=True)
algorithm.loadPolicyNet("1,3,[5e-06,1][1, 10000, 1, 1],1683196677")
returns=algorithm.playTrainedAgent(adv,100)
print(returns)
print(np.mean(np.array(returns),axis=1))