from learningAgents import ReinforceAlgorithm
from environmentModel import Model, AdversaryModes
from NeuralNetwork import NeuralNetwork
import torch
import torch.nn as nn
from torch.distributions import Categorical


adversaryProbs=torch.zeros(len(AdversaryModes))
adversaryProbs[0]=1
adversaryProbs[1]=0
adversaryProbs[2]=0
advHistoryNum=3
game = Model(totalDemand = 400, 
               tupleCosts = (57, 71),
              totalStages = 25, adversaryProbs=adversaryProbs, advHistoryNum=advHistoryNum)
neuralNet=NeuralNetwork(num_input=3+advHistoryNum)
algorithm = ReinforceAlgorithm(game, neuralNet, numberIterations=1, numberEpisodes=300_000, discountFactor =0.9)

algorithm.solver()
