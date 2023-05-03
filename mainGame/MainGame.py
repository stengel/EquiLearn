from enum import Enum
import torch
from torch.distributions import Categorical
import torch.nn as nn
import numpy as np
from environmentModelBase import Model, AdversaryModes
from learningBase import ReinforceAlgorithm

totalDemand = 400
lowCost = 57
highCost = 71
totalStages = 25


class MainGame():
    """
    strategies play against each other and dill the matrix of payoff, then the equilibria would be computed using Lemke algorithm
    """

    _strategies = []
    _matrix_A = None
    _matrix_B = None

    def __init__(self) -> None:
        pass

    def reset_matrix(self):
        self._matrix_A = np.zeros(
            (len(self._strategies), len(self._strategies)))
        self._matrix_B = np.zeros(
            (len(self._strategies), len(self._strategies)))

    def fill_matrix(self):
        self.reset_matrix()
        n = len(self._strategies)
        for low in range(n):
            for high in range(n):
                self.update_matrix_entry(low, high)
        pass

    def update_matrix_entry(self, lowIndex, highIndex):
        stratL = self._strategies[lowIndex]
        stratH = self._strategies[highIndex]
        stratL.reset()
        stratH.reset()
        [self._matrix_A[lowIndex][highIndex], self._matrix_B[lowIndex][highIndex]] = stratL.play(stratH)
        pass

    def write_all_matrix(self):
        print(self._matrix_A)
        print(self._matrix_B)


    def compute_equilibria(self):
        pass


class Strategy():
    """
    strategies can be static or they can come from neural nets or Q-tables.

    """
    _type = None
    _name = None
    _neural_net = None
    _q_table = None
    _static_index = None

    def __init__(self, strategyType, name, staticIndex=None, neuralNet=None, history_num=0, qTable=None) -> None:
        """
        Based on the type of strategy, the index or neuralnet or q-table should be given as input
        """
        self._type = strategyType
        self._name = name
        self._neural_net = neuralNet
        self._neural_net_history = history_num
        self._static_index = staticIndex
        self._q_table = qTable
        self.stochastic_iters = 1

    def reset(self):
        pass

    def play(self, adversary):
        """ 
        self is low cost and gets adversary(high cost) strategy and they play
        output: tuple (payoff of low cost, payoff of high cost)
        """
        if self._type == StrategyType.static and adversary._type == StrategyType.static:
            return self.play_static_static(adversary._static_index)
        elif self._type == StrategyType.static and adversary._type == StrategyType.neural_net:
            returns = torch.zeros(2, self.stochastic_iters)
            for iter in range(self.stochastic_iters):

                game = Model(totalDemand=400,
                             tupleCosts=(highCost, lowCost),
                             totalStages=25, adversaryProbs=None, advHistoryNum=adversary._neural_net_history)
                algorithm = ReinforceAlgorithm(
                    game, neuralNet=None, numberIterations=3, numberEpisodes=1_000_000, discountFactor=1)

                algorithm.policy = adversary._neural_net.policy
                returns[1][iter], returns[0][iter] = algorithm.playTrainedAgent(
                    AdversaryModes(self._static_index), 1)

            return [torch.mean(returns[0]), torch.mean(returns[1])]

        elif self._type == StrategyType.neural_net and adversary._type == StrategyType.static:
            returns = torch.zeros(2, self.stochastic_iters)
            for iter in range(self.stochastic_iters):
                game = Model(totalDemand=400,
                             tupleCosts=(lowCost, highCost),
                             totalStages=25, adversaryProbs=None, advHistoryNum=self._neural_net_history)
                algorithm = ReinforceAlgorithm(
                    game, neuralNet=None, numberIterations=3, numberEpisodes=1_000_000, discountFactor=1)

                algorithm.policy = self._neural_net.policy
                returns[0][iter], returns[1][iter] = algorithm.playTrainedAgent(
                    AdversaryModes(adversary._static_index), 1)

            return [torch.mean(returns[0]), torch.mean(returns[1])]

        elif self._type == StrategyType.neural_net and adversary._type == StrategyType.neural_net:
            returns = torch.zeros(2, self.stochastic_iters)
            for iter in range(self.stochastic_iters):
                advprobs = torch.zeros(len(AdversaryModes))
                advprobs[10] = 1
                game = Model(totalDemand=400, tupleCosts=(lowCost, highCost), totalStages=25, adversaryProbs=advprobs,
                             advHistoryNum=self._neural_net_history, advNN=adversary._neural_net)
                algorithm = ReinforceAlgorithm(
                    game, neuralNet=None, numberIterations=3, numberEpisodes=1_000_000, discountFactor=1)
                algorithm.policy = self._neural_net.policy

                state, reward, done = game.reset()
                while not done:
                    prevState = state
                    normPrevState = game.normalizeState(
                        prevState)
                    probs = self._neural_net.policy(normPrevState)
                    distAction = Categorical(probs)
                    action = distAction.sample()

                    state, reward, done = game.step(
                        prevState, action.item())
                returns[0][iter], returns[1][iter] = sum(
                    game.profit[0]), sum(game.profit[1])

            return [torch.mean(returns[0]), torch.mean(returns[1])]
        else:
            print("play is not implemented!")

    def play_static_static(self, adversaryIndex):
        """
        self is the low cost player
        """

        selfIndex = self._static_index

        lProbs = torch.zeros(len(AdversaryModes))
        lProbs[selfIndex] = 1

        hProbs = torch.zeros(len(AdversaryModes))
        hProbs[adversaryIndex] = 1

        lGame = Model(totalDemand=400,
                      tupleCosts=(highCost, lowCost),
                      totalStages=25, adversaryProbs=lProbs, advHistoryNum=0)
        hGame = Model(totalDemand=400,
                      tupleCosts=(lowCost, highCost),
                      totalStages=25, adversaryProbs=hProbs, advHistoryNum=0)
        lGame.reset()
        hGame.reset()

        for i in range(totalStages):
            lAction = lGame.adversaryChoosePrice()
            hAction = hGame.adversaryChoosePrice()
            lGame.updatePricesProfitDemand([hAction, lAction])
            hGame.updatePricesProfitDemand([lAction, hAction])
            lGame.stage += 1
            hGame.stage += 1

            # print("\nl: ", lAction, " h: ", hAction, "\nprofits: ", hGame.profit,"\ndemand: ", hGame.demandPotential,"\nprices:",hGame.prices)
        profits = np.array(hGame.profit)
        returns = profits.sum(axis=1)
        return returns


class StrategyType(Enum):
    static = 0
    q_table = 1
    neural_net = 2
